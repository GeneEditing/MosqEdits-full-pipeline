#!/usr/bin/env python3
"""Predict and rank PAM-free guides for the white gene.

Each candidate is scored by every supplied 24-nt CRISPRon-PAMfree model.
The prediction values are averaged per candidate, then candidates are sorted
by the averaged predicted efficiency and the top N guides are written to TSV.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf


INPUT_LENGTH = 24
SPACER_LENGTH = 20
ONEHOT_DEPTH = 4
VALID_BASES = frozenset("ACGT")
BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}

# IUPAC nucleotide ambiguity codes, used to match PAM patterns like NGG/NGTT/TTTV.
IUPAC_CODES = {
    "A": "A", "C": "C", "G": "G", "T": "T", "U": "T",
    "R": "AG", "Y": "CT", "S": "GC", "W": "AT", "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT",
}


REQUIRED_COLUMNS = {
    "guide_id",
    "record_id",
    "chrom",
    "strand",
    "upstream4",
    "spacer",
    "pam",
    "input_24nt",
    "target_with_pam",
    "local_spacer_start",
    "local_spacer_end",
    "local_pam_start",
    "local_pam_end",
    "genomic_spacer_start",
    "genomic_spacer_end",
    "genomic_pam_start",
    "genomic_pam_end",
}


OUTPUT_COLUMNS = [
    "rank",
    "guide_id",
    "gRNA_20nt",
    "pam",
    "predicted_efficiency",
    "chrom",
    "strand",
    "genomic_spacer_start",
    "genomic_spacer_end",
    "genomic_pam_start",
    "genomic_pam_end",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Score PAM-free white-gene candidates with an ensemble of 24-nt "
            "CRISPRon-PAMfree models and write the highest-ranked guides."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Candidate TSV produced by extract_pamfree_24nt.py.",
    )
    parser.add_argument(
        "--pam",
        default="NGTT",
        type=str.upper,
        help=(
            "IUPAC PAM pattern used to generate the candidates, e.g. NGG, "
            "NGTT, TTTV. Default: NGTT."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        type=Path,
        help="Fold model files; predictions are averaged per candidate.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of highest-scoring guides to output (default: 20).",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def matches_pam(pam, pattern):
    return len(pam) == len(pattern) and all(
        base in IUPAC_CODES[code] for base, code in zip(pam, pattern)
    )


def validate_pam_pattern(pattern):
    invalid = set(pattern).difference(IUPAC_CODES)
    if invalid:
        raise ValueError(
            f"Invalid IUPAC base(s) in PAM pattern {pattern!r}: "
            f"{sorted(invalid)}"
        )
    return pattern


def read_candidates(path, pam_pattern):
    records = []

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")

        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Missing columns in {path}: {', '.join(sorted(missing))}"
            )

        seen_guide_ids = set()
        for line_number, row in enumerate(reader, start=2):
            guide_id = (row.get("guide_id") or "").strip()
            upstream = (row.get("upstream4") or "").strip().upper()
            spacer = (row.get("spacer") or "").strip().upper()
            pam = (row.get("pam") or "").strip().upper()
            input_24nt = (row.get("input_24nt") or "").strip().upper()
            target_with_pam = (
                row.get("target_with_pam") or ""
            ).strip().upper()

            if not guide_id:
                raise ValueError(f"{path}:{line_number}: empty guide_id")
            if guide_id in seen_guide_ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate guide_id {guide_id}"
                )
            seen_guide_ids.add(guide_id)

            if len(upstream) != 4:
                raise ValueError(
                    f"{path}:{line_number}: upstream4 must be 4 nt"
                )
            if len(spacer) != SPACER_LENGTH:
                raise ValueError(
                    f"{path}:{line_number}: spacer must be 20 nt"
                )
            if not matches_pam(pam, pam_pattern):
                raise ValueError(
                    f"{path}:{line_number}: PAM {pam!r} does not match "
                    f"pattern {pam_pattern}"
                )
            if input_24nt != upstream + spacer:
                raise ValueError(
                    f"{path}:{line_number}: input_24nt does not equal "
                    "upstream4 + spacer"
                )
            if len(input_24nt) != INPUT_LENGTH:
                raise ValueError(
                    f"{path}:{line_number}: input_24nt must be 24 nt"
                )
            if target_with_pam != spacer + pam:
                raise ValueError(
                    f"{path}:{line_number}: target_with_pam does not equal "
                    "spacer + pam"
                )

            combined = upstream + spacer + pam
            if not set(combined).issubset(VALID_BASES):
                invalid = "".join(
                    sorted(set(combined).difference(VALID_BASES))
                )
                raise ValueError(
                    f"{path}:{line_number}: invalid nucleotide(s) {invalid!r}"
                )

            row["upstream4"] = upstream
            row["spacer"] = spacer
            row["pam"] = pam
            row["input_24nt"] = input_24nt
            row["target_with_pam"] = target_with_pam
            row["source_line"] = line_number
            records.append(row)

    if not records:
        raise ValueError(f"No candidate guides found in {path}")

    return records


def onehot_encode(records):
    x = np.zeros(
        (len(records), INPUT_LENGTH, ONEHOT_DEPTH), dtype=np.float32
    )

    for row_index, record in enumerate(records):
        for position, base in enumerate(record["input_24nt"]):
            x[row_index, position, BASE_TO_INDEX[base]] = 1.0

    return x


def normalized_input_shape(model):
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        if len(input_shape) != 1:
            raise ValueError(
                f"Expected a single-input model, found {len(input_shape)} inputs"
            )
        input_shape = input_shape[0]
    return tuple(input_shape)


def predict_with_model(model_path, x, batch_size):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(str(model_path), compile=False)
    expected_shape = (None, INPUT_LENGTH, ONEHOT_DEPTH)
    actual_shape = normalized_input_shape(model)
    if actual_shape != expected_shape:
        raise ValueError(
            f"Model {model_path} has input shape {actual_shape}; "
            f"expected {expected_shape}"
        )

    prediction = model.predict(
        x,
        batch_size=batch_size,
        verbose=0,
    )
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)

    if prediction.shape[0] != x.shape[0]:
        raise ValueError(
            f"Model {model_path} returned {prediction.shape[0]} predictions "
            f"for {x.shape[0]} candidates"
        )
    if not np.all(np.isfinite(prediction)):
        raise ValueError(f"Model {model_path} returned non-finite predictions")

    return prediction


def write_top_guides(path, ranked_records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(ranked_records)


def main():
    args = parse_args()
    if args.top < 1:
        raise ValueError("--top must be at least 1")
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    pam_pattern = validate_pam_pattern(args.pam)
    records = read_candidates(args.input, pam_pattern)
    x = onehot_encode(records)

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Candidate table: {args.input}")
    print(f"Candidates: {len(records)}")
    print(f"Input shape: {x.shape}")
    print(f"Models: {len(args.models)}")

    predictions = []
    for model_index, model_path in enumerate(args.models, start=1):
        print(f"Predicting with model {model_index}: {model_path}")
        predictions.append(
            predict_with_model(model_path, x, args.batch_size)
        )

    # Paper-style ensemble aggregation: average model predictions for each
    # candidate before ranking candidates by predicted efficiency.
    prediction_matrix = np.stack(predictions, axis=0)
    ensemble_prediction = np.mean(prediction_matrix, axis=0)

    for record, prediction in zip(records, ensemble_prediction):
        record["gRNA_20nt"] = record["spacer"]
        record["predicted_efficiency"] = repr(float(prediction))

    ranked = sorted(
        records,
        key=lambda record: (
            -float(record["predicted_efficiency"]),
            record["guide_id"],
        ),
    )
    top_records = ranked[: min(args.top, len(ranked))]

    for rank, record in enumerate(top_records, start=1):
        record["rank"] = rank

    write_top_guides(args.output, top_records)

    print(f"Top guides requested: {args.top}")
    print(f"Top guides written: {len(top_records)}")
    print(f"Output: {args.output}")
    print("Top predicted efficiencies:")
    for record in top_records:
        print(
            f"  {record['rank']:>2}  {record['guide_id']}  "
            f"{record['gRNA_20nt']}  {record['pam']}  "
            f"{float(record['predicted_efficiency']):.6f}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
