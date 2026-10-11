#!/usr/bin/env python3
"""Create a compact Top-20 NGG guide table from official CRISPRon output.

The official get_30mers_from_fa.py encodes local coordinates in each ID:

    <record_id>_p_<position>  position = plus-strand spacer start (1-based)
    <record_id>_m_<position>  position = minus-strand PAM start (1-based)

The 30mer is normalized to guide orientation:

    upstream4 + spacer20 + PAM3 + downstream3
"""

import argparse
import csv
import re
import sys
from pathlib import Path


SOURCE_LENGTH = 30
SPACER_START = 4
SPACER_END = 24
PAM_START = 24
PAM_END = 27
VALID_BASES = frozenset("ACGT")


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
            "Sort official CRISPRon NGG predictions by efficiency and write "
            "a compact Top-20 table with genomic coordinates."
        )
    )
    parser.add_argument(
        "--crispron-csv",
        required=True,
        type=Path,
        help="Official CRISPRon crispron.csv output.",
    )
    parser.add_argument(
        "--fasta",
        required=True,
        type=Path,
        help="The same FASTA file supplied to CRISPRon.sh.",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_fasta_metadata(path):
    """Read record lengths and optional chr:start-end coordinates."""
    records = {}
    header = None
    sequence_parts = []

    def save_record(current_header, parts):
        if current_header is None:
            return
        record_id = current_header.split()[0]
        if record_id in records:
            raise ValueError(f"Duplicate FASTA record ID: {record_id}")

        sequence = "".join(parts).upper()
        if not sequence:
            raise ValueError(f"Empty sequence for FASTA record {record_id}")
        invalid = set(sequence).difference(VALID_BASES.union({"N"}))
        if invalid:
            raise ValueError(
                f"Unsupported base(s) in FASTA record {record_id}: "
                f"{sorted(invalid)}"
            )

        coordinate_match = re.match(r"^(.+):(\d+)-(\d+)$", record_id)
        if coordinate_match:
            chrom = coordinate_match.group(1)
            region_start = int(coordinate_match.group(2))
            region_end = int(coordinate_match.group(3))
            if region_start > region_end:
                raise ValueError(f"Invalid FASTA coordinate range: {record_id}")
        else:
            chrom = record_id
            region_start = 1
            region_end = len(sequence)

        records[record_id] = {
            "chrom": chrom,
            "region_start": region_start,
            "region_end": region_end,
            "sequence_length": len(sequence),
        }

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                save_record(header, sequence_parts)
                header = line[1:].strip()
                sequence_parts = []
                if not header:
                    raise ValueError(f"{path}:{line_number}: empty FASTA header")
            else:
                if header is None:
                    raise ValueError(
                        f"{path}:{line_number}: sequence before FASTA header"
                    )
                sequence_parts.append(line)

    save_record(header, sequence_parts)
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def parse_crispron_id(guide_id, fasta_records):
    """Split an official CRISPRon ID without assuming record IDs lack '_' characters."""
    match = re.match(r"^(.+)_([pm])_(\d+)$", guide_id)
    if not match:
        raise ValueError(f"Unrecognized CRISPRon ID: {guide_id}")

    record_id = match.group(1)
    orientation = match.group(2)
    position = int(match.group(3))

    if record_id not in fasta_records:
        raise ValueError(
            f"CRISPRon ID {guide_id} refers to FASTA record {record_id!r}, "
            "which was not found"
        )

    return record_id, orientation, position


def local_to_genomic(local_position, region_start):
    return region_start + local_position - 1


def recover_coordinates(guide_id, fasta_records):
    record_id, orientation, position = parse_crispron_id(
        guide_id, fasta_records
    )
    metadata = fasta_records[record_id]

    if orientation == "p":
        strand = "+"
        local_spacer_start = position
        local_spacer_end = position + 19
        local_pam_start = position + 20
        local_pam_end = position + 22
    else:
        strand = "-"
        # For minus-strand IDs, the suffix is the ascending-reference PAM
        # start. The spacer lies immediately to its right on the reference.
        local_pam_start = position
        local_pam_end = position + 2
        local_spacer_start = position + 3
        local_spacer_end = position + 22

    sequence_length = metadata["sequence_length"]
    for name, value in (
        ("local_spacer_start", local_spacer_start),
        ("local_spacer_end", local_spacer_end),
        ("local_pam_start", local_pam_start),
        ("local_pam_end", local_pam_end),
    ):
        if value < 1 or value > sequence_length:
            raise ValueError(
                f"Coordinate {name}={value} from {guide_id} is outside "
                f"FASTA record length {sequence_length}"
            )

    region_start = metadata["region_start"]
    return {
        "chrom": metadata["chrom"],
        "strand": strand,
        "genomic_spacer_start": local_to_genomic(
            local_spacer_start, region_start
        ),
        "genomic_spacer_end": local_to_genomic(
            local_spacer_end, region_start
        ),
        "genomic_pam_start": local_to_genomic(
            local_pam_start, region_start
        ),
        "genomic_pam_end": local_to_genomic(
            local_pam_end, region_start
        ),
    }


def read_crispron_predictions(path, fasta_records):
    records = []

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {path}")

        required = {"ID", "30mer", "CRISPRon"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Missing columns in {path}: {', '.join(sorted(missing))}"
            )

        seen_ids = set()
        for line_number, row in enumerate(reader, start=2):
            guide_id = (row.get("ID") or "").strip()
            sequence_30nt = (row.get("30mer") or "").strip().upper()

            if not guide_id:
                raise ValueError(f"{path}:{line_number}: empty ID")
            if guide_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate ID {guide_id}")
            seen_ids.add(guide_id)

            if len(sequence_30nt) != SOURCE_LENGTH:
                raise ValueError(
                    f"{path}:{line_number}: expected 30mer, found "
                    f"{len(sequence_30nt)} nt"
                )
            if not set(sequence_30nt).issubset(VALID_BASES):
                raise ValueError(
                    f"{path}:{line_number}: non-ACGT base in 30mer"
                )

            spacer = sequence_30nt[SPACER_START:SPACER_END]
            pam = sequence_30nt[PAM_START:PAM_END]
            if len(spacer) != 20:
                raise RuntimeError(f"Spacer extraction failed for {guide_id}")
            if len(pam) != 3 or pam[1:] != "GG":
                raise ValueError(
                    f"{path}:{line_number}: expected NGG PAM, found {pam}"
                )

            try:
                efficiency = float(row["CRISPRon"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid CRISPRon value "
                    f"{row.get('CRISPRon')!r}"
                ) from exc

            if not (efficiency == efficiency and abs(efficiency) != float("inf")):
                raise ValueError(
                    f"{path}:{line_number}: non-finite CRISPRon value"
                )

            record = {
                "guide_id": guide_id,
                "gRNA_20nt": spacer,
                "pam": pam,
                "predicted_efficiency": efficiency,
            }
            record.update(recover_coordinates(guide_id, fasta_records))
            records.append(record)

    if not records:
        raise ValueError(f"No CRISPRon predictions found in {path}")
    return records


def write_top_records(path, records):
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
        writer.writerows(records)


def main():
    args = parse_args()
    if args.top < 1:
        raise ValueError("--top must be at least 1")
    if not args.crispron_csv.exists():
        raise FileNotFoundError(args.crispron_csv)
    if not args.fasta.exists():
        raise FileNotFoundError(args.fasta)

    fasta_records = read_fasta_metadata(args.fasta)
    predictions = read_crispron_predictions(
        args.crispron_csv, fasta_records
    )

    ranked = sorted(
        predictions,
        key=lambda record: (
            -record["predicted_efficiency"],
            record["guide_id"],
        ),
    )
    top_records = ranked[: min(args.top, len(ranked))]

    for rank, record in enumerate(top_records, start=1):
        record["rank"] = rank
        record["predicted_efficiency"] = repr(
            float(record["predicted_efficiency"])
        )

    write_top_records(args.output, top_records)

    print(f"CRISPRon predictions: {len(predictions)}")
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
