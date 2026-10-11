"""Inspect interpretable mismatch and GC features without training a model.

The current project data representation is used as-is:
  - sgRNA[:20] is the stored spacer/protospacer.
  - sgRNA[20:] is the three-base PAM field (all current sgRNAs are NGG).
  - Position 1..20 means left-to-right in the stored 23-nt strings.

The repository does not explicitly document a biological 5'/3' strand
orientation or PAM-proximal numbering.  This script therefore reports the
data-string convention above rather than silently assigning one.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASES = "ACGT"
SEQ_LEN = 23
PROTOSPACER_LEN = 20
PAM_LEN = 3

# Complete, non-overlapping regions in data-string positions 1..20.
REGIONS = {
    "pam_distal_mismatch_count": (0, 7),
    "middle_mismatch_count": (7, 14),
    "pam_proximal_mismatch_count": (14, 20),
}

MISMATCH_TYPES = [
    f"{source}>{target}"
    for source in BASES
    for target in BASES
    if source != target
]

PAM_FEATURE_NAMES = [
    f"pam_pos{position}_{base}"
    for position in range(1, 4)
    for base in BASES
]


def _validate_sequence(sequence: str, name: str) -> None:
    if len(sequence) != SEQ_LEN:
        raise ValueError(f"{name} must have length 23, got {len(sequence)}")
    invalid = sorted(set(sequence) - set(BASES))
    if invalid:
        raise ValueError(f"{name} contains invalid bases: {invalid}")


def _max_consecutive(values: List[int]) -> int:
    maximum = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


def calculate_pam_features(target_pam: str) -> Tuple[np.ndarray, str, List[List[int]]]:
    """Encode the candidate off-target DNA PAM as three 4-way one-hot vectors."""
    pam = str(target_pam).strip().upper()
    if len(pam) != PAM_LEN:
        raise ValueError(f"target PAM must have length 3, got {len(pam)}")
    invalid = sorted(set(pam) - set(BASES))
    if invalid:
        raise ValueError(f"target PAM contains invalid bases: {invalid}")

    one_hot = [[int(base == pam[position]) for base in BASES] for position in range(PAM_LEN)]
    vector = np.asarray([value for position in one_hot for value in position], dtype=np.float32)
    return vector, pam, one_hot


def get_feature_names() -> List[str]:
    """Return names in exactly the same order as the handcrafted vector."""
    names = (
        [f"mismatch_pos_{position}" for position in range(1, PROTOSPACER_LEN + 1)]
        + ["total_mismatch_count"]
        + list(REGIONS)
        + [f"mismatch_type_{name}" for name in MISMATCH_TYPES]
        + ["guide_GC_content", "target_GC_content"]
        + [f"guide_gc_pos_{position}" for position in range(1, PROTOSPACER_LEN + 1)]
        + [f"target_gc_pos_{position}" for position in range(1, PROTOSPACER_LEN + 1)]
        + [
            "max_consecutive_mismatches",
            "first_mismatch_position",
            "last_mismatch_position",
        ]
        + PAM_FEATURE_NAMES
    )
    return names


def build_handcrafted_features(
    sgrna: str, off_target: str
) -> Tuple[Dict[str, object], np.ndarray]:
    """Build the original 81 features followed by 12 target-PAM features."""
    guide = str(sgrna).strip().upper()
    target = str(off_target).strip().upper()
    _validate_sequence(guide, "sgRNA")
    _validate_sequence(target, "off_target")

    guide_proto = guide[:PROTOSPACER_LEN]
    target_proto = target[:PROTOSPACER_LEN]
    mismatch_vector = [int(g != t) for g, t in zip(guide_proto, target_proto)]
    full_mismatch_count = sum(g != t for g, t in zip(guide, target))
    first = next((index + 1 for index, value in enumerate(mismatch_vector) if value), 0)
    last = next(
        (index + 1 for index in range(PROTOSPACER_LEN - 1, -1, -1) if mismatch_vector[index]),
        0,
    )
    mismatch_types = Counter(
        f"{guide_base}>{target_base}"
        for guide_base, target_base in zip(guide_proto, target_proto)
        if guide_base != target_base
    )
    regional = {
        name: sum(mismatch_vector[start:end])
        for name, (start, end) in REGIONS.items()
    }
    regional_features = {
        name: regional[name] / (end - start)
        for name, (start, end) in REGIONS.items()
    }
    protospacer_mismatch_count = sum(mismatch_vector)
    guide_gc = [int(base in "GC") for base in guide_proto]
    target_gc = [int(base in "GC") for base in target_proto]
    guide_gc_content = sum(guide_gc) / PROTOSPACER_LEN
    target_gc_content = sum(target_gc) / PROTOSPACER_LEN
    pattern_features = [
        _max_consecutive(mismatch_vector),
        first / PROTOSPACER_LEN if first else 0.0,
        last / PROTOSPACER_LEN if last else 0.0,
    ]
    pam_vector, target_pam, pam_one_hot = calculate_pam_features(target[-PAM_LEN:])

    vector = np.asarray(
        mismatch_vector
        + [protospacer_mismatch_count / PROTOSPACER_LEN]
        + list(regional_features.values())
        + [mismatch_types[name] / PROTOSPACER_LEN for name in MISMATCH_TYPES]
        + [guide_gc_content, target_gc_content]
        + guide_gc
        + target_gc
        + [
            pattern_features[0] / PROTOSPACER_LEN,
            pattern_features[1],
            pattern_features[2],
        ],
        dtype=np.float32,
    )
    vector = np.concatenate([vector, pam_vector]).astype(np.float32, copy=False)
    details: Dict[str, object] = {
        "protospacer": guide_proto,
        "target_protospacer": target_proto,
        "target_pam": target_pam,
        "recomputed_mismatch_count": full_mismatch_count,
        "protospacer_mismatch_count": protospacer_mismatch_count,
        "mismatch_vector": mismatch_vector,
        "regional_mismatch_counts": regional,
        **regional,
        "mismatch_type_counts": {name: mismatch_types[name] for name in MISMATCH_TYPES},
        "guide_GC_content": guide_gc_content,
        "target_GC_content": target_gc_content,
        "guide_gc_positions": guide_gc,
        "target_gc_positions": target_gc,
        "max_consecutive_mismatches": pattern_features[0],
        "first_mismatch_position": pattern_features[1],
        "last_mismatch_position": pattern_features[2],
        "pam_feature_vector": pam_vector,
        "pam_one_hot": pam_one_hot,
    }
    return details, vector


def compute_features(sgrna: str, off_target: str) -> Tuple[Dict[str, object], np.ndarray]:
    """Return named features and the fixed-order handcrafted feature vector."""
    return build_handcrafted_features(sgrna, off_target)


def feature_groups() -> Dict[str, int]:
    return {
        "mismatch_position_1_to_20": 20,
        "total_mismatch_count": 1,
        "regional_mismatch_counts": 3,
        "mismatch_type_counts": 12,
        "GC_content": 2,
        "position_wise_GC_indicators": 40,
        "mismatch_pattern": 3,
        "target_PAM_one_hot": PAM_LEN * len(BASES),
    }


def inspect_samples(df: pd.DataFrame, sample_count: int, seed: int) -> None:
    sample = df.sample(n=min(sample_count, len(df)), random_state=seed)
    for row_number, (_, row) in enumerate(sample.iterrows(), start=1):
        named, vector = compute_features(row["sgRNA"], row["off_target"])
        print(f"\n--- sample {row_number} ---")
        print("sgRNA:", row["sgRNA"])
        print("off_target:", row["off_target"])
        print("protospacer:", named["protospacer"])
        print("target PAM:", named["target_pam"])
        print("PAM one-hot:")
        for position, values in enumerate(named["pam_one_hot"], start=1):
            print(f"  pos{position}: {values}")
        print("original mismatch_count:", row["mismatch_count"])
        print("recomputed mismatch_count:", named["recomputed_mismatch_count"])
        print("mismatch_vector:", named["mismatch_vector"])
        print(
            "regional mismatch counts:",
            {name: named[name] for name in REGIONS},
        )
        print("mismatch type counts:", json.dumps(named["mismatch_type_counts"], sort_keys=True))
        print("guide GC content:", named["guide_GC_content"])
        print("target GC content:", named["target_GC_content"])
        print("max consecutive mismatches:", named["max_consecutive_mismatches"])
        print("first mismatch position:", named["first_mismatch_position"])
        print("last mismatch position:", named["last_mismatch_position"])
        print("final feature vector shape:", vector.shape)


def run_qc(csv_path: Path, chunksize: int) -> None:
    required = {"sgRNA", "off_target", "mismatch_count"}
    mismatch_total = 0
    mismatch_examples: List[Dict[str, object]] = []
    invalid_examples: List[Dict[str, object]] = []
    length_bad = 0
    non_finite = 0
    pam_length_bad = 0
    pam_invalid_rows = 0
    pam_one_hot_bad = 0
    pam_counts: Counter = Counter()
    ngg_samples = 0
    total_rows = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        missing = required - set(chunk.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        total_rows += len(chunk)
        for index, row in chunk.iterrows():
            guide = str(row["sgRNA"]).strip().upper()
            target = str(row["off_target"]).strip().upper()
            if len(guide) != SEQ_LEN or len(target) != SEQ_LEN:
                length_bad += 1
                if len(target) != SEQ_LEN:
                    pam_length_bad += 1
                if len(invalid_examples) < 20:
                    invalid_examples.append({"row": int(index), "reason": "length"})
                continue
            target_pam = target[-PAM_LEN:]
            if len(target_pam) != PAM_LEN:
                pam_length_bad += 1
            if any(base not in BASES for base in target_pam):
                pam_invalid_rows += 1
            invalid = sorted((set(guide) | set(target)) - set(BASES))
            if invalid:
                if len(invalid_examples) < 20:
                    invalid_examples.append({"row": int(index), "reason": f"bases={invalid}"})
                continue
            named, vector = compute_features(guide, target)
            if int(row["mismatch_count"]) != named["recomputed_mismatch_count"]:
                mismatch_total += 1
                if len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {
                            "row": int(index),
                            "sgRNA": guide,
                            "off_target": target,
                            "original": row["mismatch_count"],
                            "recomputed": named["recomputed_mismatch_count"],
                        }
                    )
            pam_counts[target_pam] += 1
            pam_one_hot = named["pam_one_hot"]
            pam_one_hot_bad += int(
                any(
                    any(value not in {0, 1} for value in position_values)
                    or sum(position_values) != 1
                    for position_values in pam_one_hot
                )
            )
            ngg_samples += int(target_pam[1:] == "GG")
            if not np.isfinite(vector).all():
                non_finite += 1

    feature_dimension = sum(feature_groups().values())
    pam_qc_pass = (
        pam_length_bad == 0
        and pam_invalid_rows == 0
        and pam_one_hot_bad == 0
        and feature_dimension == len(get_feature_names()) == 93
    )
    print("\n=== full-data QC ===")
    print("rows:", total_rows)
    print("mismatch_count inconsistencies:", mismatch_total)
    print("mismatch_count inconsistency ratio:", mismatch_total / max(total_rows, 1))
    print("length-invalid rows:", length_bad)
    print("invalid-base examples (max 20):", json.dumps(invalid_examples))
    print("NaN/Inf feature rows:", non_finite)
    print("target PAM length-invalid rows:", pam_length_bad)
    print("target PAM invalid-character rows:", pam_invalid_rows)
    print("invalid PAM one-hot rows:", pam_one_hot_bad)
    print("feature dimension:", feature_dimension)
    print("feature name count:", len(get_feature_names()))
    print("PAM feature QC:", "PASS" if pam_qc_pass else "FAIL")
    print("\n=== target PAM distribution ===")
    print("Total NGG samples:", ngg_samples)
    print("NGG percentage:", ngg_samples / max(total_rows, 1) * 100.0)
    print("Top 20 target PAMs:")
    for pam, count in pam_counts.most_common(20):
        print(f"  {pam}: {count}")
    if mismatch_examples:
        print("first mismatch_count inconsistency examples:")
        print(json.dumps(mismatch_examples, indent=2))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv_path", type=Path, default=root / "data" / "processed_data" / "train.csv")
    parser.add_argument("--sample_count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=== feature convention ===")
    print("protospacer: sgRNA[1..20] / Python sgRNA[:20]")
    print("PAM: sgRNA[21..23] / Python sgRNA[20:]")
    print("position direction: stored string left-to-right")
    print("regions: distal=1..7, middle=8..14, proximal=15..20")
    print("mismatch type direction: sgRNA base -> target off_target base")
    print("no-mismatch first/last position encoding: 0.0")
    print("\nfeature groups:", json.dumps(feature_groups(), indent=2))
    print("feature dimension =", sum(feature_groups().values()))

    header = pd.read_csv(args.csv_path, nrows=0)
    print("\ncolumns:", list(header.columns))
    sample_df = pd.read_csv(args.csv_path, nrows=max(args.sample_count * 4, args.sample_count))
    inspect_samples(sample_df, args.sample_count, args.seed)
    run_qc(args.csv_path, args.chunksize)


if __name__ == "__main__":
    main()