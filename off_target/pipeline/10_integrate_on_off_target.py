#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


MATCH_COLUMNS = [
    "gRNA_20nt", "pam", "chrom", "strand",
    "genomic_spacer_start", "genomic_spacer_end",
    "genomic_pam_start", "genomic_pam_end",
]


def load_offtarget(path):
    """Load the complete guide-level output produced by Step 09."""
    print(f"Loading complete off-target guide scores: {path}")
    df = pd.read_csv(path, low_memory=False)
    required = {
        "off_target_rank", "guide_id", "spacer", "pam",
        "highest_off_target_probability",
        "top5_aggregate_off_target_probability",
        "chrom", "strand", "genomic_spacer_start",
        "genomic_spacer_end", "genomic_pam_start", "genomic_pam_end",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Complete off-target file missing columns: "
            f"{sorted(missing)}. Re-run the updated Step 09 first."
        )
    return df.rename(columns={
        "guide_id": "off_target_guide_id",
        "spacer": "gRNA_20nt",
    })


def load_ontarget(path):
    """Load the user-selected on-target Top20 table."""
    print(f"Loading on-target Top20: {path}")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    required = {
        "rank", "guide_id", "gRNA_20nt", "pam", "predicted_efficiency",
        "chrom", "strand", "genomic_spacer_start", "genomic_spacer_end",
        "genomic_pam_start", "genomic_pam_end",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"On-target file missing columns: {sorted(missing)}")
    return df.rename(columns={
        "rank": "on_target_rank",
        "guide_id": "on_target_guide_id",
    })


def normalize_matching_columns(df):
    df = df.copy()
    for col in ["gRNA_20nt", "pam"]:
        df[col] = df[col].astype(str).str.upper().str.strip()
    for col in ["chrom", "strand"]:
        df[col] = df[col].astype(str).str.strip()
    for col in [
        "genomic_spacer_start", "genomic_spacer_end",
        "genomic_pam_start", "genomic_pam_end",
    ]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    return df


def validate_inputs(on, off):
    if len(on) == 0:
        raise ValueError("The on-target Top20 file contains no guides.")
    if len(on) > 20:
        raise ValueError(
            f"The on-target Top20 file contains {len(on)} rows; "
            "expected at most 20."
        )

    ranks = pd.to_numeric(on["on_target_rank"], errors="raise").astype(int)
    expected = list(range(1, len(on) + 1))
    if ranks.tolist() != expected:
        raise ValueError(
            f"On-target ranks must be consecutive from 1 to {len(on)} "
            "and already sorted."
        )

    if on[MATCH_COLUMNS].duplicated().any():
        raise ValueError("Duplicate genomic loci in on-target Top20.")
    if off[MATCH_COLUMNS].duplicated().any():
        duplicated = off.loc[
            off[MATCH_COLUMNS].duplicated(keep=False), MATCH_COLUMNS
        ].head(10)
        raise ValueError(
            "Duplicate genomic loci in complete off-target scores:\n"
            + duplicated.to_string(index=False)
        )

    for col in [
        "highest_off_target_probability",
        "top5_aggregate_off_target_probability",
    ]:
        values = pd.to_numeric(off[col], errors="raise")
        if values.isna().any() or not values.between(0.0, 1.0).all():
            raise ValueError(f"{col} contains invalid probabilities.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Annotate on-target Top20 guides with complete Step 09 "
            "off-target scores while preserving on-target ranking."
        )
    )
    parser.add_argument(
        "--off-target", required=True, type=Path,
        help="Complete Step 09 guide score CSV, not its Top20 subset.",
    )
    parser.add_argument(
        "--on-target", required=True, type=Path,
        help="On-target Top20 TSV.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    off = normalize_matching_columns(load_offtarget(args.off_target))
    on = normalize_matching_columns(load_ontarget(args.on_target))
    validate_inputs(on, off)

    print()
    print("=" * 70)
    print("Input summary")
    print("=" * 70)
    print(f"Complete off-target guide scores: {len(off):,}")
    print(f"On-target Top guides:             {len(on):,}")

    # On-target Top20 is authoritative. Off-target scores annotate its rows;
    # they never filter or reorder them automatically.
    result = on.merge(
        off[MATCH_COLUMNS + [
            "off_target_guide_id", "off_target_rank",
            "highest_off_target_probability",
            "top5_aggregate_off_target_probability",
        ]],
        on=MATCH_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing = result.loc[
        result["_merge"] != "both",
        ["on_target_rank", "on_target_guide_id", "gRNA_20nt", "pam"],
    ]
    if not missing.empty:
        raise ValueError(
            f"No Step 09 score found for {len(missing)} on-target guide(s):\n"
            + missing.to_string(index=False)
        )

    result = (
        result.drop(columns="_merge")
        .sort_values("on_target_rank", ascending=True)
        .reset_index(drop=True)
    )
    result.insert(0, "final_rank", result["on_target_rank"].astype(int))

    output_columns = [
        "final_rank", "gRNA_20nt", "pam", "predicted_efficiency",
        "top5_aggregate_off_target_probability",
        "highest_off_target_probability",
        "on_target_rank", "off_target_rank", "chrom", "strand",
        "genomic_spacer_start", "genomic_spacer_end",
        "genomic_pam_start", "genomic_pam_end",
        "on_target_guide_id", "off_target_guide_id",
    ]
    final = result[output_columns].copy()

    expected_ranks = list(range(1, len(final) + 1))
    if final["on_target_rank"].astype(int).tolist() != expected_ranks:
        raise RuntimeError("Final output no longer follows on-target ranking.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, sep="\t", index=False)

    print()
    print("=" * 70)
    print("On-target-prioritized integration complete")
    print("=" * 70)
    print(f"On-target guides retained:       {len(final):,}/{len(on):,}")
    print("Automatic off-target deletions:  0")
    print("Ordering:                       on_target_rank (unchanged)")
    print(f"Output: {args.output}")
    print()
    print("Review top5_aggregate_off_target_probability manually; ")
    print("remove guides only when their off-target risk is judged excessive.")
    print()

    display = final[[
        "final_rank", "gRNA_20nt", "pam", "predicted_efficiency",
        "top5_aggregate_off_target_probability",
        "highest_off_target_probability", "off_target_rank",
    ]].copy()
    for col in [
        "predicted_efficiency",
        "top5_aggregate_off_target_probability",
        "highest_off_target_probability",
    ]:
        display[col] = display[col].round(4)
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
