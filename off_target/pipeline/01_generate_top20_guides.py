#!/usr/bin/env python3
"""Generate candidate guides and retain only the on-target Top20 guides."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


MATCH_COLUMNS = [
    ("chrom", "chrom"),
    ("spacer", "gRNA_20nt"),
    ("pam", "pam"),
    ("strand", "strand"),
    ("genomic_spacer_start", "genomic_spacer_start"),
    ("genomic_spacer_end", "genomic_spacer_end"),
    ("genomic_pam_start", "genomic_pam_start"),
    ("genomic_pam_end", "genomic_pam_end"),
]


def normalized_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all target-region guides, then retain on-target Top20."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--pam", required=True)
    parser.add_argument("--on-target-top20", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--guide-prefix", default="gRNA")
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.on_target_top20.is_file():
        raise FileNotFoundError(args.on_target_top20)

    generator = Path(__file__).resolve().parent / "01_generate_guides.py"
    if not generator.is_file():
        raise FileNotFoundError(generator)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="aedesguide_candidates_",
        dir=args.output.parent,
    ) as temporary_directory:
        all_guides_path = Path(temporary_directory) / "01_all_guides.csv"

        command = [
            sys.executable,
            str(generator),
            "--input",
            str(args.input),
            "--pam",
            args.pam,
            "--output",
            str(all_guides_path),
            "--guide-prefix",
            args.guide_prefix,
        ]

        print("=" * 70)
        print("Generate candidates before Top20 selection")
        print("=" * 70)
        print("Command:", " ".join(command))
        subprocess.run(command, check=True)

        candidates = pd.read_csv(all_guides_path)
        top20 = pd.read_csv(args.on_target_top20, sep="\t")

        required_candidate_columns = {
            candidate_column for candidate_column, _ in MATCH_COLUMNS
        } | {"guide_id"}

        required_top20_columns = {
            top20_column for _, top20_column in MATCH_COLUMNS
        } | {"rank", "guide_id"}

        missing_candidates = sorted(
            required_candidate_columns - set(candidates.columns)
        )
        missing_top20 = sorted(required_top20_columns - set(top20.columns))

        if missing_candidates:
            raise ValueError(
                f"Candidate guide file is missing columns: {missing_candidates}"
            )
        if missing_top20:
            raise ValueError(
                f"On-target Top20 file is missing columns: {missing_top20}"
            )

        top20 = top20.sort_values("rank", kind="mergesort").head(20).copy()

        if top20.empty:
            raise ValueError("On-target Top20 file contains no guides")

        if top20["guide_id"].astype(str).duplicated().any():
            raise ValueError("On-target Top20 contains duplicate guide_id values")

        candidate_keys = []
        top20_keys = []

        for candidate_column, top20_column in MATCH_COLUMNS:
            candidate_key = f"_match_{candidate_column}"
            top20_key = f"_match_{candidate_column}"

            if candidate_column.startswith("genomic_"):
                candidates[candidate_key] = pd.to_numeric(
                    candidates[candidate_column],
                    errors="raise",
                ).astype("Int64")

                top20[top20_key] = pd.to_numeric(
                    top20[top20_column],
                    errors="raise",
                ).astype("Int64")
            else:
                candidates[candidate_key] = normalized_text(
                    candidates[candidate_column]
                )
                top20[top20_key] = normalized_text(top20[top20_column])

            candidate_keys.append(candidate_key)
            top20_keys.append(top20_key)

        candidates["_candidate_row"] = range(len(candidates))

        selected = top20.merge(
            candidates,
            how="left",
            left_on=top20_keys,
            right_on=candidate_keys,
            suffixes=("_ontarget", ""),
            validate="one_to_one",
            indicator=True,
        )

        unmatched = selected[selected["_merge"] != "both"]
        if not unmatched.empty:
            columns = [
                "rank",
                "guide_id_ontarget",
                "gRNA_20nt",
                "pam_ontarget",
                "chrom_ontarget",
                "strand_ontarget",
            ]
            available_columns = [
                column for column in columns if column in unmatched.columns
            ]
            examples = unmatched[available_columns].head(10).to_dict("records")
            raise ValueError(
                f"{len(unmatched)} on-target guide(s) could not be matched "
                f"to generated candidates; examples: {examples}"
            )

        output_columns = [
            column
            for column in candidates.columns
            if not column.startswith("_match_")
            and column != "_candidate_row"
        ]

        selected["guide_id"] = selected["guide_id_ontarget"].astype(str)
        selected = selected.sort_values("rank", kind="mergesort")
        selected = selected[output_columns]

        if len(selected) != len(top20):
            raise RuntimeError(
                f"Top20 selection changed row count: "
                f"{len(top20)} -> {len(selected)}"
            )

        if selected["guide_id"].duplicated().any():
            raise RuntimeError("Selected guides contain duplicate guide_id values")

        selected.to_csv(args.output, index=False)

    print()
    print("=" * 70)
    print("On-target Top20 guide selection complete")
    print("=" * 70)
    print(f"All generated candidates: {len(candidates):,}")
    print(f"Selected guides:          {len(selected):,}")
    print(f"Output:                   {args.output}")


if __name__ == "__main__":
    main()