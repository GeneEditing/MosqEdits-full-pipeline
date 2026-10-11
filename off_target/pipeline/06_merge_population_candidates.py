#!/usr/bin/env python3

import argparse
import csv
import sys
from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from pam_utils import (
    normalize_pam_pattern,
    pam_length,
    pam_matches,
)


SPACER_LEN = 20


# ============================================================
# Helpers
# ============================================================

def safe_int(value, default=None):

    if value is None:
        return default

    value = str(value).strip()

    if value == "":
        return default

    return int(value)


def normalize_dna(seq):

    seq = (
        str(seq)
        .strip()
        .upper()
    )

    invalid = (
        set(seq)
        - set("ACGT")
    )

    if invalid:

        raise ValueError(
            f"Invalid DNA bases {invalid}: {seq}"
        )

    return seq


# ============================================================
# Final output schema
# ============================================================

FIELDNAMES = [

    "candidate_id",
    "guide_id",
    "source",

    "sgRNA_20nt",
    "off_target_20nt",

    "candidate_pam",
    "candidate_with_pam",

    "fasta_id",
    "reference_name",
    "strand",

    "full_start_1based",
    "full_end_1based",

    "protospacer_start_1based",
    "protospacer_end_1based",

    "pam_start_1based",
    "pam_end_1based",

    "mismatch_count_20nt",

    "snp_id",
    "snp_position_1based",
    "snp_region",
    "snp_ref",
    "snp_alt",

    "alt_allele_frequency",
    "carrier_frequency",

    "reference_candidate_id",
    "reference_off_target_20nt",
    "reference_candidate_pam",
]


# ============================================================
# Required input columns
# ============================================================

REQUIRED_COLUMNS = {

    "candidate_id",
    "guide_id",
    "candidate_source",

    "sgRNA_20nt",
    "off_target_20nt",

    "candidate_pam",
    "candidate_with_pam",

    "fasta_id",
    "reference_name",
    "strand",

    "full_start_1based",
    "full_end_1based",

    "protospacer_start_1based",
    "protospacer_end_1based",

    "pam_start_1based",
    "pam_end_1based",

    "mismatch_count_20nt",

    "snp_id",
    "snp_position_1based",
    "snp_region",
    "snp_ref",
    "snp_alt",

    "alt_allele_frequency",
    "carrier_frequency",

    "reference_candidate_id",
    "reference_off_target_20nt",
    "reference_candidate_pam",
}


# ============================================================
# Validate one input file
# ============================================================

def validate_header(path):

    with open(
        path,
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        columns = set(
            reader.fieldnames or []
        )

    missing = (
        REQUIRED_COLUMNS
        - columns
    )

    if missing:

        raise ValueError(
            f"{path} missing columns: "
            f"{sorted(missing)}"
        )


# ============================================================
# Normalize / validate one candidate
# ============================================================

def normalize_candidate(
    row,
    pam_pattern,
    max_mismatch,
):

    pam_len = pam_length(
        pam_pattern
    )

    expected_len = (
        SPACER_LEN
        + pam_len
    )

    candidate_id = (
        row["candidate_id"]
        .strip()
    )

    guide_id = (
        row["guide_id"]
        .strip()
    )

    source = (
        row["candidate_source"]
        .strip()
    )

    if source not in {
        "reference",
        "population_snp",
        "snp_created_pam",
    }:

        raise ValueError(
            f"{candidate_id}: "
            f"unexpected candidate_source "
            f"{source}"
        )

    sgrna20 = normalize_dna(
        row["sgRNA_20nt"]
    )

    off20 = normalize_dna(
        row["off_target_20nt"]
    )

    pam = normalize_dna(
        row["candidate_pam"]
    )

    candidate_with_pam = normalize_dna(
        row["candidate_with_pam"]
    )

    mismatch = int(
        row["mismatch_count_20nt"]
    )

    # --------------------------------------------------------
    # Sequence QC
    # --------------------------------------------------------

    if len(sgrna20) != SPACER_LEN:

        raise ValueError(
            f"{candidate_id}: "
            f"sgRNA length "
            f"{len(sgrna20)} != 20"
        )

    if len(off20) != SPACER_LEN:

        raise ValueError(
            f"{candidate_id}: "
            f"off-target length "
            f"{len(off20)} != 20"
        )

    if len(pam) != pam_len:

        raise ValueError(
            f"{candidate_id}: "
            f"PAM length "
            f"{len(pam)} != {pam_len}"
        )

    if (
        len(candidate_with_pam)
        != expected_len
    ):

        raise ValueError(
            f"{candidate_id}: "
            f"candidate length "
            f"{len(candidate_with_pam)} "
            f"!= {expected_len}"
        )

    if (
        candidate_with_pam[:SPACER_LEN]
        != off20
    ):

        raise ValueError(
            f"{candidate_id}: "
            "candidate_with_pam protospacer "
            "does not equal off_target_20nt."
        )

    if (
        candidate_with_pam[SPACER_LEN:]
        != pam
    ):

        raise ValueError(
            f"{candidate_id}: "
            "candidate_with_pam PAM "
            "does not equal candidate_pam."
        )

    # --------------------------------------------------------
    # Coordinate QC
    # --------------------------------------------------------

    full_start = int(
        row["full_start_1based"]
    )

    full_end = int(
        row["full_end_1based"]
    )

    protospacer_start = int(
        row["protospacer_start_1based"]
    )

    protospacer_end = int(
        row["protospacer_end_1based"]
    )

    pam_start = int(
        row["pam_start_1based"]
    )

    pam_end = int(
        row["pam_end_1based"]
    )

    strand = (
        row["strand"]
        .strip()
    )

    if strand not in {
        "+",
        "-",
    }:

        raise ValueError(
            f"{candidate_id}: "
            f"unexpected strand {strand}"
        )

    if (
        full_end
        - full_start
        + 1
        != expected_len
    ):

        raise ValueError(
            f"{candidate_id}: "
            "full genomic interval length "
            "does not match candidate length."
        )

    if (
        protospacer_end
        - protospacer_start
        + 1
        != SPACER_LEN
    ):

        raise ValueError(
            f"{candidate_id}: "
            "protospacer genomic interval "
            "length != 20."
        )

    if (
        pam_end
        - pam_start
        + 1
        != pam_len
    ):

        raise ValueError(
            f"{candidate_id}: "
            "PAM genomic interval length "
            f"!= {pam_len}."
        )

    # --------------------------------------------------------
    # Filtering
    # --------------------------------------------------------

    if mismatch > max_mismatch:

        return None, "mismatch"

    if not pam_matches(
        pam,
        pam_pattern,
    ):

        return None, "pam"

    # --------------------------------------------------------
    # Standard final row
    # --------------------------------------------------------

    output = {

        "candidate_id":
            candidate_id,

        "guide_id":
            guide_id,

        "source":
            source,

        "sgRNA_20nt":
            sgrna20,

        "off_target_20nt":
            off20,

        "candidate_pam":
            pam,

        "candidate_with_pam":
            candidate_with_pam,

        "fasta_id":
            row["fasta_id"],

        "reference_name":
            row["reference_name"],

        "strand":
            strand,

        "full_start_1based":
            full_start,

        "full_end_1based":
            full_end,

        "protospacer_start_1based":
            protospacer_start,

        "protospacer_end_1based":
            protospacer_end,

        "pam_start_1based":
            pam_start,

        "pam_end_1based":
            pam_end,

        "mismatch_count_20nt":
            mismatch,

        "snp_id":
            row.get(
                "snp_id",
                ""
            ),

        "snp_position_1based":
            row.get(
                "snp_position_1based",
                ""
            ),

        "snp_region":
            row.get(
                "snp_region",
                ""
            ),

        "snp_ref":
            row.get(
                "snp_ref",
                ""
            ),

        "snp_alt":
            row.get(
                "snp_alt",
                ""
            ),

        "alt_allele_frequency":
            row.get(
                "alt_allele_frequency",
                ""
            ),

        "carrier_frequency":
            row.get(
                "carrier_frequency",
                ""
            ),

        "reference_candidate_id":
            row.get(
                "reference_candidate_id",
                ""
            ),

        "reference_off_target_20nt":
            row.get(
                "reference_off_target_20nt",
                ""
            ),

        "reference_candidate_pam":
            row.get(
                "reference_candidate_pam",
                ""
            ),
    }

    return output, None


# ============================================================
# Process one input file
# ============================================================

def process_file(
    input_path,
    writer,
    pam_pattern,
    max_mismatch,
    counters,
    expected_sources,
    seen_candidate_ids,
):

    validate_header(
        input_path
    )

    with open(
        input_path,
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            counters[
                "input_rows"
            ] += 1

            source = (
                row[
                    "candidate_source"
                ]
                .strip()
            )

            if (
                source
                not in expected_sources
            ):

                raise ValueError(
                    f"{input_path}: "
                    f"unexpected source "
                    f"{source}"
                )

            output, filtered_reason = (
                normalize_candidate(
                    row,
                    pam_pattern,
                    max_mismatch,
                )
            )

            if (
                filtered_reason
                == "mismatch"
            ):

                counters[
                    "filtered_mismatch"
                ] += 1

                continue

            if (
                filtered_reason
                == "pam"
            ):

                counters[
                    "filtered_pam"
                ] += 1

                continue

            candidate_id = output[
                "candidate_id"
            ]

            if (
                candidate_id
                in seen_candidate_ids
            ):

                raise ValueError(
                    "Duplicate candidate_id "
                    f"after merge: "
                    f"{candidate_id}"
                )

            seen_candidate_ids.add(
                candidate_id
            )

            writer.writerow(
                output
            )

            counters[
                "output_rows"
            ] += 1

            counters[
                f"kept_{output['source']}"
            ] += 1


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Merge reference, population-SNP, "
            "and SNP-created-PAM off-target "
            "candidates using an arbitrary "
            "IUPAC PAM pattern."
        )
    )

    parser.add_argument(
        "--population-candidates",
        required=True,
        type=Path,
        help=(
            "Output of "
            "04_apply_population_snps.py"
        ),
    )

    parser.add_argument(
        "--created-pam-candidates",
        required=True,
        type=Path,
        help=(
            "Output of "
            "05_find_snp_created_pams.py"
        ),
    )

    parser.add_argument(
        "--pam",
        required=True,
        type=str,
        help=(
            "IUPAC PAM pattern, "
            "e.g. NGG, NGTT, TTTV."
        ),
    )

    parser.add_argument(
        "--max-mismatch",
        type=int,
        default=6,
        help=(
            "Maximum allowed 20nt mismatch."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    # ========================================================
    # Validation
    # ========================================================

    if not args.population_candidates.exists():

        raise FileNotFoundError(
            args.population_candidates
        )

    if not args.created_pam_candidates.exists():

        raise FileNotFoundError(
            args.created_pam_candidates
        )

    if args.max_mismatch < 0:

        raise ValueError(
            "--max-mismatch must be >= 0."
        )

    pam_pattern = (
        normalize_pam_pattern(
            args.pam
        )
    )

    pam_len = (
        pam_length(
            pam_pattern
        )
    )

    candidate_len = (
        SPACER_LEN
        + pam_len
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Counters
    # ========================================================

    counters = {

        "input_rows":
            0,

        "filtered_mismatch":
            0,

        "filtered_pam":
            0,

        "output_rows":
            0,

        "kept_reference":
            0,

        "kept_population_snp":
            0,

        "kept_snp_created_pam":
            0,
    }

    seen_candidate_ids = set()

    # ========================================================
    # Summary
    # ========================================================

    print("=" * 70)
    print(
        "Population-aware candidate merge"
    )
    print("=" * 70)

    print(
        f"PAM pattern:                 "
        f"{pam_pattern}"
    )

    print(
        f"PAM length:                  "
        f"{pam_len}"
    )

    print(
        f"Candidate length:            "
        f"{candidate_len}"
    )

    print(
        f"Maximum mismatch:            "
        f"{args.max_mismatch}"
    )

    print()

    # ========================================================
    # Merge
    # ========================================================

    with open(
        args.output,
        "w",
        newline="",
    ) as fout:

        writer = csv.DictWriter(
            fout,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()

        # ----------------------------------------------------
        # Step 04:
        #
        # reference
        # population_snp
        # ----------------------------------------------------

        process_file(
            input_path=
                args.population_candidates,

            writer=
                writer,

            pam_pattern=
                pam_pattern,

            max_mismatch=
                args.max_mismatch,

            counters=
                counters,

            expected_sources={
                "reference",
                "population_snp",
            },

            seen_candidate_ids=
                seen_candidate_ids,
        )

        # ----------------------------------------------------
        # Step 05:
        #
        # snp_created_pam
        # ----------------------------------------------------

        process_file(
            input_path=
                args.created_pam_candidates,

            writer=
                writer,

            pam_pattern=
                pam_pattern,

            max_mismatch=
                args.max_mismatch,

            counters=
                counters,

            expected_sources={
                "snp_created_pam",
            },

            seen_candidate_ids=
                seen_candidate_ids,
        )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 70)
    print(
        "Population-aware candidate merge complete"
    )
    print("=" * 70)

    print(
        f"Input rows:                  "
        f"{counters['input_rows']:,}"
    )

    print()

    print(
        f"Filtered mismatch > "
        f"{args.max_mismatch}:      "
        f"{counters['filtered_mismatch']:,}"
    )

    print(
        f"Filtered invalid PAM:        "
        f"{counters['filtered_pam']:,}"
    )

    print()

    print(
        "Final sources:"
    )

    print(
        f"  reference:                 "
        f"{counters['kept_reference']:,}"
    )

    print(
        f"  population_snp:            "
        f"{counters['kept_population_snp']:,}"
    )

    print(
        f"  snp_created_pam:           "
        f"{counters['kept_snp_created_pam']:,}"
    )

    print()

    print(
        f"Final candidate rows:        "
        f"{counters['output_rows']:,}"
    )

    print()

    print(
        f"Unique candidate IDs:        "
        f"{len(seen_candidate_ids):,}"
    )

    print()

    print(
        f"Output: {args.output}"
    )


if __name__ == "__main__":
    main()