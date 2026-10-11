#!/usr/bin/env python3

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from shutil import which

import pandas as pd


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
)


SPACER_LEN = 20


# ============================================================
# Default genome
# ============================================================

DEFAULT_GENOME = (
    PROJECT_ROOT
    / "reference"
    / "AalbF3"
    / "GCA_018104305.1_AalbF3_genomic.fna"
)


# ============================================================
# Cas-OFFinder executable
# ============================================================

def find_cas_offinder(explicit_path=None):

    candidates = []

    if explicit_path is not None:
        candidates.append(
            Path(explicit_path)
        )

    candidates.extend([
        PROJECT_ROOT / "tools" / "cas-offinder",
        PROJECT_ROOT / "cas-offinder",
        PROJECT_ROOT / "bin" / "cas-offinder",
    ])

    for candidate in candidates:

        if (
            candidate.exists()
            and os.access(
                candidate,
                os.X_OK
            )
        ):

            return candidate.resolve()

    found = which(
        "cas-offinder"
    )

    if found is not None:

        return Path(
            found
        ).resolve()

    raise FileNotFoundError(
        "Could not locate executable cas-offinder.\n"
        "Please provide --cas-offinder PATH."
    )


# ============================================================
# Guide loading
# ============================================================

def load_guides(
    path,
    pam_pattern,
):

    print(
        f"Loading guides: {path}"
    )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    required = {
        "guide_id",
        "spacer",
        "pam",
        "chrom",
        "strand",
        "genomic_spacer_start",
        "genomic_spacer_end",
        "genomic_pam_start",
        "genomic_pam_end",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Guide file missing columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    df["guide_id"] = (
        df["guide_id"]
        .astype(str)
    )

    df["spacer"] = (
        df["spacer"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["pam"] = (
        df["pam"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Guide QC
    # --------------------------------------------------------

    spacer_valid = (
        df["spacer"]
        .str.fullmatch(
            r"[ACGT]{20}"
        )
    )

    if not spacer_valid.all():

        n_bad = int(
            (~spacer_valid).sum()
        )

        raise ValueError(
            f"Invalid 20nt spacers: "
            f"{n_bad:,}"
        )

    expected_pam_len = (
        pam_length(
            pam_pattern
        )
    )

    pam_length_valid = (
        df["pam"]
        .str.len()
        .eq(
            expected_pam_len
        )
    )

    if not pam_length_valid.all():

        n_bad = int(
            (~pam_length_valid).sum()
        )

        raise ValueError(
            "Guide-table PAM length does not "
            f"match --pam {pam_pattern}. "
            f"Invalid rows: {n_bad:,}"
        )

    if df[
        "guide_id"
    ].duplicated().any():

        duplicates = (
            df.loc[
                df[
                    "guide_id"
                ].duplicated(),
                "guide_id"
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate guide_id detected: "
            f"{duplicates[:20]}"
        )

    # --------------------------------------------------------
    # Important:
    #
    # Cas-OFFinder raw output only returns query_sequence.
    # Therefore query_sequence must uniquely identify a guide.
    #
    # If the exact same spacer appears multiple times in the
    # target region, one query could map to multiple guide loci.
    #
    # For the current design pipeline we require unique spacer
    # sequences at this step to make downstream mapping exact.
    # --------------------------------------------------------

    if df[
        "spacer"
    ].duplicated().any():

        duplicated_spacers = (
            df.loc[
                df[
                    "spacer"
                ].duplicated(
                    keep=False
                ),
                [
                    "guide_id",
                    "spacer",
                ]
            ]
            .sort_values(
                "spacer"
            )
        )

        raise ValueError(
            "Duplicate spacer sequences detected in guide table. "
            "Cas-OFFinder query mapping would be ambiguous.\n"
            f"{duplicated_spacers.head(20).to_string(index=False)}"
        )

    return df


# ============================================================
# Output path helper
# ============================================================

def derive_auxiliary_paths(
    raw_output,
    input_file=None,
    query_map=None,
):

    raw_output = Path(
        raw_output
    )

    raw_name = raw_output.name

    # Example:
    #
    # white_AalbF3_NGTT_cas_offinder_raw.txt
    #
    # becomes:
    #
    # white_AalbF3_NGTT_cas_offinder_input.txt
    # white_AalbF3_NGTT_cas_offinder_query_map.tsv

    if raw_name.endswith(
        "_raw.txt"
    ):

        prefix = raw_name[
            :-len("_raw.txt")
        ]

    else:

        prefix = raw_output.stem

    if input_file is None:

        input_path = (
            raw_output.parent
            / f"{prefix}_input.txt"
        )

    else:

        input_path = Path(
            input_file
        )

    if query_map is None:

        query_map_path = (
            raw_output.parent
            / f"{prefix}_query_map.tsv"
        )

    else:

        query_map_path = Path(
            query_map
        )

    return (
        input_path,
        query_map_path,
    )


# ============================================================
# Query map
# ============================================================

def write_query_map(
    guides,
    pam_pattern,
    path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pam_len = pam_length(
        pam_pattern
    )

    fieldnames = [
        "query_sequence",
        "guide_id",
        "spacer",
        "pam",
        "chrom",
        "strand",
        "genomic_spacer_start",
        "genomic_spacer_end",
        "genomic_pam_start",
        "genomic_pam_end",
    ]

    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="\t",
        )

        writer.writeheader()

        for row in guides.itertuples(
            index=False
        ):

            # Cas-OFFinder query includes N placeholders
            # for the PAM region.
            query_sequence = (
                row.spacer
                + "N" * pam_len
            )

            writer.writerow({

                "query_sequence":
                    query_sequence,

                "guide_id":
                    row.guide_id,

                "spacer":
                    row.spacer,

                "pam":
                    row.pam,

                "chrom":
                    row.chrom,

                "strand":
                    row.strand,

                "genomic_spacer_start":
                    row.genomic_spacer_start,

                "genomic_spacer_end":
                    row.genomic_spacer_end,

                "genomic_pam_start":
                    row.genomic_pam_start,

                "genomic_pam_end":
                    row.genomic_pam_end,
            })

    return path


# ============================================================
# Cas-OFFinder input
# ============================================================

def write_cas_offinder_input(
    guides,
    genome,
    pam_pattern,
    max_mismatches,
    path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pam_len = pam_length(
        pam_pattern
    )

    candidate_length = (
        SPACER_LEN
        + pam_len
    )

    # Pattern constrains PAM.
    #
    # Example NGG:
    # NNNNNNNNNNNNNNNNNNNNNGG
    #
    # Example NGTT:
    # NNNNNNNNNNNNNNNNNNNNNGTT

    pattern = (
        "N" * SPACER_LEN
        + pam_pattern
    )

    if len(
        pattern
    ) != candidate_length:

        raise RuntimeError(
            "Internal Cas-OFFinder pattern "
            "length mismatch."
        )

    query_pam = (
        "N" * pam_len
    )

    with open(
        path,
        "w",
    ) as f:

        # ----------------------------------------------------
        # Line 1
        # Genome FASTA
        # ----------------------------------------------------

        f.write(
            f"{Path(genome).resolve()}\n"
        )

        # ----------------------------------------------------
        # Line 2
        # Search pattern
        # ----------------------------------------------------

        f.write(
            f"{pattern}\n"
        )

        # ----------------------------------------------------
        # Remaining lines
        # Query + mismatch threshold
        #
        # IMPORTANT:
        #
        # Do NOT append guide_id here.
        #
        # Cas-OFFinder query output uses the actual query
        # sequence as its identifier. We separately maintain
        # query_map.tsv to recover guide metadata.
        # ----------------------------------------------------

        for row in guides.itertuples(
            index=False
        ):

            query = (
                row.spacer
                + query_pam
            )

            if len(
                query
            ) != candidate_length:

                raise RuntimeError(
                    f"Query length error for "
                    f"{row.guide_id}: "
                    f"{len(query)} != "
                    f"{candidate_length}"
                )

            f.write(
                f"{query} "
                f"{max_mismatches}\n"
            )

    return (
        pattern,
        candidate_length,
    )


# ============================================================
# OpenCL environment
# ============================================================

def build_subprocess_environment():

    env = os.environ.copy()

    # --------------------------------------------------------
    # IMPORTANT
    #
    # If the user already exported OpenCL settings,
    # preserve them.
    #
    # We do NOT hard-code a vendor path here because
    # different servers may expose OpenCL differently.
    # --------------------------------------------------------

    return env


# ============================================================
# Run Cas-OFFinder
# ============================================================

def run_cas_offinder(
    executable,
    input_file,
    output_file,
    device,
):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        str(executable),
        str(input_file),
        str(device),
        str(output_file),
    ]

    print()
    print("=" * 70)
    print("Running Cas-OFFinder")
    print("=" * 70)

    print(
        "Command:"
    )

    print(
        " ".join(
            command
        )
    )

    print()

    env = build_subprocess_environment()

    result = subprocess.run(
        command,
        check=False,
        env=env,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Cas-OFFinder failed with "
            f"exit code "
            f"{result.returncode}."
        )

    if not output_file.exists():

        raise RuntimeError(
            "Cas-OFFinder finished but "
            "output file was not created."
        )


# ============================================================
# Raw-output QC
# ============================================================

def count_nonempty_lines(path):

    count = 0

    with open(
        path,
        "r",
    ) as f:

        for line in f:

            if line.strip():
                count += 1

    return count


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run Cas-OFFinder for 20nt guides "
            "with an arbitrary-length IUPAC PAM. "
            "Automatically writes Cas-OFFinder input "
            "and query-map files."
        )
    )

    parser.add_argument(
        "--guides",
        required=True,
        type=Path,
        help=(
            "Guide CSV generated by "
            "01_generate_guides.py"
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
        "--genome",
        type=Path,
        default=DEFAULT_GENOME,
        help=(
            "Whole-genome FASTA."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=(
            "Raw Cas-OFFinder output."
        ),
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help=(
            "Optional explicit Cas-OFFinder "
            "input path."
        ),
    )

    parser.add_argument(
        "--query-map",
        type=Path,
        default=None,
        help=(
            "Optional explicit query-map TSV path."
        ),
    )

    parser.add_argument(
        "--cas-offinder",
        type=Path,
        default=None,
        help=(
            "Path to cas-offinder executable."
        ),
    )

    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="G",
        help=(
            "Cas-OFFinder device, "
            "e.g. G, G0, G6, C."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Input validation
    # ========================================================

    if not args.guides.exists():

        raise FileNotFoundError(
            args.guides
        )

    if not args.genome.exists():

        raise FileNotFoundError(
            args.genome
        )

    if args.max_mismatches < 0:

        raise ValueError(
            "--max-mismatches must be >= 0."
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

    executable = find_cas_offinder(
        args.cas_offinder
    )

    # ========================================================
    # Derived output paths
    # ========================================================

    (
        input_file,
        query_map_file,
    ) = derive_auxiliary_paths(
        args.output,
        args.input_file,
        args.query_map,
    )

    # ========================================================
    # Load guides
    # ========================================================

    guides = load_guides(
        args.guides,
        pam_pattern,
    )

    # ========================================================
    # Write query map
    # ========================================================

    write_query_map(
        guides,
        pam_pattern,
        query_map_file,
    )

    # ========================================================
    # Write Cas-OFFinder input
    # ========================================================

    (
        pattern,
        candidate_length,
    ) = write_cas_offinder_input(
        guides,
        args.genome,
        pam_pattern,
        args.max_mismatches,
        input_file,
    )

    # ========================================================
    # Summary before execution
    # ========================================================

    print()
    print("=" * 70)
    print("Cas-OFFinder input summary")
    print("=" * 70)

    print(
        f"Guides:              "
        f"{len(guides):,}"
    )

    print(
        f"PAM pattern:         "
        f"{pam_pattern}"
    )

    print(
        f"PAM length:          "
        f"{pam_len}"
    )

    print(
        f"Spacer length:       "
        f"{SPACER_LEN}"
    )

    print(
        f"Candidate length:    "
        f"{candidate_length}"
    )

    print(
        f"Pattern:             "
        f"{pattern}"
    )

    print(
        f"Max mismatches:      "
        f"{args.max_mismatches}"
    )

    print(
        f"Genome:              "
        f"{args.genome}"
    )

    print(
        f"Cas-OFFinder:        "
        f"{executable}"
    )

    print(
        f"Device:              "
        f"{args.device}"
    )

    print()
    print(
        f"Input file:          "
        f"{input_file}"
    )

    print(
        f"Query map:           "
        f"{query_map_file}"
    )

    print(
        f"Raw output:          "
        f"{args.output}"
    )

    # ========================================================
    # Execute Cas-OFFinder
    # ========================================================

    run_cas_offinder(
        executable,
        input_file,
        args.output,
        args.device,
    )

    # ========================================================
    # Final QC
    # ========================================================

    rows = count_nonempty_lines(
        args.output
    )

    print()
    print("=" * 70)
    print("Cas-OFFinder complete")
    print("=" * 70)

    print(
        f"Guides:              "
        f"{len(guides):,}"
    )

    print(
        f"Raw output rows:     "
        f"{rows:,}"
    )

    print()

    print(
        f"Cas input:           "
        f"{input_file}"
    )

    print(
        f"Query map:           "
        f"{query_map_file}"
    )

    print(
        f"Raw output:          "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()