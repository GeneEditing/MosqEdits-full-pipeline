#!/usr/bin/env python3

import argparse
import bisect
import csv
import re
import sys
from collections import defaultdict
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
# DNA helpers
# ============================================================

COMPLEMENT = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
}


def complement_base(base):

    base = str(base).upper()

    if base not in COMPLEMENT:
        raise ValueError(
            f"Cannot complement base: {base}"
        )

    return COMPLEMENT[base]


def hamming(a, b):

    if len(a) != len(b):
        raise ValueError(
            f"Hamming length mismatch: "
            f"{len(a)} vs {len(b)}"
        )

    return sum(
        x != y
        for x, y in zip(a, b)
    )


# ============================================================
# FASTA/reference helper
# ============================================================

def extract_fasta_id(reference_name):
    """
    Cas-OFFinder reference_name can contain the complete FASTA header.

    Example:

        JAFDOQ010000565.1 Aedes albopictus ...

    Population SNP table uses:

        JAFDOQ010000565.1

    Therefore use the first whitespace-delimited token.
    """

    value = str(reference_name).strip()

    if not value:
        raise ValueError(
            "Empty reference_name."
        )

    return value.split()[0]


# ============================================================
# Population SNP loading
# ============================================================

def load_population_snps(path):
    """
    Load prepared population SNP TSV.

    Index structure:

        fasta_id
            -> sorted SNP records

    Each SNP must contain:
        snp_id
        fasta_id
        position_1based
        ref
        alt
        alt_allele_frequency
        carrier_frequency
    """

    print(
        f"Loading population SNPs: {path}"
    )

    by_contig = defaultdict(list)

    total = 0

    with open(
        path,
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        required = {
            "snp_id",
            "fasta_id",
            "position_1based",
            "ref",
            "alt",
            "alt_allele_frequency",
            "carrier_frequency",
        }

        missing = (
            required
            - set(reader.fieldnames or [])
        )

        if missing:
            raise ValueError(
                "Population SNP table missing columns: "
                f"{sorted(missing)}"
            )

        for row in reader:

            total += 1

            fasta_id = (
                row["fasta_id"]
                .strip()
            )

            pos = int(
                row["position_1based"]
            )

            ref = (
                row["ref"]
                .strip()
                .upper()
            )

            alt = (
                row["alt"]
                .strip()
                .upper()
            )

            if (
                len(ref) != 1
                or len(alt) != 1
                or ref not in "ACGT"
                or alt not in "ACGT"
            ):
                raise ValueError(
                    "Population SNP table contains "
                    f"non-biallelic SNP at "
                    f"{fasta_id}:{pos}"
                )

            by_contig[
                fasta_id
            ].append({

                "snp_id":
                    row["snp_id"],

                "position_1based":
                    pos,

                "ref":
                    ref,

                "alt":
                    alt,

                "alt_allele_frequency":
                    float(
                        row[
                            "alt_allele_frequency"
                        ]
                    ),

                "carrier_frequency":
                    float(
                        row[
                            "carrier_frequency"
                        ]
                    ),
            })

    # --------------------------------------------------------
    # Sort + build position arrays for fast interval lookup
    # --------------------------------------------------------

    indexed = {}

    for fasta_id, records in (
        by_contig.items()
    ):

        records.sort(
            key=lambda x:
                x[
                    "position_1based"
                ]
        )

        positions = [
            row[
                "position_1based"
            ]
            for row in records
        ]

        indexed[
            fasta_id
        ] = {
            "positions":
                positions,
            "records":
                records,
        }

    print(
        f"Population SNPs loaded: "
        f"{total:,}"
    )

    print(
        f"FASTA contigs with SNPs: "
        f"{len(indexed):,}"
    )

    return indexed


# ============================================================
# SNP interval lookup
# ============================================================

def find_snps_in_interval(
    snp_index,
    fasta_id,
    start1,
    end1,
):

    if fasta_id not in snp_index:
        return []

    block = snp_index[
        fasta_id
    ]

    positions = block[
        "positions"
    ]

    records = block[
        "records"
    ]

    left = bisect.bisect_left(
        positions,
        start1,
    )

    right = bisect.bisect_right(
        positions,
        end1,
    )

    return records[
        left:right
    ]


# ============================================================
# Candidate normalization
# ============================================================

def normalize_candidate_row(
    row,
    pam_pattern,
):
    """
    Convert Step 03 candidate row into the canonical
    schema used by the population-aware pipeline.
    """

    pam_len = pam_length(
        pam_pattern
    )

    expected_length = (
        SPACER_LEN
        + pam_len
    )

    guide_id = (
        row["guide_id"]
        .strip()
    )

    candidate_id = (
        row["candidate_id"]
        .strip()
    )

    sgrna20 = (
        row["sgRNA_20nt"]
        .strip()
        .upper()
    )

    off20 = (
        row["off_target_20nt"]
        .strip()
        .upper()
    )

    candidate_pam = (
        row["candidate_pam"]
        .strip()
        .upper()
    )

    candidate_with_pam = (
        row["candidate_with_pam"]
        .strip()
        .upper()
    )

    reference_name = (
        row["reference_name"]
        .strip()
    )

    strand = (
        row["off_target_strand"]
        .strip()
    )

    full_start = int(
        row["genomic_start_1based"]
    )

    full_end = int(
        row["genomic_end_1based"]
    )

    mismatch_count = int(
        row["mismatch_count_20nt"]
    )

    # --------------------------------------------------------
    # QC
    # --------------------------------------------------------

    if len(sgrna20) != SPACER_LEN:

        raise ValueError(
            f"{candidate_id}: "
            f"sgRNA length != 20"
        )

    if len(off20) != SPACER_LEN:

        raise ValueError(
            f"{candidate_id}: "
            f"off-target length != 20"
        )

    if len(candidate_pam) != pam_len:

        raise ValueError(
            f"{candidate_id}: "
            f"PAM length {len(candidate_pam)} "
            f"!= {pam_len}"
        )

    if len(
        candidate_with_pam
    ) != expected_length:

        raise ValueError(
            f"{candidate_id}: "
            f"candidate length "
            f"{len(candidate_with_pam)} "
            f"!= {expected_length}"
        )

    if (
        candidate_with_pam[
            :SPACER_LEN
        ]
        != off20
    ):

        raise ValueError(
            f"{candidate_id}: "
            "candidate/off_target mismatch."
        )

    if (
        candidate_with_pam[
            SPACER_LEN:
        ]
        != candidate_pam
    ):

        raise ValueError(
            f"{candidate_id}: "
            "candidate/PAM mismatch."
        )

    if strand not in {
        "+",
        "-",
    }:

        raise ValueError(
            f"{candidate_id}: "
            f"invalid strand {strand}"
        )

    if (
        full_end
        - full_start
        + 1
        != expected_length
    ):

        raise ValueError(
            f"{candidate_id}: "
            "genomic candidate interval length "
            "does not match sequence length."
        )

    recalculated_mm = hamming(
        sgrna20,
        off20,
    )

    if (
        recalculated_mm
        != mismatch_count
    ):

        raise ValueError(
            f"{candidate_id}: "
            f"mismatch_count_20nt "
            f"{mismatch_count} != "
            f"recalculated "
            f"{recalculated_mm}"
        )

    # --------------------------------------------------------
    # Candidate genomic coordinates
    #
    # Candidate sequence itself is stored in guide orientation:
    #
    #   [20nt protospacer][PAM]
    #
    # For '+' strand, genome physical order is also:
    #
    #   [protospacer][PAM]
    #
    # For '-' strand, forward-reference physical order is:
    #
    #   [revcomp(PAM)][revcomp(protospacer)]
    # --------------------------------------------------------

    if strand == "+":

        protospacer_start = (
            full_start
        )

        protospacer_end = (
            full_start
            + SPACER_LEN
            - 1
        )

        pam_start = (
            protospacer_end
            + 1
        )

        pam_end = (
            full_end
        )

    else:

        pam_start = (
            full_start
        )

        pam_end = (
            full_start
            + pam_len
            - 1
        )

        protospacer_start = (
            pam_end
            + 1
        )

        protospacer_end = (
            full_end
        )

    fasta_id = extract_fasta_id(
        reference_name
    )

    return {

        "candidate_id":
            candidate_id,

        "guide_id":
            guide_id,

        "sgRNA_20nt":
            sgrna20,

        "off_target_20nt":
            off20,

        "candidate_pam":
            candidate_pam,

        "candidate_with_pam":
            candidate_with_pam,

        "fasta_id":
            fasta_id,

        "reference_name":
            reference_name,

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
            mismatch_count,
    }


# ============================================================
# SNP application
# ============================================================

def apply_single_snp(
    reference,
    snp,
    pam_pattern,
):
    """
    Apply one population SNP to one candidate.

    IMPORTANT:
    candidate_with_pam is stored in guide orientation.

    '+' strand:
        oriented index = SNP_position - full_start

    '-' strand:
        oriented index = full_end - SNP_position

    On '-' strand REF/ALT bases must be complemented.
    """

    snp_pos = int(
        snp[
            "position_1based"
        ]
    )

    strand = reference[
        "strand"
    ]

    full_start = reference[
        "full_start_1based"
    ]

    full_end = reference[
        "full_end_1based"
    ]

    if not (
        full_start
        <= snp_pos
        <= full_end
    ):

        raise ValueError(
            "SNP outside candidate interval."
        )

    if strand == "+":

        oriented_index = (
            snp_pos
            - full_start
        )

        expected_ref = snp[
            "ref"
        ]

        alt_base = snp[
            "alt"
        ]

    else:

        oriented_index = (
            full_end
            - snp_pos
        )

        expected_ref = complement_base(
            snp[
                "ref"
            ]
        )

        alt_base = complement_base(
            snp[
                "alt"
            ]
        )

    reference_sequence = (
        reference[
            "candidate_with_pam"
        ]
    )

    observed_ref = (
        reference_sequence[
            oriented_index
        ]
    )

    if observed_ref != expected_ref:

        return (
            None,
            {
                "observed_ref":
                    observed_ref,

                "expected_ref":
                    expected_ref,

                "oriented_index":
                    oriented_index,
            }
        )

    alt_sequence = (
        reference_sequence[
            :oriented_index
        ]
        + alt_base
        + reference_sequence[
            oriented_index + 1:
        ]
    )

    alt_off20 = (
        alt_sequence[
            :SPACER_LEN
        ]
    )

    alt_pam = (
        alt_sequence[
            SPACER_LEN:
        ]
    )

    alt_mm = hamming(
        reference[
            "sgRNA_20nt"
        ],
        alt_off20,
    )

    alt_pam_valid = (
        pam_matches(
            alt_pam,
            pam_pattern,
        )
    )

    # --------------------------------------------------------
    # Determine whether SNP physically lies in protospacer/PAM
    # --------------------------------------------------------

    if (
        reference[
            "protospacer_start_1based"
        ]
        <= snp_pos
        <= reference[
            "protospacer_end_1based"
        ]
    ):

        snp_region = (
            "protospacer"
        )

    elif (
        reference[
            "pam_start_1based"
        ]
        <= snp_pos
        <= reference[
            "pam_end_1based"
        ]
    ):

        snp_region = (
            "pam"
        )

    else:

        raise RuntimeError(
            "SNP lies inside full candidate but "
            "outside protospacer/PAM coordinates."
        )

    return (
        {
            "off_target_20nt":
                alt_off20,

            "candidate_pam":
                alt_pam,

            "candidate_with_pam":
                alt_sequence,

            "mismatch_count_20nt":
                alt_mm,

            "pam_valid":
                int(
                    alt_pam_valid
                ),

            "snp_region":
                snp_region,
        },
        None,
    )


# ============================================================
# Output schema
# ============================================================

OUTPUT_FIELDS = [

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
    "pam_valid",

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
# Write helpers
# ============================================================

def reference_output_row(
    reference,
    pam_pattern,
):

    pam_valid = pam_matches(
        reference[
            "candidate_pam"
        ],
        pam_pattern,
    )

    if not pam_valid:

        raise ValueError(
            f"Reference candidate "
            f"{reference['candidate_id']} "
            "has a PAM that does not match "
            f"{pam_pattern}: "
            f"{reference['candidate_pam']}"
        )

    return {

        "candidate_id":
            reference[
                "candidate_id"
            ],

        "guide_id":
            reference[
                "guide_id"
            ],

        "candidate_source":
            "reference",

        "sgRNA_20nt":
            reference[
                "sgRNA_20nt"
            ],

        "off_target_20nt":
            reference[
                "off_target_20nt"
            ],

        "candidate_pam":
            reference[
                "candidate_pam"
            ],

        "candidate_with_pam":
            reference[
                "candidate_with_pam"
            ],

        "fasta_id":
            reference[
                "fasta_id"
            ],

        "reference_name":
            reference[
                "reference_name"
            ],

        "strand":
            reference[
                "strand"
            ],

        "full_start_1based":
            reference[
                "full_start_1based"
            ],

        "full_end_1based":
            reference[
                "full_end_1based"
            ],

        "protospacer_start_1based":
            reference[
                "protospacer_start_1based"
            ],

        "protospacer_end_1based":
            reference[
                "protospacer_end_1based"
            ],

        "pam_start_1based":
            reference[
                "pam_start_1based"
            ],

        "pam_end_1based":
            reference[
                "pam_end_1based"
            ],

        "mismatch_count_20nt":
            reference[
                "mismatch_count_20nt"
            ],

        "pam_valid":
            1,

        "snp_id":
            "",

        "snp_position_1based":
            "",

        "snp_region":
            "",

        "snp_ref":
            "",

        "snp_alt":
            "",

        "alt_allele_frequency":
            "",

        "carrier_frequency":
            "",

        "reference_candidate_id":
            reference[
                "candidate_id"
            ],

        "reference_off_target_20nt":
            reference[
                "off_target_20nt"
            ],

        "reference_candidate_pam":
            reference[
                "candidate_pam"
            ],
    }


def population_output_row(
    candidate_id,
    reference,
    snp,
    alt,
):

    return {

        "candidate_id":
            candidate_id,

        "guide_id":
            reference[
                "guide_id"
            ],

        "candidate_source":
            "population_snp",

        "sgRNA_20nt":
            reference[
                "sgRNA_20nt"
            ],

        "off_target_20nt":
            alt[
                "off_target_20nt"
            ],

        "candidate_pam":
            alt[
                "candidate_pam"
            ],

        "candidate_with_pam":
            alt[
                "candidate_with_pam"
            ],

        "fasta_id":
            reference[
                "fasta_id"
            ],

        "reference_name":
            reference[
                "reference_name"
            ],

        "strand":
            reference[
                "strand"
            ],

        "full_start_1based":
            reference[
                "full_start_1based"
            ],

        "full_end_1based":
            reference[
                "full_end_1based"
            ],

        "protospacer_start_1based":
            reference[
                "protospacer_start_1based"
            ],

        "protospacer_end_1based":
            reference[
                "protospacer_end_1based"
            ],

        "pam_start_1based":
            reference[
                "pam_start_1based"
            ],

        "pam_end_1based":
            reference[
                "pam_end_1based"
            ],

        "mismatch_count_20nt":
            alt[
                "mismatch_count_20nt"
            ],

        "pam_valid":
            alt[
                "pam_valid"
            ],

        "snp_id":
            snp[
                "snp_id"
            ],

        "snp_position_1based":
            snp[
                "position_1based"
            ],

        "snp_region":
            alt[
                "snp_region"
            ],

        "snp_ref":
            snp[
                "ref"
            ],

        "snp_alt":
            snp[
                "alt"
            ],

        "alt_allele_frequency":
            snp[
                "alt_allele_frequency"
            ],

        "carrier_frequency":
            snp[
                "carrier_frequency"
            ],

        "reference_candidate_id":
            reference[
                "candidate_id"
            ],

        "reference_off_target_20nt":
            reference[
                "off_target_20nt"
            ],

        "reference_candidate_pam":
            reference[
                "candidate_pam"
            ],
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Apply population SNPs to genome-wide "
            "off-target candidates using an arbitrary "
            "IUPAC PAM pattern."
        )
    )

    parser.add_argument(
        "--candidates",
        required=True,
        type=Path,
        help=(
            "Reference candidate CSV generated by "
            "03_prepare_candidates.py"
        ),
    )

    parser.add_argument(
        "--snps",
        required=True,
        type=Path,
        help=(
            "Prepared population SNP TSV."
        ),
    )

    parser.add_argument(
        "--pam",
        required=True,
        type=str,
        help=(
            "IUPAC PAM pattern, "
            "e.g. NGG or NGTT."
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

    if not args.candidates.exists():

        raise FileNotFoundError(
            args.candidates
        )

    if not args.snps.exists():

        raise FileNotFoundError(
            args.snps
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

    print("=" * 70)
    print(
        "Population SNP application"
    )
    print("=" * 70)

    print(
        f"PAM pattern:             "
        f"{pam_pattern}"
    )

    print(
        f"PAM length:              "
        f"{pam_len}"
    )

    print(
        f"Candidate length:        "
        f"{candidate_len}"
    )

    # ========================================================
    # Load SNP index
    # ========================================================

    snp_index = load_population_snps(
        args.snps
    )

    # ========================================================
    # Candidate input QC
    # ========================================================

    with open(
        args.candidates,
        newline="",
    ) as f:

        reader = csv.DictReader(
            f
        )

        required_candidate_columns = {

            "candidate_id",
            "guide_id",

            "sgRNA_20nt",
            "off_target_20nt",

            "candidate_pam",
            "candidate_with_pam",

            "reference_name",

            "genomic_start_1based",
            "genomic_end_1based",

            "off_target_strand",

            "mismatch_count_20nt",
        }

        missing = (
            required_candidate_columns
            - set(
                reader.fieldnames
                or []
            )
        )

        if missing:

            raise ValueError(
                "Candidate table missing columns: "
                f"{sorted(missing)}"
            )

    # ========================================================
    # Output
    # ========================================================

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_candidates = 0
    snp_overlap_events = 0
    population_candidates = 0

    ref_validation_failed = 0

    protospacer_snp_count = 0
    pam_snp_count = 0

    mismatch_decreased = 0
    mismatch_increased = 0
    mismatch_unchanged = 0

    pam_remains_valid = 0
    pam_destroyed = 0

    alt_over_six = 0

    population_counter = 0

    with open(
        args.candidates,
        newline="",
    ) as fin, open(
        args.output,
        "w",
        newline="",
    ) as fout:

        reader = csv.DictReader(
            fin
        )

        writer = csv.DictWriter(
            fout,
            fieldnames=
                OUTPUT_FIELDS,
        )

        writer.writeheader()

        # ----------------------------------------------------
        # Stream candidates
        # ----------------------------------------------------

        for raw_row in reader:

            reference = (
                normalize_candidate_row(
                    raw_row,
                    pam_pattern,
                )
            )

            reference_candidates += 1

            # -----------------------------------------------
            # Always keep reference candidate
            # -----------------------------------------------

            writer.writerow(
                reference_output_row(
                    reference,
                    pam_pattern,
                )
            )

            # -----------------------------------------------
            # Find population SNPs overlapping this candidate
            # -----------------------------------------------

            overlapping_snps = (
                find_snps_in_interval(
                    snp_index,
                    reference[
                        "fasta_id"
                    ],
                    reference[
                        "full_start_1based"
                    ],
                    reference[
                        "full_end_1based"
                    ],
                )
            )

            if not overlapping_snps:
                continue

            # -----------------------------------------------
            # Generate one ALT candidate per SNP.
            #
            # We deliberately do NOT combine multiple SNPs
            # into synthetic haplotypes.
            # -----------------------------------------------

            for snp in overlapping_snps:

                snp_overlap_events += 1

                alt, error = (
                    apply_single_snp(
                        reference,
                        snp,
                        pam_pattern,
                    )
                )

                if alt is None:

                    ref_validation_failed += 1

                    continue

                population_counter += 1

                population_candidate_id = (
                    f"population_"
                    f"{population_counter:09d}"
                )

                writer.writerow(
                    population_output_row(
                        population_candidate_id,
                        reference,
                        snp,
                        alt,
                    )
                )

                population_candidates += 1

                # -------------------------------------------
                # Statistics
                # -------------------------------------------

                if (
                    alt[
                        "snp_region"
                    ]
                    == "protospacer"
                ):

                    protospacer_snp_count += 1

                elif (
                    alt[
                        "snp_region"
                    ]
                    == "pam"
                ):

                    pam_snp_count += 1

                reference_mm = (
                    reference[
                        "mismatch_count_20nt"
                    ]
                )

                alt_mm = (
                    alt[
                        "mismatch_count_20nt"
                    ]
                )

                if alt_mm < reference_mm:

                    mismatch_decreased += 1

                elif alt_mm > reference_mm:

                    mismatch_increased += 1

                else:

                    mismatch_unchanged += 1

                if alt[
                    "pam_valid"
                ] == 1:

                    pam_remains_valid += 1

                else:

                    pam_destroyed += 1

                if alt_mm > 6:

                    alt_over_six += 1

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)
    print(
        "Population SNP application complete"
    )
    print("=" * 70)

    print(
        f"Reference candidates:       "
        f"{reference_candidates:,}"
    )

    print(
        f"Candidate/SNP overlaps:     "
        f"{snp_overlap_events:,}"
    )

    print(
        f"REF validation failed:      "
        f"{ref_validation_failed:,}"
    )

    print(
        f"Population ALT candidates:  "
        f"{population_candidates:,}"
    )

    print()

    print(
        f"Protospacer SNPs:           "
        f"{protospacer_snp_count:,}"
    )

    print(
        f"PAM SNPs:                   "
        f"{pam_snp_count:,}"
    )

    print()

    print(
        f"Mismatch decreased:         "
        f"{mismatch_decreased:,}"
    )

    print(
        f"Mismatch increased:         "
        f"{mismatch_increased:,}"
    )

    print(
        f"Mismatch unchanged:         "
        f"{mismatch_unchanged:,}"
    )

    print()

    print(
        f"PAM remains valid:          "
        f"{pam_remains_valid:,}"
    )

    print(
        f"PAM destroyed:              "
        f"{pam_destroyed:,}"
    )

    print(
        f"ALT mismatch > 6:           "
        f"{alt_over_six:,}"
    )

    print()

    print(
        f"Total output rows:          "
        f"{reference_candidates + population_candidates:,}"
    )

    print()

    print(
        f"Output: {args.output}"
    )


if __name__ == "__main__":
    main()