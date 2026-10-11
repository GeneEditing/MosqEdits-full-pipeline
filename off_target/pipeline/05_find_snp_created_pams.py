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
    reverse_complement,
)


SPACER_LEN = 20


# ============================================================
# Basic helpers
# ============================================================

def hamming(a, b):

    if len(a) != len(b):

        raise ValueError(
            f"Length mismatch: "
            f"{len(a)} vs {len(b)}"
        )

    return sum(
        x != y
        for x, y in zip(a, b)
    )


# ============================================================
# Guide loading
# ============================================================

def load_guides(path):

    guides = []

    seen_guide_ids = set()

    with open(
        path,
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        required = {
            "guide_id",
            "spacer",
        }

        missing = (
            required
            - set(reader.fieldnames or [])
        )

        if missing:

            raise ValueError(
                "Guide table missing columns: "
                f"{sorted(missing)}"
            )

        for row in reader:

            guide_id = (
                row["guide_id"]
                .strip()
            )

            spacer = (
                row["spacer"]
                .strip()
                .upper()
            )

            if guide_id in seen_guide_ids:

                raise ValueError(
                    f"Duplicate guide_id: "
                    f"{guide_id}"
                )

            seen_guide_ids.add(
                guide_id
            )

            if len(spacer) != SPACER_LEN:

                raise ValueError(
                    f"{guide_id}: "
                    f"spacer length "
                    f"{len(spacer)} != 20"
                )

            invalid = (
                set(spacer)
                - set("ACGT")
            )

            if invalid:

                raise ValueError(
                    f"{guide_id}: invalid "
                    f"spacer bases {invalid}"
                )

            guides.append({
                "guide_id":
                    guide_id,

                "spacer":
                    spacer,
            })

    return guides


# ============================================================
# Population SNP loading
# ============================================================

def load_snps(path):

    snps = []
    needed_contigs = set()

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
                "SNP table missing columns: "
                f"{sorted(missing)}"
            )

        for row in reader:

            position = int(
                row[
                    "position_1based"
                ]
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
                    "Non-biallelic SNP detected: "
                    f"{row['fasta_id']}:"
                    f"{position} "
                    f"{ref}>{alt}"
                )

            row[
                "position_1based"
            ] = position

            row["ref"] = ref
            row["alt"] = alt

            snps.append(row)

            needed_contigs.add(
                row["fasta_id"]
            )

    return (
        snps,
        needed_contigs,
    )


# ============================================================
# FASTA loading
# ============================================================

def load_needed_fasta_sequences(
    fasta_path,
    needed_contigs,
):
    """
    Load only FASTA contigs represented in the SNP table.

    FASTA header:
        >JAFDOQ010000574.1 Aedes albopictus ...

    fasta_id:
        JAFDOQ010000574.1
    """

    sequences = {}

    current_id = None
    current_parts = None

    with open(
        fasta_path
    ) as f:

        for line in f:

            if line.startswith(">"):

                if (
                    current_id is not None
                    and current_parts is not None
                ):

                    sequences[
                        current_id
                    ] = "".join(
                        current_parts
                    ).upper()

                fasta_id = (
                    line[1:]
                    .strip()
                    .split()[0]
                )

                if fasta_id in needed_contigs:

                    current_id = (
                        fasta_id
                    )

                    current_parts = []

                else:

                    current_id = None
                    current_parts = None

            else:

                if current_id is not None:

                    current_parts.append(
                        line.strip()
                    )

        if (
            current_id is not None
            and current_parts is not None
        ):

            sequences[
                current_id
            ] = "".join(
                current_parts
            ).upper()

    missing = (
        needed_contigs
        - set(sequences)
    )

    if missing:

        print(
            f"WARNING: "
            f"{len(missing)} SNP contigs "
            f"not found in FASTA."
        )

        for fasta_id in sorted(
            missing
        )[:10]:

            print(
                "  missing:",
                fasta_id,
            )

    return sequences


# ============================================================
# Sequence fetch
# ============================================================

def fetch_from_memory(
    sequences,
    fasta_id,
    start_1based,
    end_1based,
):
    """
    Return inclusive genomic interval [start, end].
    """

    seq = sequences.get(
        fasta_id
    )

    if seq is None:
        return None

    if start_1based < 1:
        return None

    if end_1based > len(seq):
        return None

    return seq[
        start_1based - 1:
        end_1based
    ]


# ============================================================
# Output schema
# ============================================================

FIELDNAMES = [

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
# Write one guide/candidate pair
# ============================================================

def write_candidate(
    writer,
    candidate_id,
    guide,
    off20,
    alt_pam,
    candidate_with_pam,
    fasta_id,
    strand,
    full_start,
    full_end,
    protospacer_start,
    protospacer_end,
    pam_start,
    pam_end,
    mismatch_count,
    snp,
    reference_pam,
):

    writer.writerow({

        "candidate_id":
            candidate_id,

        "guide_id":
            guide[
                "guide_id"
            ],

        "candidate_source":
            "snp_created_pam",

        "sgRNA_20nt":
            guide[
                "spacer"
            ],

        "off_target_20nt":
            off20,

        "candidate_pam":
            alt_pam,

        "candidate_with_pam":
            candidate_with_pam,

        "fasta_id":
            fasta_id,

        "reference_name":
            fasta_id,

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

        "pam_valid":
            1,

        "snp_id":
            snp[
                "snp_id"
            ],

        "snp_position_1based":
            snp[
                "position_1based"
            ],

        "snp_region":
            "pam",

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

        # ----------------------------------------------------
        # No reference Cas-OFFinder candidate exists because
        # the reference allele did not contain a valid PAM.
        # ----------------------------------------------------

        "reference_candidate_id":
            "",

        # SNP is in PAM, therefore reference protospacer
        # sequence is identical to ALT protospacer.
        "reference_off_target_20nt":
            off20,

        "reference_candidate_pam":
            reference_pam,
    })


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Find population SNPs that create new "
            "PAMs for an arbitrary IUPAC PAM pattern."
        )
    )

    parser.add_argument(
        "--guides",
        required=True,
        type=Path,
        help=(
            "Guide candidate CSV generated by "
            "01_generate_guides.py"
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
        "--fasta",
        required=True,
        type=Path,
        help=(
            "Reference genome FASTA."
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
            "Maximum 20nt guide/off-target "
            "mismatch count."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=(
            "Output SNP-created candidate CSV."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Validation
    # ========================================================

    if not args.guides.exists():

        raise FileNotFoundError(
            args.guides
        )

    if not args.snps.exists():

        raise FileNotFoundError(
            args.snps
        )

    if not args.fasta.exists():

        raise FileNotFoundError(
            args.fasta
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

    print("=" * 70)
    print(
        "SNP-created PAM search"
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

    # ========================================================
    # Load guides
    # ========================================================

    guides = load_guides(
        args.guides
    )

    print(
        f"Guides loaded:               "
        f"{len(guides):,}"
    )

    # ========================================================
    # Load SNPs
    # ========================================================

    snps, needed_contigs = (
        load_snps(
            args.snps
        )
    )

    print(
        f"Population SNPs loaded:      "
        f"{len(snps):,}"
    )

    print(
        f"SNP-containing contigs:      "
        f"{len(needed_contigs):,}"
    )

    # ========================================================
    # Load required FASTA contigs
    # ========================================================

    print()
    print(
        "Loading required FASTA "
        "sequences into memory..."
    )

    sequences = (
        load_needed_fasta_sequences(
            args.fasta,
            needed_contigs,
        )
    )

    total_loaded_bp = sum(
        len(seq)
        for seq in sequences.values()
    )

    print(
        f"FASTA contigs loaded:        "
        f"{len(sequences):,}"
    )

    print(
        f"Total sequence loaded:       "
        f"{total_loaded_bp:,} bp"
    )

    # ========================================================
    # Output
    # ========================================================

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Counters
    # ========================================================

    snps_checked = 0

    ref_validation_failed = 0
    fetch_failed = 0

    pam_creation_events = 0

    plus_created = 0
    minus_created = 0

    candidate_rows = 0

    event_seen = set()

    mismatch_distribution = {}

    # ========================================================
    # Process
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

        for snp in snps:

            snps_checked += 1

            if (
                snps_checked
                % 10000
                == 0
            ):

                print(
                    f"Processed SNPs: "
                    f"{snps_checked:,} / "
                    f"{len(snps):,}"
                )

            fasta_id = (
                snp[
                    "fasta_id"
                ]
            )

            pos = (
                snp[
                    "position_1based"
                ]
            )

            ref = (
                snp["ref"]
            )

            alt = (
                snp["alt"]
            )

            # =================================================
            # PLUS STRAND
            #
            # Guide orientation:
            #
            # [20 nt protospacer][PAM]
            #
            # SNP may occupy any one of the pam_len positions.
            # =================================================

            for pam_offset in range(
                pam_len
            ):

                pam_start = (
                    pos
                    - pam_offset
                )

                pam_end = (
                    pam_start
                    + pam_len
                    - 1
                )

                protospacer_start = (
                    pam_start
                    - SPACER_LEN
                )

                protospacer_end = (
                    pam_start
                    - 1
                )

                full_start = (
                    protospacer_start
                )

                full_end = (
                    pam_end
                )

                if full_start < 1:
                    continue

                full_seq = (
                    fetch_from_memory(
                        sequences,
                        fasta_id,
                        full_start,
                        full_end,
                    )
                )

                if full_seq is None:

                    fetch_failed += 1
                    continue

                if (
                    len(full_seq)
                    != candidate_len
                ):

                    fetch_failed += 1
                    continue

                local_snp_index = (
                    pos
                    - full_start
                )

                # SNP must be inside PAM region.
                if not (
                    SPACER_LEN
                    <= local_snp_index
                    < candidate_len
                ):

                    continue

                if (
                    full_seq[
                        local_snp_index
                    ]
                    != ref
                ):

                    ref_validation_failed += 1
                    continue

                reference_pam = (
                    full_seq[
                        SPACER_LEN:
                    ]
                )

                # Reference PAM must NOT already be valid.
                if pam_matches(
                    reference_pam,
                    pam_pattern,
                ):

                    continue

                alt_seq_list = list(
                    full_seq
                )

                alt_seq_list[
                    local_snp_index
                ] = alt

                alt_candidate = "".join(
                    alt_seq_list
                )

                alt_pam = (
                    alt_candidate[
                        SPACER_LEN:
                    ]
                )

                # ALT must create valid PAM.
                if not pam_matches(
                    alt_pam,
                    pam_pattern,
                ):

                    continue

                event_key = (
                    fasta_id,
                    pam_start,
                    pam_end,
                    "+",
                    snp["snp_id"],
                )

                if event_key in event_seen:
                    continue

                event_seen.add(
                    event_key
                )

                pam_creation_events += 1
                plus_created += 1

                off20 = (
                    alt_candidate[
                        :SPACER_LEN
                    ]
                )

                # ---------------------------------------------
                # Compare new genomic site with every guide.
                # ---------------------------------------------

                for guide in guides:

                    mm = hamming(
                        guide["spacer"],
                        off20,
                    )

                    if (
                        mm
                        > args.max_mismatch
                    ):

                        continue

                    candidate_rows += 1

                    mismatch_distribution[
                        mm
                    ] = (
                        mismatch_distribution.get(
                            mm,
                            0,
                        )
                        + 1
                    )

                    candidate_id = (
                        "snp_pam_"
                        f"{candidate_rows:09d}"
                    )

                    write_candidate(
                        writer=writer,
                        candidate_id=
                            candidate_id,
                        guide=guide,
                        off20=off20,
                        alt_pam=alt_pam,
                        candidate_with_pam=
                            alt_candidate,
                        fasta_id=fasta_id,
                        strand="+",
                        full_start=
                            full_start,
                        full_end=
                            full_end,
                        protospacer_start=
                            protospacer_start,
                        protospacer_end=
                            protospacer_end,
                        pam_start=
                            pam_start,
                        pam_end=
                            pam_end,
                        mismatch_count=
                            mm,
                        snp=snp,
                        reference_pam=
                            reference_pam,
                    )

            # =================================================
            # MINUS STRAND
            #
            # Forward-reference physical orientation:
            #
            # [genomic PAM][20 nt genomic protospacer]
            #
            # Reverse complement gives guide orientation:
            #
            # [20 nt protospacer][PAM]
            # =================================================

            for pam_offset in range(
                pam_len
            ):

                genomic_pam_start = (
                    pos
                    - pam_offset
                )

                genomic_pam_end = (
                    genomic_pam_start
                    + pam_len
                    - 1
                )

                genomic_spacer_start = (
                    genomic_pam_end
                    + 1
                )

                genomic_spacer_end = (
                    genomic_spacer_start
                    + SPACER_LEN
                    - 1
                )

                full_start = (
                    genomic_pam_start
                )

                full_end = (
                    genomic_spacer_end
                )

                if full_start < 1:
                    continue

                full_seq = (
                    fetch_from_memory(
                        sequences,
                        fasta_id,
                        full_start,
                        full_end,
                    )
                )

                if full_seq is None:

                    fetch_failed += 1
                    continue

                if (
                    len(full_seq)
                    != candidate_len
                ):

                    fetch_failed += 1
                    continue

                local_snp_index = (
                    pos
                    - full_start
                )

                # In genomic orientation the PAM is first.
                if not (
                    0
                    <= local_snp_index
                    < pam_len
                ):

                    continue

                if (
                    full_seq[
                        local_snp_index
                    ]
                    != ref
                ):

                    ref_validation_failed += 1
                    continue

                reference_candidate = (
                    reverse_complement(
                        full_seq
                    )
                )

                reference_pam = (
                    reference_candidate[
                        SPACER_LEN:
                    ]
                )

                # Reference PAM must NOT already be valid.
                if pam_matches(
                    reference_pam,
                    pam_pattern,
                ):

                    continue

                # ---------------------------------------------
                # Apply ALT in forward-reference orientation.
                # reverse_complement afterwards automatically
                # converts the ALT base to guide orientation.
                # ---------------------------------------------

                alt_seq_list = list(
                    full_seq
                )

                alt_seq_list[
                    local_snp_index
                ] = alt

                alt_genomic_seq = "".join(
                    alt_seq_list
                )

                alt_candidate = (
                    reverse_complement(
                        alt_genomic_seq
                    )
                )

                alt_pam = (
                    alt_candidate[
                        SPACER_LEN:
                    ]
                )

                if not pam_matches(
                    alt_pam,
                    pam_pattern,
                ):

                    continue

                event_key = (
                    fasta_id,
                    genomic_pam_start,
                    genomic_pam_end,
                    "-",
                    snp["snp_id"],
                )

                if event_key in event_seen:
                    continue

                event_seen.add(
                    event_key
                )

                pam_creation_events += 1
                minus_created += 1

                off20 = (
                    alt_candidate[
                        :SPACER_LEN
                    ]
                )

                for guide in guides:

                    mm = hamming(
                        guide["spacer"],
                        off20,
                    )

                    if (
                        mm
                        > args.max_mismatch
                    ):

                        continue

                    candidate_rows += 1

                    mismatch_distribution[
                        mm
                    ] = (
                        mismatch_distribution.get(
                            mm,
                            0,
                        )
                        + 1
                    )

                    candidate_id = (
                        "snp_pam_"
                        f"{candidate_rows:09d}"
                    )

                    write_candidate(
                        writer=writer,
                        candidate_id=
                            candidate_id,
                        guide=guide,
                        off20=off20,
                        alt_pam=alt_pam,
                        candidate_with_pam=
                            alt_candidate,
                        fasta_id=fasta_id,
                        strand="-",
                        full_start=
                            full_start,
                        full_end=
                            full_end,
                        protospacer_start=
                            genomic_spacer_start,
                        protospacer_end=
                            genomic_spacer_end,
                        pam_start=
                            genomic_pam_start,
                        pam_end=
                            genomic_pam_end,
                        mismatch_count=
                            mm,
                        snp=snp,
                        reference_pam=
                            reference_pam,
                    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)
    print(
        "SNP-created PAM search complete"
    )
    print("=" * 70)

    print(
        f"Population SNPs checked:     "
        f"{snps_checked:,}"
    )

    print(
        f"New {pam_pattern} PAM events: "
        f"{pam_creation_events:,}"
    )

    print(
        f"  Plus strand:               "
        f"{plus_created:,}"
    )

    print(
        f"  Minus strand:              "
        f"{minus_created:,}"
    )

    print()

    print(
        f"Candidates <= "
        f"{args.max_mismatch} mismatch: "
        f"{candidate_rows:,}"
    )

    print()

    print(
        f"REF validation failed:       "
        f"{ref_validation_failed:,}"
    )

    print(
        f"FASTA fetch failed:          "
        f"{fetch_failed:,}"
    )

    print()

    print(
        "Mismatch distribution:"
    )

    for mm in sorted(
        mismatch_distribution
    ):

        print(
            f"  {mm}: "
            f"{mismatch_distribution[mm]:,}"
        )

    print()

    print(
        f"Output: {args.output}"
    )


if __name__ == "__main__":
    main()