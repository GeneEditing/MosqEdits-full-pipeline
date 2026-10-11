#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Frozen configuration
# ============================================================

DEFAULT_ALPHA = 0.75
DEFAULT_MAX_MISMATCH = 6

TOP_N = 5


# ============================================================
# Helpers
# ============================================================

def first_existing_column(df, candidates):

    for col in candidates:

        if col in df.columns:
            return col

    return None


def build_locus_series(df):
    """
    Human-readable genomic location.

    Preferred:
        reference_name:start-end
    """

    chrom_col = first_existing_column(
        df,
        [
            "reference_name",
            "fasta_id",
        ],
    )

    start_col = first_existing_column(
        df,
        [
            "full_start_1based",
            "protospacer_start_1based",
        ],
    )

    end_col = first_existing_column(
        df,
        [
            "full_end_1based",
            "protospacer_end_1based",
        ],
    )

    if chrom_col is None:

        raise ValueError(
            "Cannot build genomic location: "
            "reference_name/fasta_id not found."
        )

    if start_col is None:

        raise ValueError(
            "Cannot build genomic location: "
            "candidate start coordinate not found."
        )

    chrom = (
        df[chrom_col]
        .astype(str)
    )

    start = (
        df[start_col]
        .astype(str)
    )

    if end_col is not None:

        end = (
            df[end_col]
            .astype(str)
        )

        return (
            chrom
            + ":"
            + start
            + "-"
            + end
        )

    return (
        chrom
        + ":"
        + start
    )


def build_locus_key(df):
    """
    Build a unique genomic-locus key.

    REF and population-SNP ALT versions at the same physical
    genomic site therefore share the same locus key.

    Key includes guide_id because the same genomic locus may
    be evaluated against multiple guides.
    """

    required = {
        "guide_id",
        "fasta_id",
        "strand",
        "protospacer_start_1based",
        "protospacer_end_1based",
        "pam_start_1based",
        "pam_end_1based",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Cannot build locus key. "
            f"Missing columns: {sorted(missing)}"
        )

    return (
        df["guide_id"].astype(str)
        + "||"
        + df["fasta_id"].astype(str)
        + "||"
        + df["strand"].astype(str)
        + "||"
        + df[
            "protospacer_start_1based"
        ].astype(str)
        + "-"
        + df[
            "protospacer_end_1based"
        ].astype(str)
        + "||"
        + df[
            "pam_start_1based"
        ].astype(str)
        + "-"
        + df[
            "pam_end_1based"
        ].astype(str)
    )


def topn_aggregate_probability(
    probabilities,
    n=TOP_N,
):
    """
    Approximate probability that at least one of the Top-N
    genomic loci undergoes off-target activity:

        1 - product(1 - P_i)

    Assumption:
        off-target events at different genomic loci are treated
        as approximately independent.

    This is therefore an aggregate probability estimate,
    not a directly measured guide-level probability.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(probabilities) == 0:
        return 0.0

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0,
    )

    probabilities = np.sort(
        probabilities
    )[::-1]

    top = probabilities[
        :min(
            n,
            len(probabilities),
        )
    ]

    return float(
        1.0
        - np.prod(
            1.0 - top
        )
    )


# ============================================================
# Loaders
# ============================================================

def load_predictions(path):

    print(
        f"Loading predictions: "
        f"{path}"
    )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    required = {
        "candidate_id",
        "guide_id",
        "source",

        "ensemble_risk",
        "off_target_probability",

        "reference_candidate_id",
        "alt_allele_frequency",

        "fasta_id",
        "strand",

        "protospacer_start_1based",
        "protospacer_end_1based",

        "pam_start_1based",
        "pam_end_1based",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Prediction file missing columns: "
            f"{sorted(missing)}"
        )

    if (
        df[
            "ensemble_risk"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "NaN detected in ensemble_risk."
        )

    if (
        df[
            "off_target_probability"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "NaN detected in "
            "off_target_probability."
        )

    if not (
        df[
            "off_target_probability"
        ]
        .between(
            0.0,
            1.0,
        )
        .all()
    ):

        raise ValueError(
            "off_target_probability contains "
            "values outside [0,1]."
        )

    df[
        "candidate_id"
    ] = (
        df[
            "candidate_id"
        ]
        .astype(str)
    )

    df[
        "guide_id"
    ] = (
        df[
            "guide_id"
        ]
        .astype(str)
    )

    df[
        "locus_key"
    ] = build_locus_key(
        df
    )

    return df


def load_population_table(path):
    """
    Complete output of 04_apply_population_snps.py.

    IMPORTANT:

    This table contains population ALT candidates that may
    later have been excluded because:

        PAM became invalid
        mismatch > threshold

    Those ALT alleles must still participate in population
    correction with ALT risk/probability = 0.
    """

    print(
        f"Loading complete population table: "
        f"{path}"
    )

    usecols = [
        "candidate_id",
        "guide_id",

        "candidate_source",

        "reference_candidate_id",

        "mismatch_count_20nt",
        "pam_valid",

        "alt_allele_frequency",

        "snp_id",
    ]

    df = pd.read_csv(
        path,
        usecols=usecols,
        low_memory=False,
    )

    df[
        "candidate_id"
    ] = (
        df[
            "candidate_id"
        ]
        .astype(str)
    )

    df[
        "guide_id"
    ] = (
        df[
            "guide_id"
        ]
        .astype(str)
    )

    return df


def load_guides(path):
    """
    Load original guide metadata from 01_generate_guides.py.
    """

    print(
        f"Loading guide metadata: "
        f"{path}"
    )

    required_columns = [
        "guide_id",
        "spacer",
        "pam",
        "target_with_pam",

        "strand",
        "chrom",

        "genomic_spacer_start",
        "genomic_spacer_end",

        "genomic_pam_start",
        "genomic_pam_end",
    ]

    df = pd.read_csv(
        path,
        usecols=required_columns,
    )

    df[
        "guide_id"
    ] = (
        df[
            "guide_id"
        ]
        .astype(str)
    )

    if (
        df[
            "guide_id"
        ]
        .duplicated()
        .any()
    ):

        duplicated = (
            df.loc[
                df[
                    "guide_id"
                ].duplicated(),
                "guide_id"
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate guide_id detected "
            "in guide table: "
            f"{duplicated[:20]}"
        )

    return df


# ============================================================
# Highest-risk candidate per guide
# ============================================================

def summarize_highest_risk_candidate(
    pred,
):
    """
    Select the single scored candidate with the highest
    calibrated off-target probability for each guide.

    This remains a candidate-level metric and is NOT
    interpreted as:

        probability that the guide has any off-target
        anywhere in the genome.
    """

    idx = (
        pred
        .groupby(
            "guide_id"
        )[
            "off_target_probability"
        ]
        .idxmax()
    )

    highest = (
        pred.loc[
            idx
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    sequence_col = (
        first_existing_column(
            highest,
            [
                "off_target_20nt",
                "off_target",
                "candidate_20nt",
                "candidate_sequence",
            ],
        )
    )

    mismatch_col = (
        first_existing_column(
            highest,
            [
                "mismatch_count_20nt",
                "mismatch_count",
            ],
        )
    )

    highest[
        "highest_risk_genomic_location"
    ] = build_locus_series(
        highest
    )

    result = pd.DataFrame({

        "guide_id":
            highest[
                "guide_id"
            ].astype(str),

        "highest_off_target_probability":
            highest[
                "off_target_probability"
            ].astype(float),

        "highest_risk_off_target_sequence":
            (
                highest[
                    sequence_col
                ].astype(str)

                if sequence_col
                is not None

                else pd.Series(
                    pd.NA,
                    index=highest.index,
                    dtype="object",
                )
            ),

        "highest_risk_mismatch_count":
            (
                highest[
                    mismatch_col
                ]

                if mismatch_col
                is not None

                else pd.Series(
                    pd.NA,
                    index=highest.index,
                )
            ),

        "highest_risk_genomic_location":
            highest[
                "highest_risk_genomic_location"
            ],
    })

    return result


# ============================================================
# Population-adjusted locus probabilities
# ============================================================

def build_population_adjusted_loci(
    pred,
    population_table,
    max_mismatch,
):
    """
    Construct one effective calibrated probability per
    guide x genomic locus.

    ----------------------------------------------------------
    Existing reference locus
    ----------------------------------------------------------

    Start with:

        P_locus = P_REF

    For each population SNP event:

        P_locus += AF * (P_ALT - P_REF)

    Valid ALT:
        use calibrated ALT probability

    Invalid ALT:
        P_ALT = 0

    Invalid means:
        PAM destroyed
        OR
        mismatch > max_mismatch
        OR
        ALT candidate was otherwise unavailable for scoring

    ----------------------------------------------------------
    SNP-created PAM
    ----------------------------------------------------------

    Reference allele has no valid PAM:

        contribution = AF * P_ALT

    ----------------------------------------------------------
    Important
    ----------------------------------------------------------

    This uses the same first-order population correction logic
    as the frozen population raw-risk aggregation.

    It does not explicitly model haplotypes or dependence among
    multiple variants at the same locus.
    """

    # ========================================================
    # Reference candidates
    # ========================================================

    reference = pred.loc[
        pred[
            "source"
        ] == "reference"
    ].copy()

    if reference.empty:

        raise ValueError(
            "No reference candidates found."
        )

    reference[
        "reference_key"
    ] = (
        reference[
            "guide_id"
        ].astype(str)
        + "||"
        + reference[
            "reference_candidate_id"
        ].astype(str)
    )

    if (
        reference[
            "reference_key"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Duplicate reference candidate keys "
            "during Top5 construction."
        )

    ref_probability_map = dict(
        zip(
            reference[
                "reference_key"
            ],
            reference[
                "off_target_probability"
            ].astype(float),
        )
    )

    ref_locus_map = dict(
        zip(
            reference[
                "reference_key"
            ],
            reference[
                "locus_key"
            ],
        )
    )

    # --------------------------------------------------------
    # Initialize each reference genomic locus with P_REF.
    # --------------------------------------------------------

    locus_probability = dict(
        zip(
            reference[
                "locus_key"
            ],
            reference[
                "off_target_probability"
            ].astype(float),
        )
    )

    locus_guide = dict(
        zip(
            reference[
                "locus_key"
            ],
            reference[
                "guide_id"
            ],
        )
    )

    # ========================================================
    # Scored population ALT candidates
    # ========================================================

    scored_population = pred.loc[
        pred[
            "source"
        ] == "population_snp",
        [
            "candidate_id",
            "off_target_probability",
        ],
    ].copy()

    alt_probability_map = dict(
        zip(
            scored_population[
                "candidate_id"
            ].astype(str),
            scored_population[
                "off_target_probability"
            ].astype(float),
        )
    )

    # ========================================================
    # Existing-locus population SNPs
    # ========================================================

    snp_rows = (
        population_table.loc[
            population_table[
                "candidate_source"
            ] == "population_snp"
        ]
        .copy()
    )

    existing_events = 0
    valid_alt_scored = 0
    invalid_alt_zero = 0

    missing_reference = 0
    invalid_af = 0

    seen_events = set()

    for row in snp_rows.itertuples(
        index=False
    ):

        guide_id = str(
            row.guide_id
        )

        ref_id = str(
            row.reference_candidate_id
        )

        ref_key = (
            f"{guide_id}||{ref_id}"
        )

        # Intended on-target reference loci were removed before
        # scoring and therefore correctly disappear here.
        if (
            ref_key
            not in ref_probability_map
        ):

            missing_reference += 1
            continue

        event_key = (
            guide_id,
            ref_id,
            str(
                row.snp_id
            ),
        )

        if (
            event_key
            in seen_events
        ):
            continue

        seen_events.add(
            event_key
        )

        try:

            af = float(
                row.alt_allele_frequency
            )

        except (
            TypeError,
            ValueError,
        ):

            invalid_af += 1
            continue

        if not (
            0.0
            <= af
            <= 1.0
        ):

            invalid_af += 1
            continue

        ref_probability = float(
            ref_probability_map[
                ref_key
            ]
        )

        alt_candidate_id = str(
            row.candidate_id
        )

        pam_valid = (
            int(
                row.pam_valid
            )
            == 1
        )

        mm_valid = (
            int(
                row.mismatch_count_20nt
            )
            <= max_mismatch
        )

        if (
            pam_valid
            and mm_valid
            and alt_candidate_id
            in alt_probability_map
        ):

            alt_probability = float(
                alt_probability_map[
                    alt_candidate_id
                ]
            )

            valid_alt_scored += 1

        else:

            alt_probability = 0.0

            invalid_alt_zero += 1

        locus_key = (
            ref_locus_map[
                ref_key
            ]
        )

        locus_probability[
            locus_key
        ] += (
            af
            * (
                alt_probability
                - ref_probability
            )
        )

        existing_events += 1

    # --------------------------------------------------------
    # Numerical protection
    # --------------------------------------------------------

    for key in list(
        locus_probability
    ):

        locus_probability[
            key
        ] = float(
            np.clip(
                locus_probability[
                    key
                ],
                0.0,
                1.0,
            )
        )

    # ========================================================
    # SNP-created PAM candidates
    # ========================================================

    created = pred.loc[
        pred[
            "source"
        ] == "snp_created_pam"
    ].copy()

    created_events = 0
    invalid_created_af = 0

    for row in created.itertuples(
        index=False
    ):

        try:

            af = float(
                row.alt_allele_frequency
            )

        except (
            TypeError,
            ValueError,
        ):

            invalid_created_af += 1
            continue

        if not (
            0.0
            <= af
            <= 1.0
        ):

            invalid_created_af += 1
            continue

        contribution = (
            af
            * float(
                row.off_target_probability
            )
        )

        locus_key = str(
            row.locus_key
        )

        if (
            locus_key
            not in locus_probability
        ):

            locus_probability[
                locus_key
            ] = 0.0

            locus_guide[
                locus_key
            ] = str(
                row.guide_id
            )

        locus_probability[
            locus_key
        ] += contribution

        locus_probability[
            locus_key
        ] = float(
            np.clip(
                locus_probability[
                    locus_key
                ],
                0.0,
                1.0,
            )
        )

        created_events += 1

    # ========================================================
    # Final locus table
    # ========================================================

    rows = []

    for (
        locus_key,
        probability,
    ) in locus_probability.items():

        rows.append({

            "guide_id":
                locus_guide[
                    locus_key
                ],

            "locus_key":
                locus_key,

            "population_adjusted_probability":
                probability,
        })

    loci = pd.DataFrame(
        rows
    )

    stats = {

        "reference_loci":
            len(reference),

        "existing_events":
            existing_events,

        "valid_alt_scored":
            valid_alt_scored,

        "invalid_alt_zero":
            invalid_alt_zero,

        "missing_reference":
            missing_reference,

        "invalid_af":
            invalid_af,

        "created_events":
            created_events,

        "invalid_created_af":
            invalid_created_af,

        "effective_loci":
            len(loci),
    }

    return (
        loci,
        stats,
    )


# ============================================================
# Top5 aggregate probability
# ============================================================

def calculate_top5_probabilities(
    loci,
):
    """
    For every guide:

        1. sort distinct population-adjusted genomic loci
           by probability

        2. take Top5

        3. calculate:

           P = 1 - product(1 - P_i)
    """

    rows = []

    for (
        guide_id,
        group,
    ) in loci.groupby(
        "guide_id"
    ):

        probabilities = (
            group[
                "population_adjusted_probability"
            ]
            .astype(float)
            .to_numpy()
        )

        probability = (
            topn_aggregate_probability(
                probabilities,
                TOP_N,
            )
        )

        rows.append({

            "guide_id":
                str(
                    guide_id
                ),

            "top5_aggregate_off_target_probability":
                probability,
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Population-aware guide-level aggregation "
            "for PAM-agnostic off-target predictions."
        )
    )

    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help=(
            "Candidate predictions produced by "
            "08_score_candidates.py."
        ),
    )

    parser.add_argument(
        "--population-table",
        "--step05-population",
        dest="population_table",
        required=True,
        type=Path,
        help=(
            "Complete output of "
            "04_apply_population_snps.py before "
            "invalid ALT candidates are removed."
        ),
    )

    parser.add_argument(
        "--guides",
        required=True,
        type=Path,
        help=(
            "Guide table produced by "
            "01_generate_guides.py."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
    )

    parser.add_argument(
        "--max-mismatch",
        type=int,
        default=DEFAULT_MAX_MISMATCH,
    )

    args = parser.parse_args()

    # ========================================================
    # Load
    # ========================================================

    pred = load_predictions(
        args.predictions
    )

    population_table = (
        load_population_table(
            args.population_table
        )
    )

    guide_info = load_guides(
        args.guides
    )

    print()
    print("=" * 70)
    print("Input summary")
    print("=" * 70)

    print(
        f"Prediction rows:           "
        f"{len(pred):,}"
    )

    print(
        f"Guides in predictions:     "
        f"{pred['guide_id'].nunique():,}"
    )

    print(
        f"Guides in guide table:     "
        f"{guide_info['guide_id'].nunique():,}"
    )

    print()

    print(
        "Prediction sources:"
    )

    print(
        pred[
            "source"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # Highest candidate probability
    # ========================================================

    highest_risk = (
        summarize_highest_risk_candidate(
            pred
        )
    )

    # ========================================================
    # Reference candidates
    # ========================================================

    reference = pred.loc[
        pred[
            "source"
        ] == "reference"
    ].copy()

    if reference.empty:

        raise ValueError(
            "No reference candidates found."
        )

    if (
        reference[
            "reference_candidate_id"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "reference_candidate_id missing "
            "from reference candidates."
        )

    reference[
        "reference_key"
    ] = (
        reference[
            "guide_id"
        ].astype(str)
        + "||"
        + reference[
            "reference_candidate_id"
        ].astype(str)
    )

    if (
        reference[
            "reference_key"
        ]
        .duplicated()
        .any()
    ):

        n_duplicate = int(
            reference[
                "reference_key"
            ]
            .duplicated()
            .sum()
        )

        raise ValueError(
            "Duplicate reference candidate "
            f"keys: {n_duplicate}"
        )

    reference_risk = dict(
        zip(
            reference[
                "reference_key"
            ],
            reference[
                "ensemble_risk"
            ],
        )
    )

    # ========================================================
    # Reference raw-risk aggregation
    # ========================================================

    guide_result = (
        reference
        .groupby(
            "guide_id"
        )
        .agg(

            reference_risk_sum=(
                "ensemble_risk",
                "sum",
            ),

            reference_candidate_count=(
                "candidate_id",
                "count",
            ),

            reference_risk_max=(
                "ensemble_risk",
                "max",
            ),
        )
        .reset_index()
    )

    reference_count = (
        guide_result[
            "reference_candidate_count"
        ]
        .astype(float)
        .clip(
            lower=1.0
        )
    )

    guide_result[
        "reference_off_target_score"
    ] = (
        guide_result[
            "reference_risk_sum"
        ]
        / np.power(
            reference_count,
            args.alpha,
        )
    )

    population_risk_by_guide = dict(
        zip(
            guide_result[
                "guide_id"
            ],
            guide_result[
                "reference_risk_sum"
            ].astype(float),
        )
    )

    # ========================================================
    # Existing-locus SNP raw-risk correction
    # ========================================================

    snp_population = (
        population_table.loc[
            population_table[
                "candidate_source"
            ] == "population_snp"
        ]
        .copy()
    )

    print()

    print(
        f"Existing-locus SNP rows:   "
        f"{len(snp_population):,}"
    )

    scored_population = pred.loc[
        pred[
            "source"
        ] == "population_snp",
        [
            "candidate_id",
            "ensemble_risk",
        ],
    ].copy()

    alt_risk_map = dict(
        zip(
            scored_population[
                "candidate_id"
            ].astype(str),
            scored_population[
                "ensemble_risk"
            ].astype(float),
        )
    )

    existing_snp_events = 0
    valid_alt_scored = 0
    invalid_alt_zero = 0

    missing_reference = 0
    invalid_af = 0

    total_existing_adjustment = 0.0

    seen_reference_events = set()

    for row in (
        snp_population
        .itertuples(
            index=False
        )
    ):

        guide_id = str(
            row.guide_id
        )

        ref_id = str(
            row.reference_candidate_id
        )

        ref_key = (
            f"{guide_id}||{ref_id}"
        )

        # Intended on-target was already removed.
        if (
            ref_key
            not in reference_risk
        ):

            missing_reference += 1
            continue

        event_key = (
            guide_id,
            ref_id,
            str(
                row.snp_id
            ),
        )

        if (
            event_key
            in seen_reference_events
        ):

            continue

        seen_reference_events.add(
            event_key
        )

        try:

            af = float(
                row.alt_allele_frequency
            )

        except (
            TypeError,
            ValueError,
        ):

            invalid_af += 1
            continue

        if not (
            0.0
            <= af
            <= 1.0
        ):

            invalid_af += 1
            continue

        ref_risk = float(
            reference_risk[
                ref_key
            ]
        )

        pam_valid = (
            int(
                row.pam_valid
            )
            == 1
        )

        mm_valid = (
            int(
                row.mismatch_count_20nt
            )
            <= args.max_mismatch
        )

        alt_candidate_id = str(
            row.candidate_id
        )

        if (
            pam_valid
            and mm_valid
            and alt_candidate_id
            in alt_risk_map
        ):

            alt_risk = float(
                alt_risk_map[
                    alt_candidate_id
                ]
            )

            valid_alt_scored += 1

        else:

            alt_risk = 0.0

            invalid_alt_zero += 1

        adjustment = (
            af
            * (
                alt_risk
                - ref_risk
            )
        )

        population_risk_by_guide[
            guide_id
        ] += adjustment

        total_existing_adjustment += (
            adjustment
        )

        existing_snp_events += 1

    # ========================================================
    # SNP-created PAM raw-risk correction
    # ========================================================

    created = pred.loc[
        pred[
            "source"
        ] == "snp_created_pam"
    ].copy()

    created_events = 0
    invalid_created_af = 0

    total_created_risk = 0.0

    for row in created.itertuples(
        index=False
    ):

        guide_id = str(
            row.guide_id
        )

        try:

            af = float(
                row.alt_allele_frequency
            )

        except (
            TypeError,
            ValueError,
        ):

            invalid_created_af += 1
            continue

        if not (
            0.0
            <= af
            <= 1.0
        ):

            invalid_created_af += 1
            continue

        contribution = (
            af
            * float(
                row.ensemble_risk
            )
        )

        population_risk_by_guide[
            guide_id
        ] += contribution

        total_created_risk += (
            contribution
        )

        created_events += 1

    # ========================================================
    # Final population raw-risk score
    # ========================================================

    guide_result[
        "population_risk_sum"
    ] = (
        guide_result[
            "guide_id"
        ]
        .map(
            population_risk_by_guide
        )
        .astype(float)
    )

    guide_result[
        "population_off_target_score"
    ] = (
        guide_result[
            "population_risk_sum"
        ]
        / np.power(
            guide_result[
                "reference_candidate_count"
            ].astype(float),
            args.alpha,
        )
    )

    # ========================================================
    # Population-adjusted Top5 probability
    # ========================================================

    (
        effective_loci,
        locus_stats,
    ) = (
        build_population_adjusted_loci(
            pred=pred,
            population_table=population_table,
            max_mismatch=args.max_mismatch,
        )
    )

    top5_result = (
        calculate_top5_probabilities(
            effective_loci
        )
    )

    # ========================================================
    # Merge guide metadata
    # ========================================================

    guide_result = (
        guide_result.merge(
            guide_info,
            on="guide_id",
            how="left",
            validate="one_to_one",
        )
    )

    guide_result = (
        guide_result.merge(
            highest_risk,
            on="guide_id",
            how="left",
            validate="one_to_one",
        )
    )

    guide_result = (
        guide_result.merge(
            top5_result,
            on="guide_id",
            how="left",
            validate="one_to_one",
        )
    )

    # ========================================================
    # Merge QC
    # ========================================================

    if (
        guide_result[
            "spacer"
        ]
        .isna()
        .any()
    ):

        missing_guides = (
            guide_result.loc[
                guide_result[
                    "spacer"
                ].isna(),
                "guide_id",
            ]
            .tolist()
        )

        raise ValueError(
            "Missing guide metadata for: "
            f"{missing_guides[:20]}"
        )

    if (
        guide_result[
            "highest_off_target_probability"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "Missing "
            "highest_off_target_probability."
        )

    if (
        guide_result[
            "top5_aggregate_off_target_probability"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "Missing "
            "top5_aggregate_off_target_probability."
        )

    # ========================================================
    # Ranking
    # ========================================================

    # Primary ranking metric:
    #
    # lower top5_aggregate_off_target_probability = safer
    #
    # Deterministic tie-breakers:
    #   1. lower highest single-site probability
    #   2. lower population raw-risk score (audit-only)
    #   3. guide_id

    guide_result = (
        guide_result
        .sort_values(
            [
                "top5_aggregate_off_target_probability",
                "highest_off_target_probability",
                "population_off_target_score",
                "guide_id",
            ],
            ascending=[True, True, True, True],
        )
        .reset_index(
            drop=True
        )
    )

    guide_result.insert(
        0,
        "off_target_rank",
        np.arange(
            1,
            len(guide_result) + 1,
        ),
    )

    # ========================================================
    # Full guide-level output
    # ========================================================

    output_columns = [
        "off_target_rank",

        "guide_id",

        "spacer",
        "pam",
        "chrom",
        "strand",

        "genomic_spacer_start",
        "genomic_spacer_end",

        "genomic_pam_start",
        "genomic_pam_end",

        "highest_off_target_probability",

        "top5_aggregate_off_target_probability",

        "highest_risk_off_target_sequence",

        "highest_risk_mismatch_count",

        "highest_risk_genomic_location",

        "reference_candidate_count",
    ]

    final_output = (
        guide_result[
            output_columns
        ]
        .copy()
    )

    # ========================================================
    # Final QC
    # ========================================================

    probability_columns = [
        "highest_off_target_probability",
        "top5_aggregate_off_target_probability",
    ]

    if (
        final_output[
            probability_columns
        ]
        .isna()
        .any()
        .any()
    ):

        raise RuntimeError(
            "NaN detected in final "
            "probability output."
        )

    for col in probability_columns:

        if not (
            final_output[
                col
            ]
            .between(
                0.0,
                1.0,
            )
            .all()
        ):

            raise RuntimeError(
                f"{col} contains values "
                "outside [0,1]."
            )

    if (
        guide_result[
            "population_off_target_score"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "NaN detected in "
            "population_off_target_score."
        )

    if (
        guide_result[
            "population_off_target_score"
        ] < 0
    ).any():

        raise RuntimeError(
            "Negative "
            "population_off_target_score "
            "detected."
        )

    if (
        final_output[
            "reference_candidate_count"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "NaN detected in "
            "reference_candidate_count."
        )

    # ========================================================
    # Save full guide-level result
    # ========================================================

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_output.to_csv(
        args.output,
        index=False,
    )

    # ========================================================
    # NEW:
    # Save Top-20 final guide table
    # ========================================================

    top20_output = (
        guide_result
        .head(20)
        .copy()
    )

    # Rename user-facing spacer column.
    top20_output = (
        top20_output.rename(
            columns={
                "spacer": "gRNA_20nt",
                "off_target_rank": "rank",
            }
        )
    )

    top20_columns = [
        "rank",

        "guide_id",

        "gRNA_20nt",

        "pam",

        "highest_off_target_probability",

        "top5_aggregate_off_target_probability",

        "chrom",

        "strand",

        "genomic_spacer_start",

        "genomic_spacer_end",

        "genomic_pam_start",

        "genomic_pam_end",
    ]

    top20_output = (
        top20_output[
            top20_columns
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Top20 QC
    # --------------------------------------------------------

    expected_top_n = min(
        20,
        len(guide_result),
    )

    if len(top20_output) != expected_top_n:

        raise RuntimeError(
            "Unexpected number of rows "
            "in Top20 output."
        )

    if (
        top20_output[
            "guide_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate guide_id detected "
            "in Top20 output."
        )

    if (
        top20_output[
            "gRNA_20nt"
        ]
        .astype(str)
        .str.len()
        .ne(20)
        .any()
    ):

        raise RuntimeError(
            "Non-20nt gRNA detected "
            "in Top20 output."
        )

    top20_required_non_null = [
        "rank",
        "guide_id",
        "gRNA_20nt",
        "pam",

        "highest_off_target_probability",
        "top5_aggregate_off_target_probability",
        "chrom",
        "strand",

        "genomic_spacer_start",
        "genomic_spacer_end",

        "genomic_pam_start",
        "genomic_pam_end",
    ]

    if (
        top20_output[
            top20_required_non_null
        ]
        .isna()
        .any()
        .any()
    ):

        raise RuntimeError(
            "NaN detected in Top20 output."
        )

    # --------------------------------------------------------
    # Top20 path:
    #
    # white_AalbF3_NGTT_guide_offtarget_scores.csv
    #
    # becomes:
    #
    # white_AalbF3_NGTT_guide_offtarget_scores_top20.csv
    # --------------------------------------------------------

    top20_path = (
        args.output.parent
        / (
            args.output.stem
            + "_top20.csv"
        )
    )

    top20_output.to_csv(
        top20_path,
        index=False,
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)
    print(
        "Population-aware aggregation complete"
    )
    print("=" * 70)

    print(
        f"Guides scored:                 "
        f"{len(final_output):,}"
    )

    print(
        f"Reference off-target rows:     "
        f"{len(reference):,}"
    )

    print()

    print(
        f"Existing-locus SNP events:     "
        f"{existing_snp_events:,}"
    )

    print(
        f"  Valid ALT scored:            "
        f"{valid_alt_scored:,}"
    )

    print(
        f"  Invalid ALT treated as 0:    "
        f"{invalid_alt_zero:,}"
    )

    print(
        f"  Missing reference locus:     "
        f"{missing_reference:,}"
    )

    print(
        f"  Invalid AF:                  "
        f"{invalid_af:,}"
    )

    print()

    print(
        f"SNP-created PAM candidates:    "
        f"{created_events:,}"
    )

    print(
        f"Invalid created-PAM AF:        "
        f"{invalid_created_af:,}"
    )

    print()

    print(
        f"Existing-SNP risk adjustment:  "
        f"{total_existing_adjustment:.6f}"
    )

    print(
        f"Created-PAM added risk:        "
        f"{total_created_risk:.6f}"
    )

    print()

    print(
        f"Alpha:                         "
        f"{args.alpha}"
    )

    print(
        f"Maximum mismatch:              "
        f"{args.max_mismatch}"
    )

    # ========================================================
    # Top5 construction audit
    # ========================================================

    print()

    print(
        "Population-adjusted Top5 construction:"
    )

    print(
        f"  Reference loci:              "
        f"{locus_stats['reference_loci']:,}"
    )

    print(
        f"  Existing SNP events:         "
        f"{locus_stats['existing_events']:,}"
    )

    print(
        f"  Valid ALT scored:            "
        f"{locus_stats['valid_alt_scored']:,}"
    )

    print(
        f"  Invalid ALT -> probability 0:"
        f" {locus_stats['invalid_alt_zero']:,}"
    )

    print(
        f"  Missing reference loci:      "
        f"{locus_stats['missing_reference']:,}"
    )

    print(
        f"  SNP-created PAM events:      "
        f"{locus_stats['created_events']:,}"
    )

    print(
        f"  Effective loci:              "
        f"{locus_stats['effective_loci']:,}"
    )

    # ========================================================
    # Guide-level metric definitions
    # ========================================================

    print()

    print(
        "Guide-level outputs:"
    )

    print(
        "  highest_off_target_probability"
    )

    print(
        "    = maximum calibrated "
        "candidate-level probability"
    )

    print()

    print(
        "  top5_aggregate_off_target_probability"
    )

    print(
        "    = 1 - product(1 - P_i) "
        "for the five highest "
        "population-adjusted genomic loci"
    )

    print(
        "    = approximate aggregate probability "
        "under the independence assumption"
    )

    print()

    print(
        "Ranking:"
    )

    print(
        "  top5_aggregate_off_target_probability "
        "(ascending; lower = safer)"
    )

    # ========================================================
    # Output paths
    # ========================================================

    print()

    print(
        f"Full output:  {args.output}"
    )

    print(
        f"Top20 output: {top20_path}"
    )

    # ========================================================
    # Top20 display
    # ========================================================

    print()

    print(
        "Top 20 safest guides:"
    )

    display_output = (
        top20_output
        .copy()
    )

    display_output[
        [
            "highest_off_target_probability",
            "top5_aggregate_off_target_probability",
        ]
    ] = (
        display_output[
            [
                "highest_off_target_probability",
                "top5_aggregate_off_target_probability",
            ]
        ]
        .round(4)
    )

    print(
        display_output
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
