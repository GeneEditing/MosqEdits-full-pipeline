#!/usr/bin/env python3


IUPAC_BASES = {
    "A": {"A"},
    "C": {"C"},
    "G": {"G"},
    "T": {"T"},

    "R": {"A", "G"},
    "Y": {"C", "T"},
    "S": {"G", "C"},
    "W": {"A", "T"},
    "K": {"G", "T"},
    "M": {"A", "C"},

    "B": {"C", "G", "T"},
    "D": {"A", "G", "T"},
    "H": {"A", "C", "T"},
    "V": {"A", "C", "G"},

    "N": {"A", "C", "G", "T"},
}


COMPLEMENT = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
}


def normalize_dna(seq):
    """
    Normalize DNA sequence:
    - strip whitespace
    - uppercase
    """

    if seq is None:
        raise ValueError(
            "Sequence cannot be None."
        )

    return (
        str(seq)
        .strip()
        .upper()
    )


def validate_dna(seq):
    """
    Validate ordinary DNA sequence.

    Only A/C/G/T are allowed.
    """

    seq = normalize_dna(seq)

    invalid = (
        set(seq)
        - {"A", "C", "G", "T"}
    )

    if invalid:
        raise ValueError(
            f"Invalid DNA bases: "
            f"{sorted(invalid)}"
        )

    return seq


def normalize_pam_pattern(
    pam_pattern
):
    """
    Normalize and validate an IUPAC PAM pattern.

    Examples:
        NGG
        NGTT
        TTTV
        NNGRRT
    """

    if pam_pattern is None:
        raise ValueError(
            "PAM pattern cannot be None."
        )

    pam_pattern = (
        str(pam_pattern)
        .strip()
        .upper()
    )

    if len(pam_pattern) == 0:
        raise ValueError(
            "PAM pattern cannot be empty."
        )

    invalid = (
        set(pam_pattern)
        - set(IUPAC_BASES)
    )

    if invalid:
        raise ValueError(
            "Invalid IUPAC characters "
            f"in PAM pattern: "
            f"{sorted(invalid)}"
        )

    return pam_pattern


def pam_length(
    pam_pattern
):
    """
    Return PAM length.
    """

    pam_pattern = (
        normalize_pam_pattern(
            pam_pattern
        )
    )

    return len(
        pam_pattern
    )


def pam_matches(
    pam_sequence,
    pam_pattern,
):
    """
    Return True if a concrete DNA PAM sequence
    matches an IUPAC PAM pattern.

    Examples:
        pam_matches("AGG", "NGG")   -> True
        pam_matches("AGTT", "NGTT") -> True
        pam_matches("TGTT", "NGTT") -> True
        pam_matches("AGCT", "NGTT") -> False
    """

    pam_sequence = (
        normalize_dna(
            pam_sequence
        )
    )

    pam_pattern = (
        normalize_pam_pattern(
            pam_pattern
        )
    )

    if len(
        pam_sequence
    ) != len(
        pam_pattern
    ):

        return False

    for base, code in zip(
        pam_sequence,
        pam_pattern,
    ):

        allowed = (
            IUPAC_BASES[
                code
            ]
        )

        if base not in allowed:
            return False

    return True


def reverse_complement(
    seq
):
    """
    Reverse complement an A/C/G/T DNA sequence.
    """

    seq = validate_dna(
        seq
    )

    return "".join(
        COMPLEMENT[base]
        for base in reversed(
            seq
        )
    )


def candidate_length(
    pam_pattern,
    spacer_length=20,
):
    """
    Total candidate length:
        spacer + PAM
    """

    return (
        spacer_length
        + pam_length(
            pam_pattern
        )
    )


def split_candidate(
    candidate_with_pam,
    pam_pattern,
    spacer_length=20,
):
    """
    Split:
        candidate_with_pam
    into:
        off_target_20nt
        candidate_pam

    Length is determined dynamically from PAM pattern.
    """

    candidate_with_pam = (
        validate_dna(
            candidate_with_pam
        )
    )

    pam_pattern = (
        normalize_pam_pattern(
            pam_pattern
        )
    )

    expected_length = (
        candidate_length(
            pam_pattern,
            spacer_length=
                spacer_length,
        )
    )

    if len(
        candidate_with_pam
    ) != expected_length:

        raise ValueError(
            "Candidate length mismatch: "
            f"got "
            f"{len(candidate_with_pam)}, "
            f"expected "
            f"{expected_length} "
            f"for PAM "
            f"{pam_pattern}."
        )

    spacer = (
        candidate_with_pam[
            :spacer_length
        ]
    )

    pam = (
        candidate_with_pam[
            spacer_length:
        ]
    )

    return (
        spacer,
        pam,
    )


def validate_candidate_pam(
    candidate_with_pam,
    pam_pattern,
    spacer_length=20,
):
    """
    Validate candidate length and PAM match.

    Returns:
        spacer
        pam
        pam_valid
    """

    spacer, pam = split_candidate(
        candidate_with_pam,
        pam_pattern,
        spacer_length=
            spacer_length,
    )

    return (
        spacer,
        pam,
        pam_matches(
            pam,
            pam_pattern,
        ),
    )