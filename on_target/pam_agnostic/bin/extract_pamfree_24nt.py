#!/usr/bin/env python3
"""Extract PAM-free 24-nt CRISPRon model inputs for an arbitrary PAM pattern.

For every target on either strand, the model input is normalized to guide
orientation and has this layout:

    4-nt upstream context + 20-nt spacer

The PAM is only used to locate candidate sites; it is retained in the output
table but is not included in the 24-nt CNN input, since the PAM-free model
does not depend on the PAM identity.
"""

import argparse
import csv
import re
import sys
from pathlib import Path


SPACER_LENGTH = 20
UPSTREAM_LENGTH = 4
VALID_BASES = frozenset("ACGTN")
UNAMBIGUOUS_BASES = frozenset("ACGT")
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")

# IUPAC nucleotide ambiguity codes, used to match PAM patterns like NGG/NGTT/TTTV.
IUPAC_CODES = {
    "A": "A", "C": "C", "G": "G", "T": "T", "U": "T",
    "R": "AG", "Y": "CT", "S": "GC", "W": "AT", "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT",
}


OUTPUT_COLUMNS = [
    "guide_id",
    "record_id",
    "chrom",
    "strand",
    "upstream4",
    "spacer",
    "pam",
    "input_24nt",
    "target_with_pam",
    "local_context_start",
    "local_context_end",
    "local_spacer_start",
    "local_spacer_end",
    "local_pam_start",
    "local_pam_end",
    "genomic_context_start",
    "genomic_context_end",
    "genomic_spacer_start",
    "genomic_spacer_end",
    "genomic_pam_start",
    "genomic_pam_end",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Search both strands for 20-nt spacers adjacent to an arbitrary "
            "PAM and write 4-nt-upstream + 20-nt-spacer CNN inputs."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--pam",
        default="NGTT",
        type=str.upper,
        help=(
            "IUPAC PAM pattern to search for, e.g. NGG, NGTT, TTTV. "
            "Default: NGTT."
        ),
    )
    parser.add_argument("--guide-prefix", default="white_pamfree")
    return parser.parse_args()


def validate_pam_pattern(pattern):
    invalid = set(pattern).difference(IUPAC_CODES)
    if invalid:
        raise ValueError(
            f"Invalid IUPAC base(s) in PAM pattern {pattern!r}: "
            f"{sorted(invalid)}"
        )
    return pattern


def reverse_complement(sequence):
    return sequence.translate(COMPLEMENT)[::-1].upper()


def pam_matches(sequence, pattern):
    return len(sequence) == len(pattern) and all(
        base in IUPAC_CODES[code] for base, code in zip(sequence, pattern)
    )


def read_fasta(path):
    records = []
    header = None
    parts = []

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(parts)))
                header = line[1:].strip()
                parts = []
                if not header:
                    raise ValueError(f"{path}:{line_number}: empty FASTA header")
            else:
                if header is None:
                    raise ValueError(
                        f"{path}:{line_number}: sequence found before FASTA header"
                    )
                parts.append(line.upper())

    if header is not None:
        records.append((header, "".join(parts)))

    if not records:
        raise ValueError(f"No FASTA records found in {path}")

    seen_ids = set()
    checked = []
    for header, sequence in records:
        record_id = header.split()[0]
        if record_id in seen_ids:
            raise ValueError(f"Duplicate FASTA record ID: {record_id}")
        seen_ids.add(record_id)

        if not sequence:
            raise ValueError(f"Empty sequence for FASTA record {record_id}")

        invalid = set(sequence).difference(VALID_BASES)
        if invalid:
            raise ValueError(
                f"Unsupported base(s) in {record_id}: {sorted(invalid)}"
            )

        checked.append((header, record_id, sequence))

    return checked


def parse_header_coordinates(header):
    """Return chromosome and 1-based start from chr:start-end when present."""
    first_token = header.split()[0]
    match = re.match(r"^(.+):(\d+)-(\d+)$", first_token)
    if not match:
        return first_token, 1

    chrom = match.group(1)
    start = int(match.group(2))
    end = int(match.group(3))
    if start > end:
        raise ValueError(f"Invalid FASTA genomic range: {first_token}")
    return chrom, start


def local_to_genomic(local_start, local_end, region_start):
    return (
        region_start + local_start - 1,
        region_start + local_end - 1,
    )


def add_genomic_coordinates(row, region_start):
    for feature in ("context", "spacer", "pam"):
        genomic_start, genomic_end = local_to_genomic(
            row[f"local_{feature}_start"],
            row[f"local_{feature}_end"],
            region_start,
        )
        row[f"genomic_{feature}_start"] = genomic_start
        row[f"genomic_{feature}_end"] = genomic_end
    return row


def scan_forward(sequence, record_id, chrom, region_start, pam_pattern):
    """Scan 5'-[upstream4][spacer20][PAM]-3' on the reference strand."""
    rows = []
    pam_length = len(pam_pattern)
    target_length = SPACER_LENGTH + pam_length

    # spacer_start0 is the zero-based start of the 20-nt spacer.
    for spacer_start0 in range(
        UPSTREAM_LENGTH,
        len(sequence) - target_length + 1,
    ):
        context_start0 = spacer_start0 - UPSTREAM_LENGTH
        pam_start0 = spacer_start0 + SPACER_LENGTH
        pam_end0 = pam_start0 + pam_length

        upstream = sequence[context_start0:spacer_start0]
        spacer = sequence[spacer_start0:pam_start0]
        pam = sequence[pam_start0:pam_end0]

        complete_sequence = upstream + spacer + pam
        if not set(complete_sequence).issubset(UNAMBIGUOUS_BASES):
            continue
        if not pam_matches(pam, pam_pattern):
            continue

        row = {
            "record_id": record_id,
            "chrom": chrom,
            "strand": "+",
            "upstream4": upstream,
            "spacer": spacer,
            "pam": pam,
            "input_24nt": upstream + spacer,
            "target_with_pam": spacer + pam,
            "local_context_start": context_start0 + 1,
            "local_context_end": spacer_start0,
            "local_spacer_start": spacer_start0 + 1,
            "local_spacer_end": pam_start0,
            "local_pam_start": pam_start0 + 1,
            "local_pam_end": pam_end0,
        }
        rows.append(add_genomic_coordinates(row, region_start))

    return rows


def scan_reverse(sequence, record_id, chrom, region_start, pam_pattern):
    """Scan reverse-strand PAM targets and normalize to guide orientation.

    On the forward reference, the arrangement is:

        5'-[reverse-complement(PAM)]
           [reverse-complement(spacer20)]
           [reverse-complement(upstream4)]-3'

    Therefore, the four bases needed upstream of the reverse-strand spacer are
    physically to the right of the spacer on the forward reference.
    """
    rows = []
    pam_length = len(pam_pattern)
    target_length = pam_length + SPACER_LENGTH
    full_length = target_length + UPSTREAM_LENGTH

    # window_start0 is the zero-based start of the PAM on the forward
    # reference. The full reference window also includes spacer and context.
    for window_start0 in range(0, len(sequence) - full_length + 1):
        pam_ref_start0 = window_start0
        pam_ref_end0 = pam_ref_start0 + pam_length
        spacer_ref_start0 = pam_ref_end0
        spacer_ref_end0 = spacer_ref_start0 + SPACER_LENGTH
        context_ref_start0 = spacer_ref_end0
        context_ref_end0 = context_ref_start0 + UPSTREAM_LENGTH

        reference_window = sequence[window_start0:context_ref_end0]
        if not set(reference_window).issubset(UNAMBIGUOUS_BASES):
            continue

        oriented = reverse_complement(reference_window)
        upstream = oriented[:UPSTREAM_LENGTH]
        spacer = oriented[
            UPSTREAM_LENGTH:UPSTREAM_LENGTH + SPACER_LENGTH
        ]
        pam = oriented[UPSTREAM_LENGTH + SPACER_LENGTH:]

        if not pam_matches(pam, pam_pattern):
            continue

        row = {
            "record_id": record_id,
            "chrom": chrom,
            "strand": "-",
            "upstream4": upstream,
            "spacer": spacer,
            "pam": pam,
            "input_24nt": upstream + spacer,
            "target_with_pam": spacer + pam,
            # Coordinates remain ascending on the forward reference, even
            # though their biological order is reversed in guide orientation.
            "local_context_start": context_ref_start0 + 1,
            "local_context_end": context_ref_end0,
            "local_spacer_start": spacer_ref_start0 + 1,
            "local_spacer_end": spacer_ref_end0,
            "local_pam_start": pam_ref_start0 + 1,
            "local_pam_end": pam_ref_end0,
        }
        rows.append(add_genomic_coordinates(row, region_start))

    return rows


def validate_rows(rows, pam_pattern):
    for row in rows:
        if len(row["upstream4"]) != UPSTREAM_LENGTH:
            raise RuntimeError(f"Invalid upstream context: {row}")
        if len(row["spacer"]) != SPACER_LENGTH:
            raise RuntimeError(f"Invalid spacer: {row}")
        if not pam_matches(row["pam"], pam_pattern):
            raise RuntimeError(f"Invalid PAM: {row}")
        if len(row["input_24nt"]) != INPUT_LENGTH:
            raise RuntimeError(f"Invalid model input: {row}")
        if row["input_24nt"] != row["upstream4"] + row["spacer"]:
            raise RuntimeError(f"Model input assembly failed: {row}")
        if row["target_with_pam"] != row["spacer"] + row["pam"]:
            raise RuntimeError(f"Target assembly failed: {row}")


INPUT_LENGTH = UPSTREAM_LENGTH + SPACER_LENGTH


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    pam_pattern = validate_pam_pattern(args.pam)

    fasta_records = read_fasta(args.input)
    rows = []

    for header, record_id, sequence in fasta_records:
        chrom, region_start = parse_header_coordinates(header)
        rows.extend(
            scan_forward(sequence, record_id, chrom, region_start, pam_pattern)
        )
        rows.extend(
            scan_reverse(sequence, record_id, chrom, region_start, pam_pattern)
        )

    rows.sort(
        key=lambda row: (
            row["chrom"],
            row["genomic_spacer_start"],
            row["strand"],
        )
    )

    id_width = max(4, len(str(len(rows))))
    for index, row in enumerate(rows, start=1):
        row["guide_id"] = f"{args.guide_prefix}_{index:0{id_width}d}"

    validate_rows(rows, pam_pattern)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    forward_count = sum(row["strand"] == "+" for row in rows)
    reverse_count = sum(row["strand"] == "-" for row in rows)
    pam_counts = {}
    for row in rows:
        pam_counts[row["pam"]] = pam_counts.get(row["pam"], 0) + 1

    print(f"Input FASTA: {args.input}")
    print(f"FASTA records: {len(fasta_records)}")
    print(f"PAM pattern: {pam_pattern}")
    print(f"Model input: upstream4 + spacer20 ({INPUT_LENGTH} nt)")
    print(f"Total candidates: {len(rows)}")
    print(f"Forward strand: {forward_count}")
    print(f"Reverse strand: {reverse_count}")
    print("Observed PAMs:")
    for pam, count in sorted(pam_counts.items()):
        print(f"  {pam}: {count}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
