#!/usr/bin/env python3
"""Generate candidate guides from one or more genomic FASTA records.

Each FASTA record is scanned independently on both strands. Windows are never
allowed to cross FASTA-record (for example, exon) boundaries.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pam_utils import normalize_pam_pattern, pam_length, pam_matches, reverse_complement


SPACER_LEN = 20
OUTPUT_COLUMNS = [
    "guide_id", "chrom", "spacer", "pam", "target_with_pam", "strand",
    "local_spacer_start", "local_spacer_end",
    "genomic_spacer_start", "genomic_spacer_end",
    "local_pam_start", "local_pam_end",
    "genomic_pam_start", "genomic_pam_end",
]


def read_fasta_records(path):
    """Return all FASTA records as ``[(header, sequence), ...]``."""
    records = []
    header = None
    seq_parts = []

    def finish_record():
        if header is None:
            return
        sequence = "".join(seq_parts).upper()
        if not sequence:
            raise ValueError(f"FASTA record is empty: {header}")
        invalid = set(sequence) - {"A", "C", "G", "T", "N"}
        if invalid:
            raise ValueError(
                f"Unsupported bases in FASTA record {header!r}: {sorted(invalid)}"
            )
        records.append((header, sequence))

    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish_record()
                header = line[1:].strip()
                seq_parts = []
                if not header:
                    raise ValueError(f"Empty FASTA header at line {line_number}")
            else:
                if header is None:
                    raise ValueError(
                        f"FASTA sequence occurs before the first header at line {line_number}"
                    )
                seq_parts.append(line)
    finish_record()

    if not records:
        raise ValueError("No FASTA records found.")
    return records


def parse_fasta_header(header, sequence_length):
    """Recover chromosome and 1-based start from ``chrom:start-end`` headers."""
    first_token = header.split()[0]
    match = re.match(r"^(.+):(\d+)-(\d+)$", first_token)
    if not match:
        return first_token, 1

    chrom = match.group(1)
    start = int(match.group(2))
    end = int(match.group(3))
    if start > end:
        raise ValueError(f"Invalid FASTA genomic range: {first_token}")
    expected_length = end - start + 1
    if expected_length != sequence_length:
        raise ValueError(
            f"FASTA range/sequence length mismatch for {first_token}: "
            f"range={expected_length}, sequence={sequence_length}"
        )
    return chrom, start


def local_to_genomic(local_start, local_end, region_start):
    return region_start + local_start - 1, region_start + local_end - 1


def scan_forward(sequence, pam_pattern, chrom, region_start):
    pam_len = pam_length(pam_pattern)
    total_len = SPACER_LEN + pam_len
    rows = []
    for start0 in range(0, len(sequence) - total_len + 1):
        spacer = sequence[start0:start0 + SPACER_LEN]
        pam = sequence[start0 + SPACER_LEN:start0 + total_len]
        if "N" in spacer or "N" in pam or not pam_matches(pam, pam_pattern):
            continue

        local_spacer_start = start0 + 1
        local_spacer_end = start0 + SPACER_LEN
        local_pam_start = start0 + SPACER_LEN + 1
        local_pam_end = start0 + total_len
        genomic_spacer_start, genomic_spacer_end = local_to_genomic(
            local_spacer_start, local_spacer_end, region_start
        )
        genomic_pam_start, genomic_pam_end = local_to_genomic(
            local_pam_start, local_pam_end, region_start
        )
        rows.append({
            "chrom": chrom, "spacer": spacer, "pam": pam,
            "target_with_pam": spacer + pam, "strand": "+",
            "local_spacer_start": local_spacer_start,
            "local_spacer_end": local_spacer_end,
            "genomic_spacer_start": genomic_spacer_start,
            "genomic_spacer_end": genomic_spacer_end,
            "local_pam_start": local_pam_start,
            "local_pam_end": local_pam_end,
            "genomic_pam_start": genomic_pam_start,
            "genomic_pam_end": genomic_pam_end,
        })
    return rows


def scan_reverse(sequence, pam_pattern, chrom, region_start):
    pam_len = pam_length(pam_pattern)
    total_len = SPACER_LEN + pam_len
    rows = []
    for start0 in range(0, len(sequence) - total_len + 1):
        genomic_window = sequence[start0:start0 + total_len]
        if "N" in genomic_window:
            continue
        oriented = reverse_complement(genomic_window)
        spacer = oriented[:SPACER_LEN]
        pam = oriented[SPACER_LEN:]
        if not pam_matches(pam, pam_pattern):
            continue

        local_pam_start = start0 + 1
        local_pam_end = start0 + pam_len
        local_spacer_start = start0 + pam_len + 1
        local_spacer_end = start0 + total_len
        genomic_spacer_start, genomic_spacer_end = local_to_genomic(
            local_spacer_start, local_spacer_end, region_start
        )
        genomic_pam_start, genomic_pam_end = local_to_genomic(
            local_pam_start, local_pam_end, region_start
        )
        rows.append({
            "chrom": chrom, "spacer": spacer, "pam": pam,
            "target_with_pam": spacer + pam, "strand": "-",
            "local_spacer_start": local_spacer_start,
            "local_spacer_end": local_spacer_end,
            "genomic_spacer_start": genomic_spacer_start,
            "genomic_spacer_end": genomic_spacer_end,
            "local_pam_start": local_pam_start,
            "local_pam_end": local_pam_end,
            "genomic_pam_start": genomic_pam_start,
            "genomic_pam_end": genomic_pam_end,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate 20nt candidate gRNAs from a single- or multi-record "
            "target FASTA using an arbitrary IUPAC PAM pattern."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Target-region FASTA.")
    parser.add_argument("--pam", required=True, help="IUPAC PAM, e.g. NGG or NGTT.")
    parser.add_argument("--output", required=True, type=Path, help="Output guide CSV.")
    parser.add_argument("--guide-prefix", default="gRNA")
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    pam_pattern = normalize_pam_pattern(args.pam)
    records = read_fasta_records(args.input)

    print("=" * 70)
    print("Candidate gRNA generation")
    print("=" * 70)
    print(f"Input FASTA:          {args.input}")
    print(f"FASTA records:        {len(records):,}")
    print(f"Spacer length:        {SPACER_LEN}")
    print(f"PAM pattern:          {pam_pattern}")
    print(f"Target length:        {SPACER_LEN + len(pam_pattern)}")

    rows = []
    record_summaries = []
    for record_index, (header, sequence) in enumerate(records, 1):
        chrom, region_start = parse_fasta_header(header, len(sequence))
        forward = scan_forward(sequence, pam_pattern, chrom, region_start)
        reverse = scan_reverse(sequence, pam_pattern, chrom, region_start)
        rows.extend(forward)
        rows.extend(reverse)
        record_summaries.append((record_index, header, len(sequence), len(forward), len(reverse)))

    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(
            ["chrom", "genomic_spacer_start", "strand", "genomic_pam_start"],
            ascending=[True, True, True, True], kind="mergesort",
        ).reset_index(drop=True)

        # The same genomic target must not be emitted twice by overlapping records.
        target_key = [
            "chrom", "strand", "genomic_spacer_start", "genomic_spacer_end",
            "genomic_pam_start", "genomic_pam_end",
        ]
        duplicated = df.duplicated(target_key, keep=False)
        if duplicated.any():
            examples = df.loc[duplicated, target_key].head(10).to_dict("records")
            raise RuntimeError(
                "Overlapping FASTA records produced duplicate genomic guides; "
                f"examples: {examples}"
            )

        width = max(4, len(str(len(df))))
        df.insert(0, "guide_id", [
            f"{args.guide_prefix}_{index:0{width}d}"
            for index in range(1, len(df) + 1)
        ])
        df = df[OUTPUT_COLUMNS]

        if not df["spacer"].str.len().eq(SPACER_LEN).all():
            raise RuntimeError("Spacer length QC failed.")
        if not df["pam"].str.len().eq(len(pam_pattern)).all():
            raise RuntimeError("PAM length QC failed.")
        if not all(pam_matches(pam, pam_pattern) for pam in df["pam"]):
            raise RuntimeError("PAM matching QC failed.")
        if not df["target_with_pam"].str.len().eq(SPACER_LEN + len(pam_pattern)).all():
            raise RuntimeError("target_with_pam length QC failed.")
    else:
        df = pd.DataFrame(columns=OUTPUT_COLUMNS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print()
    print("Per-record summary (records are scanned independently):")
    for index, header, length, forward_count, reverse_count in record_summaries:
        print(
            f"  {index:>3}: length={length:>6,}, +={forward_count:>4,}, "
            f"-={reverse_count:>4,}  {header}"
        )
    print()
    print("=" * 70)
    print("Guide generation complete")
    print("=" * 70)
    print(f"Total guides:          {len(df):,}")
    if len(df):
        print(f"Forward strand:        {(df['strand'] == '+').sum():,}")
        print(f"Reverse strand:        {(df['strand'] == '-').sum():,}")
        print("Observed PAMs:")
        print(df["pam"].value_counts().head(20).to_string())
    else:
        print("[WARN] No candidate guides found.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
