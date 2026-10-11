#!/usr/bin/env python3

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from pam_utils import (
    normalize_pam_pattern,
    pam_length,
)


SPACER_LEN = 20


def load_query_map(path):
    """
    query_sequence -> guide metadata
    """

    mapping = {}

    with open(path, newline="") as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            q = (
                row["query_sequence"]
                .upper()
            )

            if q in mapping:

                raise ValueError(
                    f"Duplicate query_sequence: {q}"
                )

            mapping[q] = row

    return mapping



def parse_cas_line(line):
    """
    Parse Cas-OFFinder mismatch-only output.

    Format:

    query
    reference_name(position may contain spaces)
    position
    candidate_with_pam
    strand
    mismatch

    """

    parts = (
        line
        .rstrip("\n")
        .split()
    )

    if len(parts) < 6:

        raise ValueError(
            f"Unexpected Cas-OFFinder line:\n{line}"
        )


    query = parts[0]

    mismatch = int(
        parts[-1]
    )

    strand = parts[-2]

    candidate = parts[-3]

    position = int(
        parts[-4]
    )

    reference_name = " ".join(
        parts[1:-4]
    )


    return {

        "query_sequence":
            query.upper(),

        "reference_name":
            reference_name,

        "cas_position_0based":
            position,

        "candidate_with_pam_raw":
            candidate,

        "strand":
            strand,

        "cas_mismatch_count":
            mismatch,
    }



def normalize_candidate(
    seq,
    expected_length,
):

    seq = (
        seq
        .upper()
        .strip()
    )

    if len(seq) != expected_length:

        raise ValueError(
            f"Candidate length "
            f"{len(seq)} != "
            f"{expected_length}: "
            f"{seq}"
        )


    invalid = (
        set(seq)
        -
        set("ACGT")
    )

    if invalid:

        raise ValueError(
            f"Invalid bases {invalid}: {seq}"
        )

    return seq



def hamming(a, b):

    if len(a) != len(b):

        raise ValueError(
            "Hamming length mismatch"
        )

    return sum(
        x != y
        for x, y in zip(a, b)
    )



def main():

    parser = argparse.ArgumentParser(
        description=(
            "Convert Cas-OFFinder raw output "
            "into PAM-agnostic candidate table."
        )
    )


    parser.add_argument(
        "--raw",
        required=True,
        help="Raw Cas-OFFinder output"
    )


    parser.add_argument(
        "--query-map",
        required=True,
        help="Cas-OFFinder query map"
    )


    parser.add_argument(
        "--pam",
        required=True,
        help=(
            "IUPAC PAM pattern, "
            "e.g. NGG or NGTT"
        )
    )


    parser.add_argument(
        "--output",
        required=True,
        help="Output candidate CSV"
    )


    args = parser.parse_args()


    pam_pattern = normalize_pam_pattern(
        args.pam
    )


    pam_len = pam_length(
        pam_pattern
    )


    expected_length = (
        SPACER_LEN
        +
        pam_len
    )


    print("=" * 70)
    print("Candidate preparation")
    print("=" * 70)

    print(
        f"PAM pattern:       {pam_pattern}"
    )

    print(
        f"PAM length:        {pam_len}"
    )

    print(
        f"Candidate length:  {expected_length}"
    )



    query_map = load_query_map(
        args.query_map
    )


    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )



    fieldnames = [

        "candidate_id",

        "guide_id",

        "sgRNA_20nt",

        "off_target_20nt",

        "candidate_pam",

        "candidate_with_pam",


        "reference_name",

        "cas_position_0based",

        "genomic_start_1based",

        "genomic_end_1based",


        "off_target_strand",


        "mismatch_count_20nt",

        "cas_mismatch_count",


        "query_sequence",


        "target_chrom",

        "target_strand",

        "target_spacer_start",

        "target_spacer_end",

        "target_pam_start",

        "target_pam_end",

    ]



    total = 0
    missing_query = 0
    invalid_length = 0
    mismatch_difference = 0


    mismatch_distribution = {}



    with open(args.raw) as fin, open(
        output_path,
        "w",
        newline=""
    ) as fout:


        writer = csv.DictWriter(
            fout,
            fieldnames=fieldnames
        )

        writer.writeheader()



        for line in fin:


            if not line.strip():
                continue


            total += 1


            parsed = parse_cas_line(
                line
            )


            query = parsed[
                "query_sequence"
            ]


            if query not in query_map:

                missing_query += 1

                continue



            meta = query_map[query]



            try:

                candidate = normalize_candidate(
                    parsed[
                        "candidate_with_pam_raw"
                    ],
                    expected_length
                )


            except ValueError:

                invalid_length += 1

                continue



            spacer20 = meta[
                "spacer"
            ].upper()



            off20 = candidate[
                :SPACER_LEN
            ]


            candidate_pam = candidate[
                SPACER_LEN:
            ]



            mm20 = hamming(
                spacer20,
                off20
            )



            cas_mm = parsed[
                "cas_mismatch_count"
            ]



            if mm20 != cas_mm:

                mismatch_difference += 1



            mismatch_distribution[mm20] = (
                mismatch_distribution.get(
                    mm20,
                    0
                )
                + 1
            )



            start1 = (
                parsed[
                    "cas_position_0based"
                ]
                + 1
            )


            end1 = (
                start1
                +
                expected_length
                -
                1
            )



            writer.writerow({

                "candidate_id":
                    f"candidate_{total:09d}",


                "guide_id":
                    meta["guide_id"],


                "sgRNA_20nt":
                    spacer20,


                "off_target_20nt":
                    off20,


                "candidate_pam":
                    candidate_pam,


                "candidate_with_pam":
                    candidate,



                "reference_name":
                    parsed[
                        "reference_name"
                    ],


                "cas_position_0based":
                    parsed[
                        "cas_position_0based"
                    ],


                "genomic_start_1based":
                    start1,


                "genomic_end_1based":
                    end1,


                "off_target_strand":
                    parsed["strand"],



                "mismatch_count_20nt":
                    mm20,


                "cas_mismatch_count":
                    cas_mm,


                "query_sequence":
                    query,



                "target_chrom":
                    meta.get(
                        "chrom",
                        ""
                    ),


                "target_strand":
                    meta.get(
                        "strand",
                        ""
                    ),


                "target_spacer_start":
                    meta.get(
                        "genomic_spacer_start",
                        ""
                    ),


                "target_spacer_end":
                    meta.get(
                        "genomic_spacer_end",
                        ""
                    ),


                "target_pam_start":
                    meta.get(
                        "genomic_pam_start",
                        ""
                    ),


                "target_pam_end":
                    meta.get(
                        "genomic_pam_end",
                        ""
                    ),

            })



    print()
    print("=" * 70)
    print(
        "Candidate preparation complete"
    )
    print("=" * 70)


    print(
        f"Raw rows:                 {total:,}"
    )

    print(
        f"Missing query mapping:    {missing_query:,}"
    )

    print(
        f"Invalid length:           {invalid_length:,}"
    )

    print(
        f"Mismatch difference:      {mismatch_difference:,}"
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
        f"Output: {output_path}"
    )



if __name__ == "__main__":
    main()