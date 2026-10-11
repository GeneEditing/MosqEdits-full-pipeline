#!/usr/bin/env python3
"""Unified on-target prediction pipeline for arbitrary-PAM and NGG guides.

NGG:
    official CRISPRon.sh
        -> make_ngg_top20.py
        -> ranked Top-N table

Other PAM (PAM-free model):
    extract_pamfree_24nt.py
        -> PAM-free five-model ensemble
        -> ranked Top-N table
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

PAM_AGNOSTIC_DIR = PROJECT_DIR / "pam_agnostic"

DEFAULT_OFFICIAL_CRISPRON = (
    PROJECT_DIR / "crispron_official"
)

DEFAULT_MODELS = [
    PAM_AGNOSTIC_DIR / "models" / f"fold_{fold}" / "best_model.h5"
    for fold in range(1, 6)
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run PAM-free prediction for a non-NGG PAM, or official NGG "
            "CRISPRon prediction, and output a ranked Top-N guide table."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input target-region FASTA file.",
    )
    parser.add_argument(
        "--pam",
        required=True,
        type=str.upper,
        help=(
            "PAM type. Use 'NGG' to run the official CRISPRon model. Any "
            "other IUPAC PAM pattern (e.g. NGTT, TTTV) runs the PAM-free "
            "model, which does not depend on the PAM identity."
        ),
    )
    parser.add_argument(
        "--name",
        default=None,
        help=(
            "Output name prefix. By default, the input FASTA filename "
            "without its extension is used."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top guides to output. Default: 20.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Prediction batch size for the PAM-free model. Default: 500.",
    )
    parser.add_argument(
        "--gpu",
        default="",
        help=(
            "GPU number assigned through CUDA_VISIBLE_DEVICES. "
            "Use an empty string for CPU. Default: '' (CPU)."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_DIR / "results",
        help="Root directory for prediction results.",
    )
    parser.add_argument(
        "--official-crispron",
        type=Path,
        default=DEFAULT_OFFICIAL_CRISPRON,
        help="Path to the official CRISPRon repository.",
    )
    parser.add_argument(
        "--models",
        nargs=5,
        type=Path,
        default=DEFAULT_MODELS,
        metavar="MODEL",
        help="Five PAM-free fold models for prediction.",
    )

    return parser.parse_args()


def sanitize_name(name):
    """Convert a name into a safe filename/directory component."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = name.strip("._-")

    if not name:
        raise ValueError("The resolved output name is empty")

    return name


def run_command(command, environment=None, cwd=None):
    printable = " ".join(str(value) for value in command)

    print()
    print("=" * 72)
    print(f"Running: {printable}")
    if cwd is not None:
        print(f"Working directory: {cwd}")
    print("=" * 72)
    print()

    subprocess.run(
        [str(value) for value in command],
        check=True,
        env=environment,
        cwd=str(cwd) if cwd is not None else None,
    )


def validate_common_inputs(args):
    args.input = args.input.expanduser().resolve()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input FASTA does not exist: {args.input}")

    if args.top < 1:
        raise ValueError("--top must be at least 1")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    output_name = args.name if args.name else args.input.stem
    return sanitize_name(output_name)


def build_environment(gpu):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return environment


def build_result_name(output_name, pam):
    suffix = f"_{pam.lower()}"
    if output_name.lower().endswith(suffix):
        return output_name
    return f"{output_name}{suffix}"


def run_pamfree(args, output_name, environment):
    extractor = PAM_AGNOSTIC_DIR / "bin" / "extract_pamfree_24nt.py"
    predictor = PAM_AGNOSTIC_DIR / "bin" / "predict_white_top20.py"

    if not extractor.is_file():
        raise FileNotFoundError(f"Missing PAM-free extractor: {extractor}")

    if not predictor.is_file():
        raise FileNotFoundError(f"Missing PAM-free predictor: {predictor}")

    models = [model.expanduser().resolve() for model in args.models]

    for model in models:
        if not model.is_file():
            raise FileNotFoundError(f"Missing PAM-free model: {model}")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    result_name = build_result_name(output_name, args.pam.lower())
    top_file = output_root / f"{result_name}_top{args.top}.tsv"
    guide_prefix = sanitize_name(result_name)

    with tempfile.TemporaryDirectory(
        prefix=f"{result_name}_"
    ) as temporary_dir:
        candidate_file = Path(temporary_dir) / "candidates.tsv"

        run_command(
            [
                sys.executable,
                "-u",
                extractor,
                "--input",
                args.input,
                "--pam",
                args.pam,
                "--output",
                candidate_file,
                "--guide-prefix",
                guide_prefix,
            ],
            environment=environment,
            cwd=PROJECT_DIR,
        )

        run_command(
            [
                sys.executable,
                "-u",
                predictor,
                "--input",
                candidate_file,
                "--pam",
                args.pam,
                "--models",
                *models,
                "--top",
                args.top,
                "--batch-size",
                args.batch_size,
                "--output",
                top_file,
            ],
            environment=environment,
            cwd=PROJECT_DIR,
        )

    return {
        "pam": args.pam,
        "output_dir": output_root,
        "top_file": top_file,
    }


def validate_official_crispron(official_root):
    required_paths = [
        official_root / "bin" / "CRISPRon.sh",
        official_root / "bin" / "get_30mers_from_fa.py",
        official_root / "bin" / "CRISPRspec_CRISPRoff_pipeline.py",
        official_root / "bin" / "DeepCRISPRon_eval.py",
        official_root / "data" / "model" / "energy_dics.pkl",
    ]

    for fold in range(1, 7):
        required_paths.append(
            official_root
            / "data"
            / "deep_models"
            / "best"
            / f"{fold}.model.best"
            / "saved_model.pb"
        )

    missing = [path for path in required_paths if not path.exists()]

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Official CRISPRon files are missing:\n" + formatted
        )


def run_ngg(args, output_name, environment):
    official_root = args.official_crispron.expanduser().resolve()
    validate_official_crispron(official_root)

    ranking_script = SCRIPT_DIR / "make_ngg_top20.py"

    if not ranking_script.is_file():
        raise FileNotFoundError(
            f"Missing NGG ranking script: {ranking_script}"
        )

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    result_name = build_result_name(output_name, "ngg")
    top_file = output_root / f"{result_name}_top{args.top}.tsv"

    with tempfile.TemporaryDirectory(
        prefix=f"{result_name}_"
    ) as temporary_dir:
        temporary_path = Path(temporary_dir)
        crispron_csv = temporary_path / "crispron.csv"

        run_command(
            [
                "bash",
                official_root / "bin" / "CRISPRon.sh",
                args.input,
                temporary_path,
            ],
            environment=environment,
            cwd=official_root,
        )

        if not crispron_csv.is_file():
            raise FileNotFoundError(
                "Official CRISPRon completed without producing "
                f"the expected file: {crispron_csv}"
            )

        run_command(
            [
                sys.executable,
                "-u",
                ranking_script,
                "--crispron-csv",
                crispron_csv,
                "--fasta",
                args.input,
                "--top",
                args.top,
                "--output",
                top_file,
            ],
            environment=environment,
            cwd=PROJECT_DIR,
        )

    return {
        "pam": "NGG",
        "output_dir": output_root,
        "top_file": top_file,
    }


def main():
    args = parse_args()
    output_name = validate_common_inputs(args)
    environment = build_environment(args.gpu)

    print(f"Input FASTA: {args.input}")
    print(f"PAM: {args.pam.upper()}")
    print(f"Output name: {output_name}")
    print(f"Top guides: {args.top}")
    print(f"CUDA_VISIBLE_DEVICES: {args.gpu!r}")

    if args.pam == "NGG":
        result = run_ngg(
            args=args,
            output_name=output_name,
            environment=environment,
        )
    else:
        result = run_pamfree(
            args=args,
            output_name=output_name,
            environment=environment,
        )

    print()
    print("=" * 72)
    print("Prediction completed successfully")
    print(f"PAM: {result['pam']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Top-{args.top} result: {result['top_file']}")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(
            f"ERROR: pipeline command failed with exit code "
            f"{exc.returncode}",
            file=sys.stderr,
        )
        sys.exit(exc.returncode)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)