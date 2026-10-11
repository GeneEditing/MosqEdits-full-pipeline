#!/usr/bin/env python3
"""Run on-target activity ranking followed by the off-target risk pipeline."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "examples" / "white_AalbF3_region.fa"
DEFAULT_GENOME = (
    ROOT / "off_target" / "reference" / "AalbF3"
    / "GCA_018104305.1_AalbF3_genomic.fna"
)
DEFAULT_SNPS = (
    ROOT / "off_target" / "population_variants"
    / "AalbF3_population_snps.tsv"
)
CAS_OFFINDER = ROOT / "off_target" / "tools" / "cas-offinder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict on-target activity, search genome-wide off-targets, "
            "and produce a combined guide-risk table."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Target-region FASTA.")
    parser.add_argument("--gene", default="white_demo", help="Safe run label for output files.")
    parser.add_argument("--pam", required=True, help="IUPAC PAM, e.g. NGG or NGTT.")
    parser.add_argument("--genome", type=Path, default=DEFAULT_GENOME, help="Matching whole-genome FASTA.")
    parser.add_argument("--snps", type=Path, default=DEFAULT_SNPS, help="Population SNP TSV for the same assembly.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results", help="Directory for this run's outputs.")
    parser.add_argument("--max-mismatches", type=int, default=6)
    parser.add_argument("--gpu", default="", help="Optional GPU index for on-target/Cas-OFFinder use.")
    parser.add_argument("--score-device", default="cpu", help="PyTorch scoring device: cpu, cuda, or a physical GPU ID.")
    parser.add_argument("--keep-work-dir", action="store_true", help="Keep off-target intermediate files for debugging.")
    parser.add_argument("--on-target-env", default="crispron", help="Conda environment for on-target prediction.")
    parser.add_argument("--off-target-env", default="mosqguide-off", help="Conda environment for off-target analysis.")
    return parser.parse_args()


def conda_run_prefix(environment: str) -> list[str]:
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        raise RuntimeError("Could not find conda; activate Conda or add it to PATH.")
    return [conda, "run", "--no-capture-output", "--name", environment, "python"]


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.gene):
        raise SystemExit("--gene may contain only letters, digits, dots, underscores, and hyphens.")
    args.gene = args.gene.strip("._-")
    if not args.gene:
        raise SystemExit("--gene must contain at least one letter or digit.")
    if args.max_mismatches < 0:
        raise SystemExit("--max-mismatches must be non-negative.")

    input_fasta = args.input.expanduser().resolve()
    genome = args.genome.expanduser().resolve()
    snps = args.snps.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() / args.gene
    for label, path in (("input FASTA", input_fasta), ("genome FASTA", genome), ("SNP table", snps)):
        if not path.is_file():
            raise SystemExit(f"Missing {label}: {path}")

    pam = args.pam.strip().upper()
    on_target_dir = output_dir / "on_target"
    suffix = f"_{pam.lower()}"
    result_name = args.gene if args.gene.lower().endswith(suffix) else f"{args.gene}{suffix}"
    on_target_result = on_target_dir / f"{result_name}_top20.tsv"
    on_target_script = ROOT / "on_target" / "bin" / "run_ontarget.py"
    off_target_script = ROOT / "off_target" / "run_pipeline.py"

    on_target_command = [
        *conda_run_prefix(args.on_target_env), str(on_target_script),
        "--input", str(input_fasta), "--pam", pam,
        "--name", args.gene, "--top", "20",
        "--output-root", str(on_target_dir), "--gpu", args.gpu,
    ]
    print("\n[1/2] Predicting and ranking on-target activity", flush=True)
    subprocess.run(on_target_command, cwd=ROOT, check=True)
    if not on_target_result.is_file():
        raise FileNotFoundError(f"On-target output was not created: {on_target_result}")

    off_target_command = [
        *conda_run_prefix(args.off_target_env), str(off_target_script),
        "--gene", args.gene,
        "--input", str(input_fasta),
        "--genome", str(genome),
        "--snps", str(snps),
        "--pam", pam,
        "--on-target-top20", str(on_target_result),
        "--cas-offinder", str(CAS_OFFINDER),
        "--score-device", args.score_device,
        "--gpu", args.gpu,
        "--max-mismatches", str(args.max_mismatches),
    ]
    if args.keep_work_dir:
        off_target_command.append("--keep-work-dir")

    print("\n[2/2] Searching genome-wide off-targets and aggregating risk", flush=True)
    subprocess.run(off_target_command, cwd=ROOT, check=True)

    generated_result = (
        ROOT / "off_target" / "results" / args.gene
        / f"top20_{pam.lower()}_gRNA.tsv"
    )
    if not generated_result.is_file():
        raise FileNotFoundError(f"Combined result was not created: {generated_result}")
    output_dir.mkdir(parents=True, exist_ok=True)
    final_result = output_dir / "final_guides_with_risk.tsv"
    shutil.copy2(generated_result, final_result)

    print("\nCompleted successfully.")
    print(f"On-target activity ranking: {on_target_result}")
    print(f"Combined activity and off-target risk: {final_result}")


if __name__ == "__main__":
    main()