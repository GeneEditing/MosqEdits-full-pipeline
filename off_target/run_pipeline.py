#!/usr/bin/env python3
"""Run the unified NGG/PAM-agnostic off-target pipeline (steps 01-10).

Every step is executed with an argument list, followed by a strict QC gate.
Step 02 has explicit NVIDIA OpenCL setup and a controlled GPU-device retry.
Intermediate step outputs live in a scratch work directory outside results/;
the only artifact left under results/ is the final annotated Top20 TSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "pam_agnostic").is_dir():
    OFFTARGET_ROOT = SCRIPT_DIR
    PAM_AGNOSTIC_ROOT = OFFTARGET_ROOT / "pam_agnostic"
    PIPELINE_DIR = OFFTARGET_ROOT / "pipeline"
else:
    PAM_AGNOSTIC_ROOT = SCRIPT_DIR
    OFFTARGET_ROOT = PAM_AGNOSTIC_ROOT.parent
    PIPELINE_DIR = PAM_AGNOSTIC_ROOT / "pipeline"
DNA20 = re.compile(r"^[ACGT]{20}$")
IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "R": "[AG]", "Y": "[CT]", "S": "[GC]", "W": "[AT]",
    "K": "[GT]", "M": "[AC]", "B": "[CGT]", "D": "[AGT]",
    "H": "[ACT]", "V": "[ACG]", "N": "[ACGT]",
}

def first_existing_path(*paths: Path) -> Path:
    return next((path for path in paths if path.exists()), paths[0])


NGG_ROOT = first_existing_path(
    OFFTARGET_ROOT / "pam_ngg",
    OFFTARGET_ROOT / "off_target_model_NGG",
)
MODEL_PROFILES = {
    "ngg": {
        "dl_checkpoint": NGG_ROOT / "checkpoints" / "offtarget_focal_best.pt",
        "rf_model": NGG_ROOT / "checkpoints" / "rf_handcrafted_baseline.joblib",
        "dl_calibrator": NGG_ROOT / "checkpoints" / "dl_probability_calibrator.joblib",
        "rf_calibrator": NGG_ROOT / "checkpoints" / "rf_probability_calibrator.joblib",
        "ensemble_weight": 0.86,
    },
    "pam-agnostic": {
        "dl_checkpoint": PAM_AGNOSTIC_ROOT / "checkpoints" / "offtarget_pam_agnostic_focal_best.pt",
        "rf_model": PAM_AGNOSTIC_ROOT / "checkpoints" / "rf_handcrafted_baseline.joblib",
        "dl_calibrator": PAM_AGNOSTIC_ROOT / "checkpoints" / "dl_probability_calibrator.joblib",
        "rf_calibrator": PAM_AGNOSTIC_ROOT / "checkpoints" / "rf_probability_calibrator.joblib",
        "ensemble_weight": 0.84,
    },
}


class QCError(RuntimeError):
    pass


def absolute(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    cwd_path = (Path.cwd() / path).resolve()
    project_path = (OFFTARGET_ROOT / path).resolve()
    if cwd_path.exists():
        return cwd_path
    if project_path.exists():
        return project_path
    if Path.cwd().resolve() == OFFTARGET_ROOT.resolve():
        return cwd_path
    return project_path


def read_csv(path: Path, *, allow_empty: bool = False):
    if not path.is_file():
        raise QCError(f"Missing output: {path}")
    if path.stat().st_size == 0:
        if allow_empty:
            return [], []
        raise QCError(f"Empty output: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not fields:
        raise QCError(f"No CSV header: {path}")
    if not rows and not allow_empty:
        raise QCError(f"No data rows: {path}")
    return fields, rows


def require_columns(fields: Iterable[str], required: Iterable[str], label: str):
    missing = sorted(set(required) - set(fields))
    if missing:
        raise QCError(f"{label}: missing columns {missing}")


def first_column(fields: Iterable[str], choices: Iterable[str]):
    available = set(fields)
    return next((name for name in choices if name in available), None)


def numeric(values: Iterable[str], label: str, *, minimum=None, maximum=None):
    result = []
    for index, value in enumerate(values, 1):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise QCError(f"{label}: non-numeric value at row {index}: {value!r}") from exc
        if number != number or number in (float("inf"), float("-inf")):
            raise QCError(f"{label}: non-finite value at row {index}")
        if minimum is not None and number < minimum:
            raise QCError(f"{label}: value below {minimum} at row {index}: {number}")
        if maximum is not None and number > maximum:
            raise QCError(f"{label}: value above {maximum} at row {index}: {number}")
        result.append(number)
    return result


def pam_regex(pattern: str):
    try:
        return re.compile("^" + "".join(IUPAC[base] for base in pattern.upper()) + "$")
    except KeyError as exc:
        raise QCError(f"Unsupported IUPAC PAM symbol: {exc.args[0]}") from exc


def qc_guides(path: Path, pam_pattern: str):
    fields, rows = read_csv(path)
    require_columns(fields, ["guide_id", "spacer", "pam"], "Step 01")
    ids = [row["guide_id"].strip() for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise QCError("Step 01: guide_id values must be non-empty and unique")
    pam_match = pam_regex(pam_pattern)
    for index, row in enumerate(rows, 1):
        spacer = row["spacer"].strip().upper()
        pam = row["pam"].strip().upper()
        if not DNA20.fullmatch(spacer):
            raise QCError(f"Step 01: invalid 20nt spacer at row {index}: {spacer!r}")
        if not pam_match.fullmatch(pam):
            raise QCError(f"Step 01: PAM does not match {pam_pattern} at row {index}: {pam!r}")


def qc_raw(path: Path):
    if not path.is_file() or path.stat().st_size == 0:
        raise QCError(f"Step 02: Cas-OFFinder output is missing or empty: {path}")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first = next((line for line in handle if line.strip()), "")
    if len(first.rstrip("\n").split("\t")) < 6:
        raise QCError("Step 02: first non-empty Cas-OFFinder row has fewer than 6 fields")


def qc_candidates(path: Path, label: str, max_mismatch: int, *, allow_empty=False):
    fields, rows = read_csv(path, allow_empty=allow_empty)
    if not rows and allow_empty:
        return
    require_columns(fields, ["guide_id"], label)
    mismatch = first_column(fields, ["mismatch", "mismatches", "mismatch_count"])
    if mismatch:
        numeric((row[mismatch] for row in rows), f"{label} mismatch", minimum=0, maximum=max_mismatch)


def qc_population(path: Path, max_mismatch: int):
    qc_candidates(path, "Step 04", max_mismatch)
    fields, rows = read_csv(path)
    af = first_column(fields, ["alt_allele_frequency", "allele_frequency", "af"])
    if af:
        values = [row[af] for row in rows if row[af].strip()]
        numeric(values, "Step 04 allele frequency", minimum=0, maximum=1)


def qc_merged(path: Path, max_mismatch: int):
    qc_candidates(path, "Step 06", max_mismatch)
    fields, rows = read_csv(path)
    if "source" in fields:
        allowed = {"reference", "population_snp", "snp_created_pam"}
        bad = sorted({row["source"] for row in rows} - allowed)
        if bad:
            raise QCError(f"Step 06: unexpected source values: {bad}")


def candidate_keys(fields, rows):
    preferred = [
        "candidate_id", "guide_id", "fasta_id", "chrom", "strand",
        "protospacer_start_1based", "protospacer_end_1based",
        "pam_start_1based", "pam_end_1based", "candidate_sequence",
    ]
    keys = [name for name in preferred if name in fields]
    if "candidate_id" in keys:
        keys = ["candidate_id"]
    return {tuple(row.get(name, "") for name in keys) for row in rows} if keys else set()


def qc_exclusion(output: Path, excluded: Path, guides: Path, max_mismatch: int):
    qc_candidates(output, "Step 07", max_mismatch, allow_empty=True)
    out_fields, out_rows = read_csv(output, allow_empty=True)
    ex_fields, ex_rows = read_csv(excluded)
    require_columns(ex_fields, ["guide_id"], "Step 07 excluded audit")
    _, guide_rows = read_csv(guides)
    expected = {row["guide_id"].strip() for row in guide_rows}
    observed = {row["guide_id"].strip() for row in ex_rows}
    missing = sorted(expected - observed)
    if missing:
        raise QCError(
            f"Step 07: no intended on-target excluded for {len(missing)} guide(s); "
            f"examples: {missing[:10]}"
        )
    overlap = candidate_keys(out_fields, out_rows) & candidate_keys(ex_fields, ex_rows)
    if overlap:
        raise QCError(f"Step 07: {len(overlap)} excluded candidate(s) remain")


def qc_predictions(path: Path, expected_rows: int):
    fields, rows = read_csv(path, allow_empty=(expected_rows == 0))
    if len(rows) != expected_rows:
        raise QCError(f"Step 08: row count changed ({expected_rows} -> {len(rows)})")
    require_columns(fields, ["guide_id", "ensemble_risk", "off_target_probability"], "Step 08")
    for column in ["ensemble_risk", "off_target_probability"]:
        numeric((row[column] for row in rows), f"Step 08 {column}", minimum=0, maximum=1)


def qc_scores(path: Path, guide_count: int):
    fields, rows = read_csv(path)
    metric = "top5_aggregate_off_target_probability"
    require_columns(fields, ["guide_id", metric], "Step 09")
    if len(rows) != guide_count:
        raise QCError(f"Step 09: expected {guide_count} guides, found {len(rows)}")
    if len({row["guide_id"] for row in rows}) != len(rows):
        raise QCError("Step 09: duplicate guide_id values")
    scores = numeric((row[metric] for row in rows), f"Step 09 {metric}", minimum=0, maximum=1)
    if scores != sorted(scores):
        raise QCError(f"Step 09: rows are not sorted by ascending {metric}")


def qc_top20(path: Path, guide_count: int):
    fields, rows = read_csv(path)
    require_columns(
        fields,
        ["rank", "guide_id", "gRNA_20nt", "pam", "top5_aggregate_off_target_probability"],
        "Step 09 Top20",
    )
    expected = min(20, guide_count)
    if len(rows) != expected:
        raise QCError(f"Step 09 Top20: expected {expected} rows, found {len(rows)}")
    if any(not DNA20.fullmatch(row["gRNA_20nt"].strip().upper()) for row in rows):
        raise QCError("Step 09 Top20: invalid gRNA_20nt")


def qc_final(path: Path):
    if not path.is_file() or path.stat().st_size == 0:
        raise QCError(f"Step 10: output is missing or empty: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames or []
        rows = list(reader)
    if rows:
        require_columns(fields, ["final_rank", "gRNA_20nt", "pam"], "Step 10")
        ranks = [row["final_rank"].strip() for row in rows]
        if ranks != [str(i) for i in range(1, len(rows) + 1)]:
            raise QCError("Step 10: final_rank is not consecutive from 1")


def remove_outputs(paths: Iterable[Path]):
    for path in paths:
        if path.is_file() or path.is_symlink():
            path.unlink()


def run_logged(command, env, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    collected = []
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND: " + " ".join(map(str, command)) + "\n")
        process = subprocess.Popen(
            command,
            cwd=OFFTARGET_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            collected.append(line)
        return_code = process.wait()
    return return_code, "".join(collected)


def run_step(number, name, command, outputs, qc: Callable[[], None], env, resume):
    print(f"\n{'=' * 78}\nStep {number:02d}: {name}\n{'=' * 78}", flush=True)
    if resume and outputs and all(path.exists() for path in outputs):
        print("Existing outputs found; validating before skip.", flush=True)
        try:
            qc()
        except QCError as exc:
            print(f"Existing QC failed ({exc}); rerunning.", flush=True)
        else:
            print(f"Step {number:02d} QC: PASS", flush=True)
            return
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    print("Command:", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=OFFTARGET_ROOT, env=env, check=True)
    qc()
    print(f"Step {number:02d} QC: PASS", flush=True)


def pocl_paths():
    """Locate the pocl (CPU OpenCL) ICD and its shared libraries, if installed
    in the current Python environment (e.g. via `conda install -c conda-forge pocl`).
    """
    prefix = Path(sys.prefix)
    vendors_dir = prefix / "etc" / "OpenCL" / "vendors"
    if not vendors_dir.is_dir() or not any(vendors_dir.glob("*.icd")):
        return None, []
    lib_dirs = [d for d in (prefix / "lib", prefix / "lib64") if d.is_dir()]
    return vendors_dir, lib_dirs


def run_cas_step(args, script, paths, python, base_env):
    outputs = [paths["raw"], paths["co_input"], paths["query_map"]]
    print(f"\n{'=' * 78}\nStep 02: run Cas-OFFinder\n{'=' * 78}", flush=True)
    if args.resume and all(path.exists() for path in outputs):
        try:
            qc_raw(paths["raw"])
        except QCError as exc:
            print(f"Existing Step 02 QC failed ({exc}); rerunning.", flush=True)
        else:
            print("Step 02 QC: PASS", flush=True)
            return

    log_path = paths["cas_log"]
    if log_path.exists():
        log_path.unlink()

    def command(device):
        return [
            python, str(script), "--guides", str(paths["guides"]),
            "--pam", args.pam, "--genome", str(args.genome),
            "--output", str(paths["raw"]), "--input-file", str(paths["co_input"]),
            "--query-map", str(paths["query_map"]), "--cas-offinder", str(args.cas_offinder),
            "--max-mismatches", str(args.max_mismatches), "--device", device,
        ]

    attempts = []
    if args.cas_device:
        attempts.append(("explicit", args.cas_device, None))
    elif args.gpu:
        attempts.append(("physical", f"G{args.gpu}", None))
        attempts.append(("visible", "G0", str(args.gpu)))
        if args.allow_cpu_fallback:
            attempts.append(("cpu", "C", None))
    else:
        attempts.append(("cpu", "C", None))

    last_code = 1
    for index, (mode, device, visible_gpu) in enumerate(attempts, 1):
        remove_outputs(outputs)
        env = base_env.copy()
        if visible_gpu is None:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            env["CUDA_VISIBLE_DEVICES"] = visible_gpu
        output = ""
        if device.upper().startswith("G") and not args.opencl_library.is_file():
            output = f"NVIDIA OpenCL library not found: {args.opencl_library}"
            last_code = 1
            print(output, flush=True)
        elif device.upper().startswith("G"):
            env.pop("OCL_ICD_VENDORS", None)
            env["OCL_ICD_FILENAMES"] = str(args.opencl_library)
        elif device.upper().startswith("C"):
            vendors_dir, lib_dirs = pocl_paths()
            if vendors_dir is None:
                raise QCError(
                    "No CPU OpenCL runtime found for Cas-OFFinder. Install one with: "
                    "conda install -n mosqguide-off -c conda-forge pocl"
                )
            env.pop("OCL_ICD_FILENAMES", None)
            env["OCL_ICD_VENDORS"] = str(vendors_dir)
            existing_ld = env.get("LD_LIBRARY_PATH", "")
            new_ld = ":".join(str(d) for d in lib_dirs)
            env["LD_LIBRARY_PATH"] = f"{new_ld}:{existing_ld}" if existing_ld else new_ld

        if not output:
            print(
                f"Cas-OFFinder attempt {index}/{len(attempts)}: mode={mode}, "
                f"device={device}, CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
                flush=True,
            )
            last_code, output = run_logged(command(device), env, log_path)
            if last_code == 0:
                qc_raw(paths["raw"])
                print(f"Step 02 QC: PASS (mode={mode}, device={device})", flush=True)
                return

        device_error = bool(re.search(
            r"No OpenCL devices found|OpenCL|CL_DEVICE|invalid device|device.*not found",
            output,
            flags=re.IGNORECASE,
        ))
        if args.cas_device or (not device_error and mode != "cpu"):
            break
        print("Cas-OFFinder device/OpenCL failure detected; trying next configured mode.", flush=True)

    raise subprocess.CalledProcessError(last_code, command(attempts[-1][1]))


def parse_args():
    parser = argparse.ArgumentParser(description="Unified NGG/PAM-agnostic off-target pipeline orchestration.")
    parser.add_argument("--gene", required=True, help="Run label, e.g. u2af38")
    parser.add_argument("--input", required=True, type=Path, help="CDS-exon FASTA")
    parser.add_argument("--genome", required=True, type=Path, help="Whole-genome FASTA matching the target coordinates and SNP table")
    parser.add_argument("--snps", required=True, type=Path, help="Prepared population SNP TSV")
    parser.add_argument(
        "--pam",
        required=True,
        help="IUPAC PAM pattern, for example NGG, NGTT, TTTV, or NNGRRT",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help=(
            "Scratch work directory for intermediate step outputs. "
            "Defaults to work/<gene>_<pam>/ and is deleted after the "
            "pipeline finishes successfully."
        ),
    )
    parser.add_argument("--guide-prefix", default="gRNA")
    parser.add_argument("--max-mismatches", type=int, default=6)
    parser.add_argument(
        "--cas-offinder", type=Path,
        default=OFFTARGET_ROOT / "tools" / "cas-offinder",
    )
    parser.add_argument("--gpu", default="", help="Physical GPU index; leave empty to run Cas-OFFinder on CPU only")
    parser.add_argument("--cas-device", help="Explicit Cas-OFFinder device; disables automatic retry")
    parser.add_argument("--allow-cpu-fallback", action="store_true", default=True)
    parser.add_argument(
        "--opencl-library", type=Path,
        default=Path("/lib/x86_64-linux-gnu/libnvidia-opencl.so.1"),
    )
    parser.add_argument(
        "--score-python", type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument("--score-device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dl-checkpoint", type=Path)
    parser.add_argument("--rf-model", type=Path)
    parser.add_argument("--dl-calibrator", type=Path)
    parser.add_argument("--rf-calibrator", type=Path)
    parser.add_argument("--ensemble-weight", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--on-target-top20", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--keep-work-dir", action="store_true",
        help="Keep the scratch work directory after a successful run (for debugging).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.gene):
        raise SystemExit("--gene must be a safe run label starting with a letter or digit.")
    args.pam = args.pam.strip().upper()
    pam_regex(args.pam)
    model_profile = "ngg" if args.pam == "NGG" else "pam-agnostic"
    profile = MODEL_PROFILES[model_profile]

    for name in ["dl_checkpoint", "rf_model", "dl_calibrator", "rf_calibrator"]:
        if getattr(args, name) is None:
            setattr(args, name, profile[name])
    if args.ensemble_weight is None:
        args.ensemble_weight = profile["ensemble_weight"]

    args.input = absolute(args.input)
    args.genome = absolute(args.genome)
    args.snps = absolute(args.snps)
    args.cas_offinder = absolute(args.cas_offinder)
    args.opencl_library = absolute(args.opencl_library)
    args.score_python = absolute(args.score_python)
    if args.output_dir:
        args.output_dir = absolute(args.output_dir)
    if args.on_target_top20:
        args.on_target_top20 = absolute(args.on_target_top20)
    for name in ["dl_checkpoint", "rf_model", "dl_calibrator", "rf_calibrator"]:
        value = getattr(args, name)
        if value:
            setattr(args, name, absolute(value))

    if args.max_mismatches < 0:
        raise SystemExit("--max-mismatches must be >= 0")
    if args.ensemble_weight is not None and not 0 <= args.ensemble_weight <= 1:
        raise SystemExit("--ensemble-weight must be in [0,1]")

    scripts = {
        number: PIPELINE_DIR / filename for number, filename in {
            1: "01_generate_top20_guides.py", 2: "02_run_cas_offinder.py",
            3: "03_prepare_candidates.py", 4: "04_apply_population_snps.py",
            5: "05_find_snp_created_pams.py", 6: "06_merge_population_candidates.py",
            7: "07_exclude_on_target.py", 8: "08_score_candidates.py",
            9: "09_aggregate_guide_risk.py", 10: "10_integrate_on_off_target.py",
        }.items()
    }
    required = [args.input, args.genome, args.snps, args.cas_offinder, args.score_python]
    required += [scripts[number] for number in range(1, 11)]
    required += [value for value in [args.dl_checkpoint, args.rf_model, args.dl_calibrator, args.rf_calibrator] if value]
    required += [args.on_target_top20]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing required files:\n  " + "\n  ".join(missing))
    if not os.access(args.cas_offinder, os.X_OK):
        raise SystemExit(f"Cas-OFFinder is not executable: {args.cas_offinder}")
    if not os.access(args.score_python, os.X_OK):
        raise SystemExit(f"Scoring Python is not executable: {args.score_python}")

    owns_work_dir = False
    if args.output_dir:
        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        print(f"Using caller-provided work directory (will not be deleted): {out}", flush=True)
    elif args.resume:
        out = OFFTARGET_ROOT / "work" / f"{args.gene}_{args.pam}"
        out.mkdir(parents=True, exist_ok=True)
    else:
        work_root = OFFTARGET_ROOT / "work"
        work_root.mkdir(parents=True, exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix=f"{args.gene}_{args.pam}_", dir=work_root))
        owns_work_dir = True
    paths = {
        "guides": out / "01_guides.csv",
        "raw": out / "02_cas_offinder.txt",
        "co_input": out / "02_cas_offinder_input.txt",
        "query_map": out / "02_query_map.tsv",
        "cas_log": out / "02_cas_offinder_attempts.log",
        "reference": out / "03_reference_candidates.csv",
        "population": out / "04_population_candidates.csv",
        "created": out / "05_snp_created_pam_candidates.csv",
        "merged": out / "06_merged_candidates.csv",
        "offtargets": out / "07_off_target_candidates.csv",
        "excluded": out / "07_excluded_on_targets.csv",
        "predictions": out / "08_candidate_predictions.csv",
        "scores": out / "09_guide_offtarget_scores.csv",
        "top20": out / "09_guide_offtarget_scores_top20.csv",
        "final": out / "10_high_quality_gRNAs.tsv",
    }

    orchestration_python = sys.executable
    normal_env = os.environ.copy()
    score_env = os.environ.copy()
    if args.score_device.startswith("cuda"):
        score_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(
        f"PAM: {args.pam} | model profile: {model_profile} | "
        f"DL weight: {args.ensemble_weight:.2f}",
        flush=True,
    )

    run_step(
        1, f"generate and select on-target Top20 {args.pam} guides",
        [
            orchestration_python,
            str(scripts[1]),
            "--input",
            str(args.input),
            "--pam",
            args.pam,
            "--on-target-top20",
            str(args.on_target_top20),
            "--output",
            str(paths["guides"]),
            "--guide-prefix",
            args.guide_prefix,
        ],
        [paths["guides"]],
        lambda: qc_guides(paths["guides"], args.pam),
        normal_env,
        args.resume,
    )
    run_cas_step(args, scripts[2], paths, orchestration_python, normal_env)
    run_step(
        3, "prepare reference candidates",
        [orchestration_python, str(scripts[3]), "--raw", str(paths["raw"]),
         "--query-map", str(paths["query_map"]), "--pam", args.pam,
         "--output", str(paths["reference"])],
        [paths["reference"]], lambda: qc_candidates(paths["reference"], "Step 03", args.max_mismatches),
        normal_env, args.resume,
    )
    run_step(
        4, "apply population SNPs",
        [orchestration_python, str(scripts[4]), "--candidates", str(paths["reference"]),
         "--snps", str(args.snps), "--pam", args.pam, "--output", str(paths["population"])],
        [paths["population"]], lambda: qc_population(paths["population"], args.max_mismatches),
        normal_env, args.resume,
    )
    run_step(
        5, f"find SNP-created {args.pam} PAMs",
        [orchestration_python, str(scripts[5]), "--guides", str(paths["guides"]),
         "--snps", str(args.snps), "--fasta", str(args.genome), "--pam", args.pam,
         "--max-mismatch", str(args.max_mismatches), "--output", str(paths["created"])],
        [paths["created"]],
        lambda: qc_candidates(paths["created"], "Step 05", args.max_mismatches, allow_empty=True),
        normal_env, args.resume,
    )
    run_step(
        6, "merge population candidates",
        [orchestration_python, str(scripts[6]), "--population-candidates", str(paths["population"]),
         "--created-pam-candidates", str(paths["created"]), "--pam", args.pam,
         "--max-mismatch", str(args.max_mismatches), "--output", str(paths["merged"])],
        [paths["merged"]], lambda: qc_merged(paths["merged"], args.max_mismatches),
        normal_env, args.resume,
    )
    run_step(
        7, "exclude intended on-target loci",
        [orchestration_python, str(scripts[7]), "--guides", str(paths["guides"]),
         "--candidates", str(paths["merged"]), "--output", str(paths["offtargets"]),
         "--excluded-output", str(paths["excluded"])],
        [paths["offtargets"], paths["excluded"]],
        lambda: qc_exclusion(paths["offtargets"], paths["excluded"], paths["guides"], args.max_mismatches),
        normal_env, args.resume,
    )

    _, off_rows = read_csv(paths["offtargets"], allow_empty=True)
    command8 = [
        str(args.score_python), str(scripts[8]), "--input", str(paths["offtargets"]),
        "--guides", str(paths["guides"]), "--pam", args.pam,
        "--model-profile", model_profile,
        "--output", str(paths["predictions"]), "--device", args.score_device,
        "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers),
    ]
    optional8 = [
        ("--dl-checkpoint", args.dl_checkpoint), ("--rf-model", args.rf_model),
        ("--dl-calibrator", args.dl_calibrator), ("--rf-calibrator", args.rf_calibrator),
        ("--ensemble-weight", args.ensemble_weight),
    ]
    for flag, value in optional8:
        if value is not None:
            command8 += [flag, str(value)]
    run_step(
        8, f"score with frozen {model_profile} ensemble and calibrators",
        command8, [paths["predictions"]],
        lambda: qc_predictions(paths["predictions"], len(off_rows)), score_env, args.resume,
    )

    command9 = [
        orchestration_python, str(scripts[9]), "--predictions", str(paths["predictions"]),
        "--population-table", str(paths["population"]), "--guides", str(paths["guides"]),
        "--output", str(paths["scores"]), "--max-mismatch", str(args.max_mismatches),
    ]
    if args.alpha is not None:
        command9 += ["--alpha", str(args.alpha)]
    _, guide_rows = read_csv(paths["guides"])

    def qc9():
        qc_scores(paths["scores"], len(guide_rows))
        qc_top20(paths["top20"], len(guide_rows))

    run_step(
        9, "aggregate guide-level population risk", command9,
        [paths["scores"], paths["top20"]], qc9, normal_env, args.resume,
    )

    run_step(
        10, "annotate on-target Top20 with off-target risk",
        [orchestration_python, str(scripts[10]), "--off-target", str(paths["scores"]),
         "--on-target", str(args.on_target_top20), "--output", str(paths["final"])],
        [paths["final"]], lambda: qc_final(paths["final"]), normal_env, args.resume,
    )

    # The only artifact the pipeline leaves under results/ is this final file.
    final_dir = OFFTARGET_ROOT / "results" / args.gene
    final_dir.mkdir(parents=True, exist_ok=True)
    final_output = final_dir / f"top20_{args.pam.lower()}_gRNA.tsv"
    shutil.copy2(paths["final"], final_output)

    if owns_work_dir and not args.keep_work_dir:
        shutil.rmtree(out, ignore_errors=True)

    print(f"\nPipeline complete. Final guides: {final_output}")


if __name__ == "__main__":
    try:
        main()
    except (QCError, subprocess.CalledProcessError) as exc:
        print(f"\nPIPELINE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
