#!/usr/bin/env python3

import argparse
import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Project paths
# ============================================================

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if (PIPELINE_ROOT / "pam_agnostic").is_dir():
    OFFTARGET_ROOT = PIPELINE_ROOT
    PAM_AGNOSTIC_ROOT = OFFTARGET_ROOT / "pam_agnostic"
else:
    PAM_AGNOSTIC_ROOT = PIPELINE_ROOT
    OFFTARGET_ROOT = PAM_AGNOSTIC_ROOT.parent

SHARED_SRC_DIR = OFFTARGET_ROOT / "src"
SRC_DIR = PAM_AGNOSTIC_ROOT / "src"
FEATURE_DIR = PAM_AGNOSTIC_ROOT / "feature_enhancement"

for path in [FEATURE_DIR, SRC_DIR, SHARED_SRC_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from dataset import PAIR_TO_ID
from model import OffTargetModel
from inspect_features import build_handcrafted_features as build_pam_agnostic_features
from pam_utils import pam_matches


# ============================================================
# Frozen model configuration
# ============================================================

FEATURE_DIM = 81
MODEL_PROFILES = ("ngg", "pam-agnostic")


def first_existing_path(*paths):
    return next((path for path in paths if path.exists()), paths[0])

DEFAULT_DL_CHECKPOINT = (
    PAM_AGNOSTIC_ROOT / "checkpoints" / "offtarget_pam_agnostic_focal_best.pt"
)

DEFAULT_RF_MODEL = (
    PAM_AGNOSTIC_ROOT / "checkpoints" / "rf_handcrafted_baseline.joblib"
)

DEFAULT_DL_CALIBRATOR = (
    PAM_AGNOSTIC_ROOT / "checkpoints" / "dl_probability_calibrator.joblib"
)

DEFAULT_RF_CALIBRATOR = (
    PAM_AGNOSTIC_ROOT / "checkpoints" / "rf_probability_calibrator.joblib"
)

DEFAULT_DL_WEIGHT = 0.84
NGG_ROOT = first_existing_path(
    OFFTARGET_ROOT / "pam_ngg",
    OFFTARGET_ROOT / "off_target_model_NGG",
)
PROFILE_DEFAULTS = {
    "ngg": {
        "dl_checkpoint": NGG_ROOT / "checkpoints" / "offtarget_focal_best.pt",
        "rf_model": NGG_ROOT / "checkpoints" / "rf_handcrafted_baseline.joblib",
        "dl_calibrator": NGG_ROOT / "checkpoints" / "dl_probability_calibrator.joblib",
        "rf_calibrator": NGG_ROOT / "checkpoints" / "rf_probability_calibrator.joblib",
        "ensemble_weight": 0.86,
    },
    "pam-agnostic": {
        "dl_checkpoint": DEFAULT_DL_CHECKPOINT,
        "rf_model": DEFAULT_RF_MODEL,
        "dl_calibrator": DEFAULT_DL_CALIBRATOR,
        "rf_calibrator": DEFAULT_RF_CALIBRATOR,
        "ensemble_weight": DEFAULT_DL_WEIGHT,
    },
}


def load_ngg_feature_builder():
    module_path = (
        NGG_ROOT / "src" / "inspect_features.py"
    )
    if not module_path.is_file():
        raise FileNotFoundError(
            f"NGG feature builder not found: {module_path}"
        )
    spec = importlib.util.spec_from_file_location(
        "ngg_inspect_features", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load NGG feature builder: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_handcrafted_features


def encode_pair(sgrna, off_target, sequence_length):
    guide = str(sgrna).strip().upper()
    target = str(off_target).strip().upper()
    if len(guide) != sequence_length or len(target) != sequence_length:
        raise ValueError(
            f"Both sequences must have length {sequence_length}, "
            f"got {len(guide)} and {len(target)}"
        )
    try:
        return [PAIR_TO_ID[g + t] for g, t in zip(guide, target)]
    except KeyError as exc:
        raise ValueError(f"Invalid DNA pair: {exc.args[0]}") from exc


# ============================================================
# Column aliases
# ============================================================

SGRNA_ALIASES = [
    "sgRNA_20nt",
    "sgRNA",
    "spacer",
    "guide_sequence",
]

OFFTARGET_ALIASES = [
    "off_target_20nt",
    "off_target",
    "candidate_20nt",
    "candidate_sequence",
]


def resolve_column(df, aliases, role):

    for col in aliases:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find {role} column.\n"
        f"Tried: {aliases}\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# Pair dataset
# ============================================================

class PairDataset(Dataset):

    def __init__(
        self,
        sgrnas,
        off_targets,
        sequence_length,
    ):
        self.sgrnas = sgrnas
        self.off_targets = off_targets
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.sgrnas)

    def __getitem__(self, idx):

        pair_ids = encode_pair(
            self.sgrnas[idx],
            self.off_targets[idx],
            self.sequence_length,
        )

        return torch.tensor(
            pair_ids,
            dtype=torch.long,
        )


# ============================================================
# Device
# ============================================================

def resolve_device(device_arg):

    import os

    value = str(device_arg).strip().lower()

    if value.isdigit():

        os.environ["CUDA_VISIBLE_DEVICES"] = value

        if torch.cuda.is_available():
            return torch.device("cuda")

        print(
            "[WARN] CUDA unavailable. "
            "Falling back to CPU."
        )

        return torch.device("cpu")

    if value == "cuda":

        if torch.cuda.is_available():
            return torch.device("cuda")

        print(
            "[WARN] CUDA unavailable. "
            "Falling back to CPU."
        )

        return torch.device("cpu")

    if value == "cpu":
        return torch.device("cpu")

    raise ValueError(
        "--device must be cpu, cuda, "
        "or a GPU ID such as 6."
    )


# ============================================================
# Input QC
# ============================================================

def load_candidates(path, guides_path, pam_pattern, model_profile):

    print(f"Loading candidates: {path}")

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    if len(df) == 0:
        raise ValueError(
            "Candidate file contains zero rows."
        )

    sgrna_col = resolve_column(
        df,
        SGRNA_ALIASES,
        "sgRNA 20nt",
    )

    off_col = resolve_column(
        df,
        OFFTARGET_ALIASES,
        "off-target 20nt",
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    df[sgrna_col] = (
        df[sgrna_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df[off_col] = (
        df[off_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Strict sequence and PAM QC
    # --------------------------------------------------------

    sgrna_valid = (
        df[sgrna_col]
        .str.fullmatch(r"[ACGT]{20}")
    )

    off_valid = (
        df[off_col]
        .str.fullmatch(r"[ACGT]{20}")
    )

    valid = (
        sgrna_valid
        & off_valid
    )

    invalid_count = int(
        (~valid).sum()
    )

    if invalid_count > 0:

        bad = df.loc[
            ~valid,
            [sgrna_col, off_col],
        ].head(10)

        raise ValueError(
            f"Detected {invalid_count:,} invalid "
            f"20nt sequence pairs.\n"
            f"Examples:\n{bad}"
        )

    if "candidate_pam" not in df.columns:
        raise ValueError(
            "Candidate file is missing candidate_pam."
        )

    guides = pd.read_csv(
        guides_path,
        usecols=["guide_id", "pam"],
        dtype={"guide_id": str},
    )
    if guides["guide_id"].duplicated().any():
        raise ValueError("Guide table contains duplicate guide_id values.")
    df["guide_id"] = df["guide_id"].astype(str)
    df = df.merge(guides, on="guide_id", how="left", validate="many_to_one")
    if df["pam"].isna().any():
        missing = df.loc[df["pam"].isna(), "guide_id"].drop_duplicates().head(10)
        raise ValueError(f"Candidate guide_id missing from guide table: {missing.tolist()}")

    for column in ["pam", "candidate_pam"]:
        df[column] = df[column].astype(str).str.strip().str.upper()
        matches = df[column].map(lambda value: pam_matches(value, pam_pattern))
        if not matches.all():
            bad = df.loc[~matches, column].head(10).tolist()
            raise ValueError(
                f"{column} contains PAMs that do not match {pam_pattern}: {bad}"
            )

    if model_profile == "ngg" and pam_pattern != "NGG":
        raise ValueError("The NGG model profile requires --pam NGG.")

    df["sgRNA_model"] = df[sgrna_col]
    df["off_target_model"] = df[off_col]
    if model_profile == "ngg":
        df["sgRNA_model"] = df[sgrna_col] + df["pam"]
        df["off_target_model"] = df[off_col] + df["candidate_pam"]

    print()
    print("=" * 70)
    print("Candidate QC")
    print("=" * 70)

    print(
        f"Candidates:            "
        f"{len(df):,}"
    )

    if "guide_id" in df.columns:

        print(
            f"Guides:                "
            f"{df['guide_id'].nunique():,}"
        )

    print(
        f"Invalid sequences:     "
        f"{invalid_count:,}"
    )

    print(
        f"sgRNA column:          "
        f"{sgrna_col}"
    )

    print(
        f"Off-target column:     "
        f"{off_col}"
    )

    return (
        df,
        "sgRNA_model",
        "off_target_model",
    )


# ============================================================
# DL raw prediction
# ============================================================

def predict_dl(
    df,
    sgrna_col,
    off_col,
    checkpoint_path,
    device,
    batch_size,
    num_workers,
    sequence_length,
):

    print()
    print("=" * 70)
    print("DL raw prediction")
    print("=" * 70)

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = OffTargetModel(seq_len=sequence_length)

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(device)
    model.eval()

    dataset = PairDataset(
        df[sgrna_col].tolist(),
        df[off_col].tolist(),
        sequence_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    predictions = []

    total_batches = len(loader)

    with torch.no_grad():

        for batch_idx, pair_ids in enumerate(
            loader,
            start=1,
        ):

            pair_ids = pair_ids.to(
                device,
                non_blocking=True,
            )

            logits = model(
                pair_ids
            )

            raw_score = torch.sigmoid(
                logits
            )

            predictions.append(
                raw_score
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            if (
                batch_idx % 100 == 0
                or batch_idx == total_batches
            ):

                print(
                    f"DL batches: "
                    f"{batch_idx:,} / "
                    f"{total_batches:,}"
                )

    predictions = np.concatenate(
        predictions
    ).astype(
        np.float32
    )

    if len(predictions) != len(df):

        raise RuntimeError(
            "DL prediction length mismatch."
        )

    print()
    print(
        f"DL mean raw score:     "
        f"{predictions.mean():.6f}"
    )

    print(
        f"DL min raw score:      "
        f"{predictions.min():.6f}"
    )

    print(
        f"DL max raw score:      "
        f"{predictions.max():.6f}"
    )

    return predictions


# ============================================================
# RF raw prediction
# ============================================================

def predict_rf(
    df,
    sgrna_col,
    off_col,
    rf_model_path,
    feature_dim,
    feature_builder,
):

    print()
    print("=" * 70)
    print("RF raw prediction")
    print("=" * 70)

    print(
        f"RF model: "
        f"{rf_model_path}"
    )

    rf_model = joblib.load(
        rf_model_path
    )

    model_feature_dim = getattr(rf_model, "n_features_in_", None)
    if model_feature_dim != feature_dim:
        raise ValueError(
            f"RF model expects {model_feature_dim} features; "
            f"profile requires {feature_dim}."
        )

    n = len(df)

    features = np.empty(
        (
            n,
            feature_dim,
        ),
        dtype=np.float32,
    )

    sgrnas = df[
        sgrna_col
    ].tolist()

    off_targets = df[
        off_col
    ].tolist()

    for i, (
        sgrna,
        off_target,
    ) in enumerate(
        zip(
            sgrnas,
            off_targets,
        )
    ):

        _, vector = (
            feature_builder(
                sgrna,
                off_target,
            )
        )

        vector = np.asarray(
            vector,
            dtype=np.float32,
        ).reshape(-1)

        if len(vector) != feature_dim:

            raise RuntimeError(
                f"RF feature dimension error "
                f"at row {i}: "
                f"{len(vector)} != {feature_dim}"
            )

        features[i] = vector

        if (
            (i + 1) % 50000 == 0
            or (i + 1) == n
        ):

            print(
                f"RF features: "
                f"{i + 1:,} / "
                f"{n:,}"
            )

    print()
    print(
        "Running RF predict_proba..."
    )

    predictions = (
        rf_model
        .predict_proba(
            features
        )[:, 1]
        .astype(
            np.float32
        )
    )

    if len(predictions) != n:

        raise RuntimeError(
            "RF prediction length mismatch."
        )

    print()
    print(
        f"RF mean raw score:     "
        f"{predictions.mean():.6f}"
    )

    print(
        f"RF min raw score:      "
        f"{predictions.min():.6f}"
    )

    print(
        f"RF max raw score:      "
        f"{predictions.max():.6f}"
    )

    return predictions


# ============================================================
# Calibration
# ============================================================

def apply_calibrator(
    calibrator_bundle,
    raw_scores,
):

    if not isinstance(
        calibrator_bundle,
        dict,
    ):

        raise ValueError(
            "Calibrator file must contain "
            "a dictionary bundle."
        )

    if (
        "method" not in calibrator_bundle
        or "model" not in calibrator_bundle
    ):

        raise ValueError(
            "Invalid calibrator bundle. "
            "Expected keys: method, model."
        )

    method = calibrator_bundle[
        "method"
    ]

    model = calibrator_bundle[
        "model"
    ]

    raw_scores = np.asarray(
        raw_scores,
        dtype=float,
    )

    if method == "platt":

        probability = (
            model
            .predict_proba(
                raw_scores.reshape(-1, 1)
            )[:, 1]
        )

    elif method == "isotonic":

        probability = np.asarray(
            model.predict(
                raw_scores
            ),
            dtype=float,
        )

    else:

        raise ValueError(
            f"Unknown calibration method: "
            f"{method}"
        )

    probability = np.asarray(
        probability,
        dtype=np.float32,
    )

    if np.isnan(
        probability
    ).any():

        raise RuntimeError(
            f"NaN detected after "
            f"{method} calibration."
        )

    if not np.all(
        (
            probability >= 0.0
        )
        &
        (
            probability <= 1.0
        )
    ):

        raise RuntimeError(
            f"Calibrated probability from "
            f"{method} outside [0,1]."
        )

    return probability


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Score off-target candidates using "
            "frozen PAM-agnostic DL + RF models "
            "and frozen probability calibrators."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help=(
            "Candidate CSV after intended "
            "on-target exclusion."
        ),
    )

    parser.add_argument(
        "--pam",
        required=True,
        help="Concrete/IUPAC PAM pattern used to generate candidates.",
    )

    parser.add_argument(
        "--guides",
        required=True,
        type=Path,
        help="Guide CSV produced by Step01; supplies each guide's concrete PAM.",
    )

    parser.add_argument(
        "--model-profile",
        choices=MODEL_PROFILES,
        required=True,
        help="NGG uses 23nt/93 features; PAM-agnostic uses 20nt/81 features.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--dl-checkpoint",
        type=Path,
    )

    parser.add_argument(
        "--rf-model",
        type=Path,
    )

    parser.add_argument(
        "--dl-calibrator",
        type=Path,
    )

    parser.add_argument(
        "--rf-calibrator",
        type=Path,
    )

    parser.add_argument(
        "--ensemble-weight",
        type=float,
        help=(
            "DL probability weight. "
            "RF weight = 1 - this value."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    args.pam = args.pam.strip().upper()
    profile_defaults = PROFILE_DEFAULTS[args.model_profile]
    for name in ["dl_checkpoint", "rf_model", "dl_calibrator", "rf_calibrator"]:
        if getattr(args, name) is None:
            setattr(args, name, profile_defaults[name])
    if args.ensemble_weight is None:
        args.ensemble_weight = profile_defaults["ensemble_weight"]

    if args.model_profile == "ngg":
        sequence_length = 23
        feature_dim = 93
        feature_builder = load_ngg_feature_builder()
    else:
        sequence_length = 20
        feature_dim = 81
        feature_builder = build_pam_agnostic_features

    # --------------------------------------------------------
    # File checks
    # --------------------------------------------------------

    required_paths = [
        args.input,
        args.guides,
        args.dl_checkpoint,
        args.rf_model,
        args.dl_calibrator,
        args.rf_calibrator,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    if not (
        0.0
        <= args.ensemble_weight
        <= 1.0
    ):

        raise ValueError(
            "--ensemble-weight must "
            "be between 0 and 1."
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = resolve_device(
        args.device
    )

    print("=" * 70)
    print(
        "PAM-agnostic off-target candidate scoring"
    )
    print("=" * 70)

    print(
        f"Device:                "
        f"{device}"
    )

    if device.type == "cuda":

        print(
            f"GPU:                   "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"DL probability weight: "
        f"{args.ensemble_weight:.2f}"
    )

    print(
        f"RF probability weight: "
        f"{1.0 - args.ensemble_weight:.2f}"
    )

    # --------------------------------------------------------
    # Load candidates
    # --------------------------------------------------------

    (
        df,
        sgrna_col,
        off_col,
    ) = load_candidates(
        args.input,
        args.guides,
        args.pam,
        args.model_profile,
    )

    # --------------------------------------------------------
    # Raw model predictions
    # --------------------------------------------------------

    dl_raw_score = predict_dl(
        df,
        sgrna_col,
        off_col,
        args.dl_checkpoint,
        device,
        args.batch_size,
        args.num_workers,
        sequence_length,
    )

    rf_raw_score = predict_rf(
        df,
        sgrna_col,
        off_col,
        args.rf_model,
        feature_dim,
        feature_builder,
    )

    # --------------------------------------------------------
    # Old raw-score ensemble
    #
    # Kept for audit/backward compatibility.
    # This is NOT a calibrated probability.
    # --------------------------------------------------------

    w = args.ensemble_weight

    ensemble_raw_score = (
        w * dl_raw_score
        +
        (1.0 - w)
        * rf_raw_score
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Load frozen calibrators
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "Probability calibration"
    )
    print("=" * 70)

    dl_calibrator = joblib.load(
        args.dl_calibrator
    )

    rf_calibrator = joblib.load(
        args.rf_calibrator
    )

    print(
        f"DL calibration method: "
        f"{dl_calibrator['method']}"
    )

    print(
        f"RF calibration method: "
        f"{rf_calibrator['method']}"
    )

    # --------------------------------------------------------
    # Individual calibrated probabilities
    # --------------------------------------------------------

    dl_probability = apply_calibrator(
        dl_calibrator,
        dl_raw_score,
    )

    rf_probability = apply_calibrator(
        rf_calibrator,
        rf_raw_score,
    )

    # --------------------------------------------------------
    # Final candidate-level probability
    # --------------------------------------------------------

    off_target_probability = (
        w * dl_probability
        +
        (1.0 - w)
        * rf_probability
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Final QC
    # --------------------------------------------------------

    arrays_to_check = {
        "dl_raw_score":
            dl_raw_score,

        "rf_raw_score":
            rf_raw_score,

        "ensemble_raw_score":
            ensemble_raw_score,

        "dl_probability":
            dl_probability,

        "rf_probability":
            rf_probability,

        "off_target_probability":
            off_target_probability,
    }

    for name, values in (
        arrays_to_check.items()
    ):

        if len(values) != len(df):

            raise RuntimeError(
                f"{name} length mismatch."
            )

        if np.isnan(values).any():

            raise RuntimeError(
                f"NaN detected in {name}."
            )

        if not np.all(
            (
                values >= 0
            )
            &
            (
                values <= 1
            )
        ):

            raise RuntimeError(
                f"{name} contains values "
                f"outside [0,1]."
            )

    # --------------------------------------------------------
    # Add prediction columns
    # --------------------------------------------------------

    output = df.copy()

    output[
        "dl_raw_score"
    ] = dl_raw_score

    output[
        "rf_raw_score"
    ] = rf_raw_score

    output[
        "ensemble_raw_score"
    ] = ensemble_raw_score

    output[
        "dl_probability"
    ] = dl_probability

    output[
        "rf_probability"
    ] = rf_probability

    output[
        "off_target_probability"
    ] = off_target_probability

    # --------------------------------------------------------
    # Backward-compatible alias
    #
    # Existing aggregation code may still expect ensemble_risk.
    # It remains the OLD uncalibrated ensemble score.
    # We will remove/change this when aggregation is updated.
    # --------------------------------------------------------

    output[
        "ensemble_risk"
    ] = ensemble_raw_score

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        args.output,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "Candidate scoring complete"
    )
    print("=" * 70)

    print(
        f"Candidates scored:      "
        f"{len(output):,}"
    )

    if "guide_id" in output.columns:

        print(
            f"Guides:                 "
            f"{output['guide_id'].nunique():,}"
        )

    print()
    print(
        "Raw scores "
        "(NOT calibrated probabilities)"
    )

    print(
        f"DL raw mean:            "
        f"{dl_raw_score.mean():.6f}"
    )

    print(
        f"RF raw mean:            "
        f"{rf_raw_score.mean():.6f}"
    )

    print(
        f"Ensemble raw mean:      "
        f"{ensemble_raw_score.mean():.6f}"
    )

    print()
    print(
        "Calibrated probabilities"
    )

    print(
        f"DL probability mean:    "
        f"{dl_probability.mean():.6f}"
    )

    print(
        f"RF probability mean:    "
        f"{rf_probability.mean():.6f}"
    )

    print(
        f"Final probability mean: "
        f"{off_target_probability.mean():.6f}"
    )

    print(
        f"Final probability min:  "
        f"{off_target_probability.min():.6f}"
    )

    print(
        f"Final probability max:  "
        f"{off_target_probability.max():.6f}"
    )

    print()
    print(
        f"Output: {args.output}"
    )


if __name__ == "__main__":
    main()