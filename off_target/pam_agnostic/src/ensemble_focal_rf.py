"""Ensemble the focal-loss PAM-agnostic DL model (alpha=0.25, gamma=2.0) with
the Random Forest trained on handcrafted (no-PAM) features.

Blend rule: ensemble_prob = w * dl_prob + (1 - w) * rf_prob.
The blend weight w is grid-searched on the validation split (maximizing
AUPRC) and then applied, frozen, to the test split. This is searched fresh
for the PAM-agnostic model rather than reusing the NGG project's w=0.86.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


FEATURE_ENHANCEMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FEATURE_ENHANCEMENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

for path in (FEATURE_ENHANCEMENT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inspect_features import build_handcrafted_features  # noqa: E402
from dataset import create_dataloader  # noqa: E402  (src/dataset.py)
from model import OffTargetModel  # noqa: E402  (src/model.py)


FEATURE_DIM = 81


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "val.csv"),
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "test.csv"),
    )
    parser.add_argument(
        "--dl_checkpoint_path",
        type=str,
        default=str(
            PROJECT_ROOT / "checkpoints" / "offtarget_pam_agnostic_focal_best.pt"
        ),
    )
    parser.add_argument(
        "--rf_model_path",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "rf_handcrafted_baseline.joblib"),
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for the DL model: 'cpu', 'cuda', '3', or 'cuda:3'.",
    )
    parser.add_argument(
        "--weight_grid_steps",
        type=int,
        default=101,
        help="Number of w values swept in [0, 1] on the validation split.",
    )
    parser.add_argument(
        "--ensemble_weight_path",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "ensemble_focal_rf_weight.json"),
        help="Where to save the frozen blend weight w (JSON) after grid search on val.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    normalized = str(device_arg).strip().lower()
    if normalized in ("cpu", ""):
        return torch.device("cpu")
    if normalized.isdigit():
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", normalized)
    elif normalized.startswith("cuda:"):
        gpu_id = normalized.split(":", 1)[1]
        if not gpu_id.isdigit():
            raise ValueError(f"Invalid --device value: {device_arg}")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_id)
    elif normalized != "cuda":
        raise ValueError(
            f"Invalid --device value: {device_arg}. Use 'cpu', 'cuda', '<gpu_id>', or 'cuda:<gpu_id>'."
        )
    if not torch.cuda.is_available():
        print("[WARN] CUDA unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda")


def load_dl_model(checkpoint_path: str, device: torch.device) -> OffTargetModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = OffTargetModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_dl(model: OffTargetModel, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    all_labels: List[torch.Tensor] = []
    all_probs: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            pair_ids = batch["pair_ids"].to(device)
            probs = torch.sigmoid(model(pair_ids))
            all_labels.append(batch["label"].detach().cpu())
            all_probs.append(probs.detach().cpu())
    labels = torch.cat(all_labels).numpy().astype(int)
    probs = torch.cat(all_probs).numpy().astype(float)
    return labels, probs


def predict_rf(model, csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path, usecols=["sgRNA", "off_target"])
    rows = np.empty((len(df), FEATURE_DIM), dtype=np.float32)
    for row_index, (sgrna, off_target) in enumerate(zip(df["sgRNA"], df["off_target"])):
        _, vector = build_handcrafted_features(sgrna, off_target)
        rows[row_index] = vector
    return model.predict_proba(rows)[:, 1]


def safe_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def best_f1_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    if thresholds.size == 0:
        return 0.5
    p, r = precision[:-1], recall[:-1]
    denom = p + r
    f1_scores = np.where(denom > 0, 2.0 * p * r / denom, 0.0)
    return float(thresholds[int(np.nanargmax(f1_scores))])


def calculate_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "AUROC": safe_auroc(labels, probs),
        "AUPRC": float(average_precision_score(labels, probs)),
        "Precision": float(precision_score(labels, preds, zero_division=0)),
        "Recall": float(recall_score(labels, preds, zero_division=0)),
        "F1": float(f1_score(labels, preds, zero_division=0)),
        "MCC": float(matthews_corrcoef(labels, preds)),
    }


def search_best_weight(
    labels: np.ndarray, dl_probs: np.ndarray, rf_probs: np.ndarray, steps: int
) -> Tuple[float, float]:
    """Grid-search blend weight w in [0, 1] maximizing validation AUPRC."""
    best_w, best_auprc = 0.5, -1.0
    for w in np.linspace(0.0, 1.0, steps):
        blend = w * dl_probs + (1.0 - w) * rf_probs
        auprc = float(average_precision_score(labels, blend))
        if auprc > best_auprc:
            best_auprc, best_w = auprc, float(w)
    return best_w, best_auprc


def format_metrics(name: str, metrics: Dict[str, float]) -> str:
    return "\n".join(
        [
            f"{name}:",
            f"  threshold: {metrics['threshold']:.6f}",
            f"  AUROC: {metrics['AUROC']:.6f}",
            f"  AUPRC: {metrics['AUPRC']:.6f}",
            f"  Precision: {metrics['Precision']:.6f}",
            f"  Recall: {metrics['Recall']:.6f}",
            f"  F1: {metrics['F1']:.6f}",
            f"  MCC: {metrics['MCC']:.6f}",
        ]
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    print("Loading DL (focal loss, alpha=0.25, gamma=2.0, PAM-agnostic 20nt) model...")
    dl_model = load_dl_model(args.dl_checkpoint_path, device)

    print("Loading RF (handcrafted features, no PAM) model...")
    rf_model = joblib.load(args.rf_model_path)

    val_loader = create_dataloader(
        csv_path=args.val_path, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = create_dataloader(
        csv_path=args.test_path, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    print("Predicting DL probabilities on val/test...")
    val_labels, val_dl_probs = predict_dl(dl_model, val_loader, device)
    test_labels, test_dl_probs = predict_dl(dl_model, test_loader, device)

    print("Predicting RF probabilities on val/test...")
    val_rf_probs = predict_rf(rf_model, args.val_path)
    test_rf_probs = predict_rf(rf_model, args.test_path)

    best_w, best_val_auprc = search_best_weight(
        val_labels, val_dl_probs, val_rf_probs, args.weight_grid_steps
    )
    print(f"Best blend weight (DL share) on val: w={best_w:.3f}, val AUPRC={best_val_auprc:.6f}")

    val_ensemble_probs = best_w * val_dl_probs + (1.0 - best_w) * val_rf_probs
    val_threshold = best_f1_threshold(val_labels, val_ensemble_probs)

    test_ensemble_probs = best_w * test_dl_probs + (1.0 - best_w) * test_rf_probs

    dl_test_metrics = calculate_metrics(test_labels, test_dl_probs, best_f1_threshold(val_labels, val_dl_probs))
    rf_test_metrics = calculate_metrics(test_labels, test_rf_probs, best_f1_threshold(val_labels, val_rf_probs))
    ensemble_test_metrics = calculate_metrics(test_labels, test_ensemble_probs, val_threshold)

    print(f"\nBlend rule: ensemble_prob = w * dl_prob + (1 - w) * rf_prob, w={best_w:.3f} (chosen on val)")
    print(format_metrics("DL alone (test)", dl_test_metrics))
    print(format_metrics("RF alone (test)", rf_test_metrics))
    print(format_metrics("Ensemble (test)", ensemble_test_metrics))

    ensemble_weight_path = Path(args.ensemble_weight_path)
    ensemble_weight_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_weight_path.write_text(
        json.dumps(
            {
                "dl_checkpoint_path": args.dl_checkpoint_path,
                "rf_model_path": args.rf_model_path,
                "blend_weight_dl": best_w,
                "decision_threshold": val_threshold,
            },
            indent=2,
        )
        + "\n"
    )

    print("\nEnsemble weight path:", args.ensemble_weight_path)
    print("Ensemble: PASS")


if __name__ == "__main__":
    main()
