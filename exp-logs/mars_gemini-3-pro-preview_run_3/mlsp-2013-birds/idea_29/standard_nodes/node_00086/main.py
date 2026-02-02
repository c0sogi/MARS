import os
import glob
import re
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.data import get_dataloaders
from library.models import get_cnn_model, SymbolicMLP
from library.training import run_training
from library.inference import run_inference


def parse_auc_from_filename(filepath):
    """Extracts AUC score from filename for sorting."""
    match = re.search(r"auc(\d+\.\d+)", filepath)
    if match:
        return float(match.group(1))
    return 0.0


def get_best_checkpoints(model_name, fold_idx, top_k=1):
    """Retrieves the best checkpoints for a specific model and fold."""
    if model_name == "mlp":
        ckpt_dir = os.path.join(Config.CHECKPOINT_DIR, "mlp")
    else:
        ckpt_dir = os.path.join(Config.CHECKPOINT_DIR, model_name)

    pattern = os.path.join(ckpt_dir, f"{model_name}_fold{fold_idx}_*.pth")
    files = glob.glob(pattern)

    # Sort by AUC descending
    files.sort(key=parse_auc_from_filename, reverse=True)
    return files[:top_k]


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Training
    # Running with 10 epochs to ensure completion within the 24-minute limit.
    # The dataset is small, so convergence is rapid.
    print("\n=== Starting Training Phase ===")
    run_training(debug_epochs=10, load_cached_data=True)

    # 3. Validation & Metric Calculation
    print("\n=== Starting Validation Phase ===")
    oof_preds = []
    oof_targets = []
    oof_ids = []

    # Iterate through all folds to generate Out-Of-Fold (OOF) predictions
    for fold_idx in range(Config.N_FOLDS):
        dataloaders = get_dataloaders(fold_idx, load_cached_data=True)
        val_cnn_loader = dataloaders["val_cnn"]
        val_mlp_loader = dataloaders["val_mlp"]

        # Accumulators for the current fold
        fold_probs_sum = {}
        fold_counts = {}
        fold_targets_map = {}

        def accumulate_predictions(loader, model, is_mlp=False):
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    if is_mlp:
                        inputs = batch["features"].to(device)
                    else:
                        inputs = batch["image"].to(device)

                    targets = batch["labels"].cpu().numpy()
                    rec_ids = batch["rec_id"].numpy()

                    outputs = model(inputs)
                    probs = torch.sigmoid(outputs).cpu().numpy()

                    for rid, p, t in zip(rec_ids, probs, targets):
                        rid = int(rid)
                        if rid not in fold_probs_sum:
                            fold_probs_sum[rid] = np.zeros(Config.NUM_CLASSES)
                            fold_counts[rid] = 0
                            fold_targets_map[rid] = t

                        fold_probs_sum[rid] += p
                        fold_counts[rid] += 1

        # A. CNN Ensemble (Top-3 Snapshots)
        for model_name in Config.CNN_MODELS:
            checkpoints = get_best_checkpoints(model_name, fold_idx, top_k=3)
            for ckpt in checkpoints:
                try:
                    model = get_cnn_model(model_name, pretrained=False).to(device)
                    model.load_state_dict(torch.load(ckpt, map_location=device))
                    accumulate_predictions(val_cnn_loader, model, is_mlp=False)
                    del model
                except Exception as e:
                    print(f"Error validating {ckpt}: {e}")
            torch.cuda.empty_cache()

        # B. MLP Ensemble (Top-1 Snapshot)
        checkpoints = get_best_checkpoints("mlp", fold_idx, top_k=1)
        for ckpt in checkpoints:
            try:
                model = SymbolicMLP().to(device)
                model.load_state_dict(torch.load(ckpt, map_location=device))
                accumulate_predictions(val_mlp_loader, model, is_mlp=True)
                del model
            except Exception as e:
                print(f"Error validating {ckpt}: {e}")
        torch.cuda.empty_cache()

        # Aggregate fold results
        for rid in fold_probs_sum:
            if fold_counts[rid] > 0:
                avg_prob = fold_probs_sum[rid] / fold_counts[rid]
                oof_preds.append(avg_prob)
                oof_targets.append(fold_targets_map[rid])
                oof_ids.append(rid)

    # Convert to numpy arrays
    oof_preds = np.array(oof_preds)
    oof_targets = np.array(oof_targets)

    # Calculate Final Metric
    final_metric = calculate_roc_auc(oof_targets, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    if len(oof_preds) > 0:
        # Calculate Mean Absolute Error per sample
        per_sample_error = np.mean(np.abs(oof_targets - oof_preds), axis=1)

        # Feature: Label Cardinality (Number of active species)
        label_counts = np.sum(oof_targets, axis=1)

        if len(per_sample_error) > 1:
            corr, _ = pearsonr(per_sample_error, label_counts)
            print(f"Correlation between Error and Label Count: {corr}")
        else:
            print("Insufficient samples for correlation analysis.")
    else:
        print("No predictions generated for failure analysis.")

    # 5. Submission Generation
    threshold = 0.993393498099723
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) exceeds threshold ({threshold}). Generating submission..."
        )
        run_inference(load_cached_data=True)
    else:
        print(
            f"\nMetric ({final_metric}) did not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
