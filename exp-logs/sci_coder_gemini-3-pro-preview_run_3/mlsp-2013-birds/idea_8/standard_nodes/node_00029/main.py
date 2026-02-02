import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.data import get_dataloaders
from library.model import MultiViewResNet
from library.train import run_fold


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Training Loop (5 Folds)
    print("Starting Training...")
    for fold_idx in range(Config.NUM_FOLDS):
        # run_fold handles training, validation, and saving the best checkpoint
        run_fold(fold_idx, debug=Config.DEBUG)

    # 3. Validation / OOF Inference & Failure Analysis
    print("\nStarting OOF Inference and Failure Analysis...")
    oof_preds = []
    oof_targets = []
    oof_errors = []
    oof_num_labels = []

    # Iterate through folds to generate OOF predictions
    for fold_idx in range(Config.NUM_FOLDS):
        print(f"Evaluating Fold {fold_idx}...")

        # Get dataloaders for this fold (we only need val_loader here)
        _, val_loader, _ = get_dataloaders(
            fold_idx, load_cached_data=True, debug=Config.DEBUG
        )

        # Load the best model for this fold
        model = MultiViewResNet()
        model.to(device)
        checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, f"fold_{fold_idx}_best.pth"
        )

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint for fold {fold_idx} not found. Skipping.")
            continue

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        fold_preds = []
        fold_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                logits = model(images)
                probs = torch.sigmoid(logits)

                # Store predictions and targets
                fold_preds.append(probs.cpu().numpy())
                fold_targets.append(labels.cpu().numpy())

                # Failure Analysis: Compute per-sample error
                # Error = Mean Absolute Error across all classes for the sample
                batch_errors = torch.abs(probs - labels).mean(dim=1).cpu().numpy()
                oof_errors.extend(batch_errors)

                # Feature: Number of labels (complexity)
                batch_num_labels = labels.sum(dim=1).cpu().numpy()
                oof_num_labels.extend(batch_num_labels)

        if fold_preds:
            oof_preds.append(np.concatenate(fold_preds, axis=0))
            oof_targets.append(np.concatenate(fold_targets, axis=0))

        # Cleanup
        del model
        torch.cuda.empty_cache()

    # Calculate Final Metrics
    if oof_preds:
        y_pred_oof = np.concatenate(oof_preds, axis=0)
        y_true_oof = np.concatenate(oof_targets, axis=0)

        final_metric = calculate_roc_auc(y_true_oof, y_pred_oof)
        print(f"Final Validation Metric: {final_metric}")

        # Failure Analysis Correlation
        if len(oof_errors) > 1:
            errors = np.array(oof_errors)
            label_counts = np.array(oof_num_labels)
            corr, _ = pearsonr(errors, label_counts)
            print(f"Correlation between Error Magnitude and Number of Labels: {corr}")
        else:
            print("Insufficient data for failure analysis correlation.")
    else:
        print("No OOF predictions generated.")
        final_metric = 0.0

    # 4. Submission
    THRESHOLD = 0.9072993371210134

    if final_metric > THRESHOLD:
        print("\nMetric passed threshold. Generating Submission...")

        # Load Test Data (from fold 0, test set is constant)
        _, _, test_loader = get_dataloaders(
            0, load_cached_data=True, debug=Config.DEBUG
        )

        # Retrieve rec_ids for mapping
        test_df = test_loader.dataset.df
        test_rec_ids = test_df["rec_id"].values

        # Ensemble Inference
        ensemble_probs = np.zeros((len(test_df), Config.NUM_CLASSES))
        models_found = 0

        for fold_idx in range(Config.NUM_FOLDS):
            checkpoint_path = os.path.join(
                Config.CHECKPOINT_DIR, f"fold_{fold_idx}_best.pth"
            )
            if not os.path.exists(checkpoint_path):
                continue

            print(f"Inference with model fold {fold_idx}...")
            model = MultiViewResNet()
            model.to(device)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()

            fold_test_preds = []
            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(device)
                    logits = model(images)
                    probs = torch.sigmoid(logits)
                    fold_test_preds.append(probs.cpu().numpy())

            if fold_test_preds:
                ensemble_probs += np.concatenate(fold_test_preds, axis=0)
                models_found += 1

            del model
            torch.cuda.empty_cache()

        if models_found > 0:
            ensemble_probs /= models_found

            # Format Submission
            submission_rows = []
            for i, rec_id in enumerate(test_rec_ids):
                probs = ensemble_probs[i]
                for species_id in range(Config.NUM_CLASSES):
                    # Id format: rec_id * 100 + species_id
                    row_id = int(rec_id * 100 + species_id)
                    row_prob = probs[species_id]
                    submission_rows.append({"Id": row_id, "Probability": row_prob})

            submission_df = pd.DataFrame(submission_rows)
            submission_df = submission_df.sort_values("Id")

            # Ensure output directory exists
            os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
            submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

            # Also save to ./submission/submission.csv as requested by prompt specific instruction
            alt_submission_path = "./submission/submission.csv"
            os.makedirs(os.path.dirname(alt_submission_path), exist_ok=True)
            submission_df.to_csv(alt_submission_path, index=False)

            print(
                f"Submission saved to {Config.SUBMISSION_PATH} and {alt_submission_path}"
            )
        else:
            print("No models found for inference.")
    else:
        print(
            f"Metric {final_metric} did not pass threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
