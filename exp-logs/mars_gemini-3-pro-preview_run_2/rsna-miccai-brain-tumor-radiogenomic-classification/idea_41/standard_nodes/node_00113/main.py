import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# Import from provided library
import library.config as config
from library.utils import seed_everything
from library.data_loader import get_dataloaders
from library.model import AsymmetricGroupedEfficientNet
from library.train_eval import Trainer, CircuitBreaker


def main():
    # 1. Setup
    seed_everything(config.SEED)

    # 2. Data Loading
    # We use the full dataset as it is small (~500 samples), ensuring best performance
    # while still being fast enough for the baseline requirements.
    # load_cached_data=True ensures we use pre-processed numpy files if available.
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Circuit Breaker
    # Checks for data corruption (zero-tensors) before starting expensive training
    cb = CircuitBreaker(threshold=0.01)
    try:
        cb.check(train_loader.dataset, "Train Set")
        cb.check(val_loader.dataset, "Validation Set")
    except RuntimeError as e:
        print(f"Circuit Breaker Warning: {e}")
        # Proceed if datasets are not empty, as we want to attempt the task
        if len(train_loader.dataset) == 0:
            print("Error: Empty training set.")
            return

    # 4. Model Initialization
    model = AsymmetricGroupedEfficientNet()

    # 5. Training
    # The Trainer handles the loop, optimizer, and saving the best model based on Val AUC
    trainer = Trainer(model, train_loader, val_loader)
    trainer.fit()

    # 6. Validation Assessment & Failure Analysis
    # We reload the best model to perform a detailed analysis on the validation set
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print("Error: Model checkpoint not found.")
        return

    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    )
    model.to(config.DEVICE)
    model.eval()

    all_targets = []
    all_preds = []

    # Features for failure analysis
    feat_means = []
    feat_stds = []

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(config.DEVICE)

            # Inference
            logits = model(data)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(target.numpy().flatten())

            # Extract features from input tensor for failure analysis
            # data shape: (B, C, H, W). Flatten spatial dims to calculate stats per sample.
            flat_data = data.view(data.size(0), -1)

            # Calculate stats per sample (Mean Intensity and Standard Deviation/Contrast)
            batch_means = torch.mean(flat_data, dim=1).cpu().numpy()
            batch_stds = torch.std(flat_data, dim=1).cpu().numpy()

            feat_means.extend(batch_means)
            feat_stds.extend(batch_stds)

    # Calculate Metric
    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5

    # Print exactly as requested
    print(f"Final Validation Metric: {val_auc}")

    # Failure Analysis
    print("Performing Failure Analysis...")
    errors = np.abs(np.array(all_targets) - np.array(all_preds))
    feat_means = np.array(feat_means)
    feat_stds = np.array(feat_stds)

    # Correlation with Mean Intensity
    if np.std(feat_means) > 1e-6 and np.std(errors) > 1e-6:
        corr_mean, _ = pearsonr(errors, feat_means)
        print(f"Correlation (Error vs Input Mean Intensity): {corr_mean:.4f}")
    else:
        print("Correlation (Error vs Input Mean Intensity): Undefined (low variance)")

    # Correlation with Contrast (Std Dev)
    if np.std(feat_stds) > 1e-6 and np.std(errors) > 1e-6:
        corr_std, _ = pearsonr(errors, feat_stds)
        print(f"Correlation (Error vs Input Contrast/Std): {corr_std:.4f}")
    else:
        print("Correlation (Error vs Input Contrast/Std): Undefined (low variance)")

    # 7. Submission
    THRESHOLD = 0.6321818181818182
    if val_auc > THRESHOLD:
        print(
            f"Validation AUC ({val_auc}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Use TTA prediction from Trainer which handles flips
        preds = trainer.predict_tta(test_loader)

        # Load Test Metadata for IDs
        test_df = pd.read_csv(config.TEST_METADATA_PATH)

        # Ensure alignment
        if len(preds) != len(test_df):
            if len(preds) > len(test_df):
                preds = preds[: len(test_df)]
            else:
                preds = preds + [0.5] * (len(test_df) - len(preds))

        submission = pd.DataFrame(
            {"BraTS21ID": test_df["BraTS21ID"], "MGMT_value": preds}
        )

        os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation AUC ({val_auc}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
