import os
import torch
import numpy as np
import pandas as pd
import sys

# Append current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, optimize_threshold
from library.model import KCVRNet
from library.trainer import Trainer
from library.data_loader import get_dataloaders, get_test_loader


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    # get_dataloaders handles feature engineering and caching internally
    print("Loading training and validation data...")
    train_loader, val_loader = get_dataloaders()

    # 3. Model Initialization
    print("Initializing KCVR-Net model...")
    model = KCVRNet()

    # 4. Training
    # We limit epochs to 3 for a fast baseline execution as per requirements
    print("Starting training...")
    trainer = Trainer(model, device=device)
    trainer.fit(train_loader, val_loader, epochs=3)

    # 5. Validation & Failure Analysis
    print("Loading best model for validation...")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    print("Generating validation predictions...")
    all_preds = []
    all_targets = []
    all_kin_feats = []

    # Inference loop on validation set
    with torch.no_grad():
        for X_kin, X_vis, y in val_loader:
            X_kin = X_kin.to(device)
            X_vis = X_vis.to(device)

            # Forward pass
            logits = model(X_kin, X_vis)
            probs = torch.sigmoid(logits)

            # Store results
            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            # Store kinematic features for failure analysis (keep on CPU)
            all_kin_feats.append(X_kin.cpu().numpy())

    # Concatenate results
    all_preds = np.concatenate(all_preds).ravel()
    all_targets = np.concatenate(all_targets).ravel()
    all_kin_feats = np.concatenate(all_kin_feats, axis=0)

    # Optimize Threshold and Compute Final Metric
    best_thresh, best_mcc = optimize_threshold(all_targets, all_preds)
    print(f"Final Validation Metric: {best_mcc}")

    # Failure Analysis
    print("\nPerforming failure analysis...")
    # Calculate error magnitude
    errors = np.abs(all_targets - all_preds)

    # Calculate correlation between error and kinematic features
    # Center the data for covariance calculation
    errors_centered = errors - errors.mean()
    feats_centered = all_kin_feats - all_kin_feats.mean(axis=0)

    # Compute correlation
    numerator = np.dot(feats_centered.T, errors_centered)
    ss_errors = np.sum(errors_centered**2)
    ss_feats = np.sum(feats_centered**2, axis=0)
    denominator = np.sqrt(ss_errors * ss_feats)

    # Avoid division by zero
    correlations = numerator / (denominator + 1e-8)

    # Identify top 5 features associated with error
    top_indices = np.argsort(np.abs(correlations))[::-1][:5]
    print("Top 5 Input Features correlated with Error Magnitude:")
    for idx in top_indices:
        print(f"  Feature Index {idx}: Correlation = {correlations[idx]:.4f}")

    # 6. Submission Generation
    THRESHOLD_SCORE = 0.6634847318478787

    if best_mcc > THRESHOLD_SCORE:
        print(
            f"\nValidation MCC ({best_mcc}) > Threshold ({THRESHOLD_SCORE}). Generating submission..."
        )

        # Load Test Data
        test_loader, test_ids = get_test_loader()

        test_probs = []

        # Inference on Test Set
        with torch.no_grad():
            for X_kin, X_vis in test_loader:
                X_kin = X_kin.to(device)
                X_vis = X_vis.to(device)

                logits = model(X_kin, X_vis)
                probs = torch.sigmoid(logits)
                test_probs.append(probs.cpu().numpy())

        test_probs = np.concatenate(test_probs).ravel()

        # Apply Optimized Threshold
        predictions = (test_probs >= best_thresh).astype(int)

        # Create Submission DataFrame
        submission = pd.DataFrame({"contact_id": test_ids, "contact": predictions})

        # Save to ./submission/submission.csv
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nValidation MCC ({best_mcc}) did not meet the threshold ({THRESHOLD_SCORE}). Submission skipped."
        )


if __name__ == "__main__":
    main()
