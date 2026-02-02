import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, compute_auc
from library.data import get_dataloaders
from library.train import Trainer


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Override Config for full training (Cite solution_lesson_node_00122)
    Config.EPOCHS = 35
    Config.DEBUG_SAMPLE_SIZE = None
    Config.NUM_WORKERS = 8

    # Update submission path to match task requirement
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    Config.SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # --------------------------------------------------------------------------
    # 3. Model Training
    # --------------------------------------------------------------------------
    print("Initializing Trainer...")
    trainer = Trainer(device)

    print("Starting training...")
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # --------------------------------------------------------------------------
    # 4. Validation & Metric Calculation
    # --------------------------------------------------------------------------
    print("Performing final validation...")
    trainer.model.eval()

    val_preds = []
    val_targets = []
    val_cont_features = []

    # Collect predictions, targets, and features for metric and failure analysis
    with torch.no_grad():
        for batch in val_loader:
            cont_x = batch["cont_features"].to(device)
            cat_x = batch["cat_features"].to(device)
            targets = batch["target"].to(device).view(-1, 1)

            outputs = trainer.model(cont_x, cat_x)

            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())
            val_cont_features.append(cont_x.cpu().numpy())

    val_preds = np.concatenate(val_preds).flatten()
    val_targets = np.concatenate(val_targets).flatten()
    val_cont_features = np.concatenate(val_cont_features, axis=0)

    final_auc = compute_auc(val_targets, val_preds)

    # REQUIRED OUTPUT: Final Validation Metric
    print(f"Final Validation Metric: {final_auc}")

    # --------------------------------------------------------------------------
    # 5. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(val_targets - val_preds)

    # Calculate correlation between error and continuous features
    print("Correlation between Error Magnitude and Input Features:")
    feature_names = Config.CONT_FEATURES

    correlations = []
    for i, feature_name in enumerate(feature_names):
        # Ensure we don't go out of bounds if features were sliced/modified
        if i < val_cont_features.shape[1]:
            feat_values = val_cont_features[:, i]
            # Handle potential constant values to avoid warning
            if np.std(feat_values) == 0 or np.std(errors) == 0:
                corr = 0.0
            else:
                corr, _ = pearsonr(errors, feat_values)
            correlations.append((feature_name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, corr in correlations[:10]:
        print(f"{name}: {corr:.6f}")

    # --------------------------------------------------------------------------
    # 6. Conditional Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9972883264620234

    if final_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({final_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # The trainer.predict method handles loading the best model state
        # However, since we just trained and the model in memory is the last epoch state,
        # we should ensure we use the best model saved during training.
        # Trainer.predict loads from Config.MODEL_PATH automatically.

        submission_df = trainer.predict(test_loader)

        print(f"Saving submission to {Config.SUBMISSION_PATH}")
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission saved successfully.")
    else:
        print(
            f"\nValidation AUC ({final_auc}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
