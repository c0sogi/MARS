import sys
import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.data_processor import get_dataloaders, process_data
from library.architecture import AsymmetricParallelNet
from library.trainer import Trainer


def main():
    # 1. Configuration & Setup
    # Override epochs to fit within the time limit (28 mins).
    # A100 can handle ~1 min/epoch for this data size.
    # We set to 15 epochs to allow convergence while staying under time limit.
    Config.EPOCHS = 15

    # Ensure reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    print(f"Initializing run with {Config.EPOCHS} epochs...")

    # 2. Data Loading
    # Load cached data if available, otherwise process from scratch
    # get_dataloaders calls process_data internally
    train_loader, val_loader, test_loader, test_ids, input_dim = get_dataloaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    model = AsymmetricParallelNet(input_dim=input_dim, num_classes=Config.NUM_CLASSES)

    # 4. Training
    trainer = Trainer(model)
    best_acc = trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # 5. Validation Metric
    # The trainer returns the best accuracy observed during training (on validation set)
    # We print it in the required format.
    print(f"Final Validation Metric: {best_acc}")

    # 6. Failure Analysis
    print("\nRunning Failure Analysis...")
    # We need the raw validation data and labels to compute correlations
    # process_data returns numpy arrays. Since load_cached_data=True, this is fast.
    _, _, val_X, val_y, _, _ = process_data(load_cached_data=True)

    # Get predictions on validation set
    val_preds = trainer.predict(val_loader)

    # Calculate error vector (1 if error, 0 if correct)
    errors = (val_preds != val_y).astype(int)

    # Calculate correlation between each feature and the error
    # val_X shape: (N, Features)
    n_features = val_X.shape[1]
    correlations = []

    for i in range(n_features):
        feature_vals = val_X[:, i]
        # Handle constant features to avoid warnings
        if np.std(feature_vals) == 0:
            corr = 0
        else:
            corr, _ = pearsonr(feature_vals, errors)
        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error:")
    for idx, corr in correlations[:5]:
        print(f"  Feature {idx}: Correlation = {corr:.4f}")

    # 7. Submission
    THRESHOLD = 0.9626291666666666

    if best_acc > THRESHOLD:
        print(f"\nValidation metric {best_acc} > {THRESHOLD}. Generating submission...")

        # Generate predictions
        test_preds = trainer.predict(test_loader)

        # Map back to original class labels (0-6 -> 1-7)
        test_preds_labels = test_preds + 1

        # Create submission DataFrame
        submission_df = pd.DataFrame({"Id": test_ids, "Cover_Type": test_preds_labels})

        # Save
        submission_path = Config.SUBMISSION_PATH
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

        # Verify first few rows
        print("Submission head:")
        print(submission_df.head())
    else:
        print(
            f"\nValidation metric {best_acc} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
