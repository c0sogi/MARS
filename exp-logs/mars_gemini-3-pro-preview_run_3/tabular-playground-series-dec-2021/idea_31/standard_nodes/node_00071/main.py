import sys
import os
import warnings
import numpy as np
import torch
from scipy.stats import pearsonr

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import config
from library import utils, data, model, train


def main():
    # 1. Setup and Reproducibility
    utils.seed_everything(config.train.seed)
    device = torch.device(config.train.device)

    # 2. Data Loading
    # We use the full dataset (debug=False) to ensure we meet the high accuracy threshold.
    # The A100 GPU is powerful enough to process the full dataset within the time limit.
    train_loader, val_loader, test_loader, test_ids = data.get_dataloaders(
        load_cached_data=True, debug=False
    )

    # 3. Model Initialization
    net = model.ParallelDCNResNet().to(device)

    # 4. Training
    # The Trainer handles the training loop, validation, scheduler, and early stopping.
    trainer = train.Trainer(net, train_loader, val_loader, device)

    # Execute training
    best_val_acc = trainer.fit()

    # Required Output: Final Validation Metric
    print(f"Final Validation Metric: {best_val_acc}")

    # 5. Failure Analysis
    print("Running failure analysis...")
    net.eval()

    all_errors = []
    all_features = []

    # Perform inference on validation set to gather error statistics
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = net(inputs)
            probs = torch.softmax(outputs, dim=1)

            # Calculate probability assigned to the true class
            # targets is (B,), view as (B, 1) for gather
            true_class_probs = probs.gather(1, targets.view(-1, 1)).squeeze()

            # Error magnitude: 1.0 - probability of the true class.
            # High error magnitude (near 1.0) implies the model missed the correct class with high confidence
            # or assigned very low probability to it.
            error_magnitude = 1.0 - true_class_probs

            all_errors.append(error_magnitude.cpu().numpy())
            all_features.append(inputs.cpu().numpy())

    # Concatenate all batches
    all_errors = np.concatenate(all_errors)
    all_features = np.concatenate(all_features)

    # Calculate Pearson correlation between error magnitude and each feature
    n_features = all_features.shape[1]
    correlations = []

    for i in range(n_features):
        feature_col = all_features[:, i]
        # Check for constant features to avoid warnings/NaNs
        if np.std(feature_col) > 1e-9:
            corr, _ = pearsonr(all_errors, feature_col)
            correlations.append((i, corr))
        else:
            correlations.append((i, 0.0))

    # Sort by absolute correlation to find most significant associations
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features correlated with Error Magnitude:")
    for idx, corr in correlations[:10]:
        print(f"Feature {idx}: {corr:.6f}")

    # 6. Submission Generation
    # Strict threshold check
    threshold = 0.9625222222222222

    if best_val_acc > threshold:
        print(
            f"Validation metric {best_val_acc} exceeds threshold {threshold}. Generating submission..."
        )
        predictions = trainer.predict(test_loader)
        utils.save_submission(test_ids, predictions, config.paths.submission_path)
    else:
        print(
            f"Validation metric {best_val_acc} does not exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
