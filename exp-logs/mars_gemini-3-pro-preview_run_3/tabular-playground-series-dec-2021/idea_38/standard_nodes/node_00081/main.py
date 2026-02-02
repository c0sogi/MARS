import sys
import os
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.model as model_lib
import library.train as train_lib


def main():
    # 1. Configuration Overrides for Fast Baseline
    # We limit epochs to ensure the run finishes well within the 2-hour limit.
    # We use the full dataset (MAX_TRAIN_SAMPLES = None) to ensure we have enough data
    # to hit the high accuracy threshold (>0.9626).
    config.EPOCHS = 15
    config.MAX_TRAIN_SAMPLES = None

    # 2. Setup
    utils.seed_everything(config.SEED)

    # 3. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids, input_dim = (
        data_loader.get_dataloaders(load_cached_data=True)
    )

    # Load raw validation data for failure analysis
    # We need the numpy arrays to correlate features with errors
    _, _, val_X, val_y, _, _ = data_loader.preprocess_data(load_cached_data=True)

    # Reconstruct feature names for analysis
    # We load a tiny sample of raw data to get the schema after engineering
    print("Reconstructing feature names...")
    temp_df = pd.read_parquet(config.TRAIN_PATH).iloc[:10]
    temp_df = data_loader.engineer_features(temp_df)
    cont_cols, bin_cols = data_loader.get_feature_groups(temp_df)
    feature_names = cont_cols + bin_cols

    # 4. Model Initialization
    print(f"Initializing model on {config.DEVICE}...")
    model = model_lib.WideAsymmetricDCNResNet(
        input_dim=input_dim, num_classes=config.NUM_CLASSES
    )
    model.to(config.DEVICE)

    # 5. Training
    print("Starting training...")
    trainer = train_lib.Trainer(model, train_loader, val_loader, config.DEVICE)
    trainer.fit(config.EPOCHS)

    # 6. Final Validation Metric
    print("Computing final validation metric...")
    val_loss, val_acc = trainer.validate()
    print(f"Final Validation Metric: {val_acc}")

    # 7. Failure Analysis
    print("\nRunning Failure Analysis...")
    model.eval()

    # Get probabilities for validation set
    # We process in batches to avoid OOM, though val_X fits in RAM, GPU memory is limited
    val_probs = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(config.DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            val_probs.append(probs.cpu().numpy())

    val_probs = np.concatenate(val_probs, axis=0)

    # Calculate Error Magnitude: 1.0 - Probability of the true class
    # val_y is 0-indexed
    rows = np.arange(len(val_y))
    true_class_probs = val_probs[rows, val_y]
    error_magnitude = 1.0 - true_class_probs

    # Calculate correlations
    print("Calculating correlations between features and error magnitude...")
    correlations = []
    for i, feature_name in enumerate(feature_names):
        if i < val_X.shape[1]:  # Safety check
            feature_values = val_X[:, i]
            # Handle potential constant columns (std=0) to avoid NaN correlation
            if np.std(feature_values) > 1e-9:
                corr = np.corrcoef(feature_values, error_magnitude)[0, 1]
                correlations.append((feature_name, corr))
            else:
                correlations.append((feature_name, 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 10 Features Correlated with Error:")
    print(f"{'Feature':<40} {'Correlation':<10}")
    print("-" * 50)
    for name, corr in correlations[:10]:
        print(f"{name:<40} {corr:.4f}")
    print("-" * 50)

    # 8. Conditional Submission
    THRESHOLD = 0.9626291666666666

    if val_acc > THRESHOLD:
        print(
            f"\nValidation metric ({val_acc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Generate predictions
        preds = trainer.predict(test_loader)

        # Post-processing: Add 1 to shift from 0-6 to 1-7
        final_preds = preds + 1

        # Save submission
        submission_df = pd.DataFrame(
            {config.ID_COL: test_ids, config.TARGET_COL: final_preds}
        )

        # Ensure ID is int
        submission_df[config.ID_COL] = submission_df[config.ID_COL].astype(int)

        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
        print(submission_df.head())

    else:
        print(
            f"\nValidation metric ({val_acc}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
