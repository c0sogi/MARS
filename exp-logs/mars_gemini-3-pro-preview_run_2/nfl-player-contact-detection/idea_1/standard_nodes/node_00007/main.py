import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef

# Import provided libraries
from library import config
from library import training
from library import dataset
from library import model
from library import feature_engineering


def main():
    # 1. Setup and Configuration
    # Ensure reproducibility
    training.set_seed(config.SEED)

    # Override configuration for a fast baseline execution
    # Reducing epochs ensures the run completes well within the 2-hour limit
    config.EPOCHS = 15
    print(f"Configuration: EPOCHS set to {config.EPOCHS}")
    print(f"Device: {config.DEVICE}")

    # 2. Training
    print("\n>>> Starting Training Phase")
    # train() handles the training loop, early stopping, and returns the best threshold
    # It also saves the best model to config.MODEL_SAVE_PATH
    best_threshold = training.train(debug=False)
    print(f"Training finished. Optimal Threshold: {best_threshold}")

    # 3. Validation and Failure Analysis
    print("\n>>> Starting Validation & Failure Analysis")

    # Load Validation Dataset
    # We use load_cached_data=True to reuse features generated during training if available
    val_dataset = dataset.NFLContactDataset(split="validation", load_cached_data=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )

    # Initialize Model and Load Best Checkpoint
    input_dim = val_dataset.features.shape[1]
    net = model.KinematicMLP(input_dim).to(config.DEVICE)

    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {config.MODEL_SAVE_PATH}")

    print(f"Loading best model from {config.MODEL_SAVE_PATH}...")
    net.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    net.eval()

    # Run Inference on Validation Set
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(config.DEVICE)
            targets = batch["target"].to(config.DEVICE)

            outputs = net(features)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    # Compute Final Validation Metric
    binary_preds = (all_preds > best_threshold).astype(int)
    mcc = matthews_corrcoef(all_targets, binary_preds)
    print(f"Final Validation Metric: {mcc}")

    # Failure Analysis: Correlation of Error with Input Features
    print("\nPerforming Failure Analysis...")
    # Calculate error magnitude
    errors = np.abs(all_targets - all_preds)

    # Get feature names to make analysis readable
    # We reload the dataframe structure from the cache generator to get columns
    X_df, _, _ = feature_engineering.generate_features(
        split="validation", load_cached_data=True
    )
    feature_names = X_df.columns.tolist()

    # Compute correlations
    # We iterate through features in the numpy array to compute correlation with error
    feature_matrix = val_dataset.features
    correlations = []

    for i, name in enumerate(feature_names):
        # Extract feature column
        feat_col = feature_matrix[:, i]

        # Compute Pearson correlation
        # Handle constant features to avoid warnings/NaNs
        if np.std(feat_values := feat_col) > 1e-9:
            corr = np.corrcoef(feat_values, errors)[0, 1]
        else:
            corr = 0.0

        correlations.append((name, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features Correlated with Prediction Error:")
    for name, corr in correlations[:10]:
        print(f"  {name}: {corr:.6f}")

    # 4. Submission
    print("\n>>> Generating Submission")

    # Check against baseline performance
    BASELINE_MCC = 0.6148387392560759
    if mcc > BASELINE_MCC:
        print(
            f"Validation MCC ({mcc:.4f}) improved over baseline ({BASELINE_MCC}). Generating submission..."
        )
        # Predict on test set and save to CSV
        training.predict(best_threshold, debug=False)
    else:
        print(
            f"Validation MCC ({mcc:.4f}) did not improve over baseline ({BASELINE_MCC}). Skipping submission."
        )

    print(">>> Pipeline Execution Complete")


if __name__ == "__main__":
    main()
