import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# 1. Configuration Adjustments for Fast Baseline
# We modify the Config class attributes before other modules use them.
from library.config import Config

# Optimize for speed and memory on A100 to ensure completion within 2 hours
Config.NUM_EPOCHS = 2
Config.BATCH_SIZE = 32
Config.IMG_SIZE = (512, 512)
Config.NUM_WORKERS = 12

# Import library modules after config adjustment
from library.utils import seed_everything, probabilistic_f1
from library.data import get_dataloaders
from library.model import HybridEfficientNet
from library.train import run_training


def main():
    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print("=== Starting Fast Baseline Pipeline ===")
    print(
        f"Configuration: Epochs={Config.NUM_EPOCHS}, Batch={Config.BATCH_SIZE}, Size={Config.IMG_SIZE}"
    )

    # 2. Training
    # run_training handles the training loop, validation monitoring, and saves the best model to Config.MODEL_PATH
    run_training(debug=False)

    # 3. Validation and Failure Analysis
    print("\n=== Starting Validation & Failure Analysis ===")

    device = Config.DEVICE

    # Re-load dataloaders to ensure we have access to the full validation set
    train_loader, val_loader, test_loader = get_dataloaders(debug=False)

    # Determine tabular dimension from a sample batch to correctly instantiate the model
    temp_img, temp_tab, _ = next(iter(val_loader))
    tabular_dim = temp_tab.shape[1]

    # Load the best model saved during training
    model = HybridEfficientNet(tabular_input_dim=tabular_dim)
    if not os.path.exists(Config.MODEL_PATH):
        print(f"Error: Model file not found at {Config.MODEL_PATH}")
        return

    print(f"Loading best model from {Config.MODEL_PATH}...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Inference on Validation Set
    val_probs = []
    val_targets = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for images, tab_features, targets in val_loader:
            images = images.to(device)
            tab_features = tab_features.to(device)

            logits = model(images, tab_features)
            probs = torch.sigmoid(logits).cpu().numpy()

            val_probs.extend(probs.flatten())
            val_targets.extend(targets.numpy().flatten())

    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)

    # Metric Calculation
    pf1 = probabilistic_f1(val_targets, val_probs)
    # REQUIRED: Print the final validation metric with full precision
    print(f"Final Validation Metric: {pf1}")

    # Failure Analysis
    # Load metadata to map back features
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Ensure alignment (DataLoader is sequential)
    if len(val_df) != len(val_probs):
        print(
            f"Warning: Validation dataframe length ({len(val_df)}) mismatch with predictions ({len(val_probs)}). Truncating to match."
        )
        val_df = val_df.iloc[: len(val_probs)]

    val_df["prediction"] = val_probs
    val_df["target"] = val_targets
    val_df["error"] = np.abs(val_df["target"] - val_df["prediction"])

    print("\n--- Correlation Analysis (Error vs Features) ---")

    # Correlation with Age
    if "age" in val_df.columns:
        # Drop NaNs for correlation calculation
        tmp = val_df.dropna(subset=["age", "error"])
        if len(tmp) > 0:
            corr, _ = pearsonr(tmp["error"], tmp["age"])
            print(f"Correlation (Error vs Age): {corr}")

    # Correlation with Density (Ordinal: A=1, B=2, C=3, D=4)
    if "density" in val_df.columns:
        density_map = {"A": 1, "B": 2, "C": 3, "D": 4}
        val_df["density_ord"] = val_df["density"].map(density_map)
        tmp = val_df.dropna(subset=["density_ord", "error"])
        if len(tmp) > 0:
            corr, _ = pearsonr(tmp["error"], tmp["density_ord"])
            print(f"Correlation (Error vs Density): {corr}")

    # 4. Submission
    THRESHOLD = 0.03510196892777368

    if pf1 > THRESHOLD:
        print(f"\nMetric ({pf1}) > Threshold ({THRESHOLD}). Generating submission...")

        test_probs = []
        test_ids = []

        print("Running inference on test set...")
        with torch.no_grad():
            for images, tab_features, sample_ids in test_loader:
                images = images.to(device)
                tab_features = tab_features.to(device)

                logits = model(images, tab_features)
                probs = torch.sigmoid(logits).cpu().numpy()

                test_probs.extend(probs.flatten())
                test_ids.extend(sample_ids)

        # Create DataFrame
        sub_df = pd.DataFrame({"prediction_id": test_ids, "cancer": test_probs})

        # Aggregate by prediction_id (Max probability across views)
        # This handles cases where a patient has multiple images (views) for the same breast.
        submission = sub_df.groupby("prediction_id", as_index=False)["cancer"].max()

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(submission.head())

    else:
        print(f"\nMetric ({pf1}) <= Threshold ({THRESHOLD}). Skipping submission.")


if __name__ == "__main__":
    main()
