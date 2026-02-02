import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Import from the provided library files
from library.config import Config, seed_everything
from library.data import get_dataloaders
from library.model import AsymmetricEfficientNet
from library.train import run_training, validate, predict_and_submit


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    print("Initializing Asymmetric Grouped EfficientNet Pipeline...")

    # 2. Run Training
    # This function handles the full training loop, early stopping,
    # and saves the best model to Config.MODEL_SAVE_PATH.
    # It also performs an initial inference on the test set.
    trained_model = run_training(load_cached_data=True)

    # 3. Final Validation Assessment
    print("\n--- Final Validation Assessment ---")
    device = torch.device(Config.DEVICE)

    # Initialize model architecture
    model = AsymmetricEfficientNet()
    model = model.to(device)

    # Load the best saved weights to ensure we evaluate the optimal state
    best_model_path = Config.MODEL_SAVE_PATH
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model weights.")
    else:
        print("Warning: Best model weights not found. Using current model state.")

    model.eval()

    # Retrieve DataLoaders (using cache for speed)
    # We specifically need the val_loader for metric calculation and failure analysis
    _, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Compute Validation Metric
    criterion = torch.nn.BCEWithLogitsLoss()
    val_loss, val_auc = validate(model, val_loader, criterion, device)

    # Print the required metric string
    print(f"Final Validation Metric: {val_auc}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    all_preds = []
    all_labels = []

    # Collect predictions on validation set
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs)

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Create analysis DataFrame
    val_df = val_loader.dataset.df.copy()

    if len(val_df) == len(all_preds):
        val_df["pred"] = all_preds
        val_df["label"] = all_labels
        val_df["error"] = np.abs(val_df["pred"] - val_df["label"])

        # Add metadata features for correlation
        # 'anchor_idx' is a proxy for the Z-axis location of the tumor
        anchor_dict = val_loader.dataset.anchor_dict
        val_df["anchor_idx"] = val_df["BraTS21ID"].map(anchor_dict)

        # Calculate correlations
        # We check if error correlates with the tumor location (anchor_idx) or the class itself
        corr_cols = ["error", "anchor_idx", "label"]
        correlations = val_df[corr_cols].corr()["error"].drop("error")

        print("Correlation between Error Magnitude and Input Features:")
        print(correlations)

        print("\nTop 5 Worst Failures:")
        print(
            val_df.sort_values("error", ascending=False).head(5)[
                ["BraTS21ID", "label", "pred", "error"]
            ]
        )
    else:
        print(
            "Warning: Validation set size mismatch. Skipping detailed failure analysis."
        )

    # 5. Conditional Submission
    THRESHOLD = 0.6303636363636363
    submission_path = Config.SUBMISSION_PATH

    if val_auc > THRESHOLD:
        print(f"\nValidation metric ({val_auc}) > Threshold ({THRESHOLD}).")
        print("Generating final submission file...")
        # Regenerate submission to ensure it uses the explicitly loaded best model
        predict_and_submit(model, test_loader, device, submission_path)
    else:
        print(f"\nValidation metric ({val_auc}) <= Threshold ({THRESHOLD}).")
        print("Discarding submission.")
        # run_training generates a submission file by default; we must remove it if the metric is low
        if os.path.exists(submission_path):
            os.remove(submission_path)
            print("Submission file removed.")


if __name__ == "__main__":
    main()
