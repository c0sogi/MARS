import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library import config
from library import utils
from library import model
from library import dataset
from library import train
from library import inference


def main():
    # 1. Set Random Seeds for Reproducibility
    train.set_seed(config.SEED)

    # 2. Train the Model
    # We use the provided Trainer class.
    # The dataset size is small (412 patches), so 30 epochs is very fast (~10-15 mins).
    print("Initializing Trainer...")
    trainer = train.Trainer()

    print("Starting Training Loop...")
    # fit() handles loading data, training, and saving the best model to config.WORKING_DIR
    trainer.fit(epochs=config.NUM_EPOCHS)

    # 3. Validation & Failure Analysis
    print("Starting Failure Analysis on Validation Set...")

    # Load Validation Metadata
    val_csv_path = os.path.join(config.METADATA_DIR, "validation.csv")
    if not os.path.exists(val_csv_path):
        print(f"Error: Validation metadata not found at {val_csv_path}")
        return

    val_df = pd.read_csv(val_csv_path)

    # Create Validation Dataset and Loader
    # We use the same transforms as validation in training (ToTensorV2)
    val_dataset = dataset.InkDataset(
        val_df, mode="val", transforms=dataset.get_transforms("val")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load the Best Model
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        print("No best model found. Training likely did not exceed the save threshold.")
        # If no model saved, we cannot proceed to submission or analysis
        return

    device = config.DEVICE
    # Initialize model structure (pretrained=False as we load weights)
    net = model.InkSegFormer(pretrained=False).to(device)
    state_dict = torch.load(best_model_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()

    # Containers for metrics and analysis
    all_preds = []
    all_labels = []

    feature_means = []
    error_magnitudes = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)  # (B, 3, H, W)
            labels = batch["label"].to(device)  # (B, 1, H, W)

            # Inference
            outputs = net(images)
            probs = torch.sigmoid(outputs)

            # Store for global F0.5 calculation
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())

            # --- Failure Analysis Data Collection ---
            # Feature: Mean Pixel Intensity of the input patch
            # (B, 3, H, W) -> Mean over (1, 2, 3) -> (B,)
            batch_means = images.mean(dim=(1, 2, 3)).cpu().numpy()

            # Error: Mean Absolute Error (MAE) for the patch
            # (B, 1, H, W) -> Abs Diff -> Mean over (1, 2, 3) -> (B,)
            batch_errors = torch.abs(probs - labels).mean(dim=(1, 2, 3)).cpu().numpy()

            feature_means.extend(batch_means)
            error_magnitudes.extend(batch_errors)

    # Calculate Final Validation Metric (F0.5)
    full_preds = torch.cat(all_preds)
    full_labels = torch.cat(all_labels)

    # Use the provided utility function
    val_f05 = utils.fbeta_score(full_preds, full_labels, beta=0.5)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {val_f05}")

    # Calculate Correlation
    if len(feature_means) > 1:
        # Using numpy for correlation
        correlation_matrix = np.corrcoef(feature_means, error_magnitudes)
        correlation = correlation_matrix[0, 1]
        print(
            f"Failure Analysis: Correlation between Input Intensity and Error Magnitude: {correlation:.4f}"
        )
    else:
        print("Failure Analysis: Insufficient data for correlation.")

    # 4. Conditional Submission
    # Threshold defined in the task description
    SUBMISSION_THRESHOLD = 0.597622633

    if val_f05 > SUBMISSION_THRESHOLD:
        print(
            f"Validation metric ({val_f05}) exceeds threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )
        # Call the provided inference runner
        inference.run_inference()
    else:
        print(
            f"Validation metric ({val_f05}) does NOT exceed threshold ({SUBMISSION_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
