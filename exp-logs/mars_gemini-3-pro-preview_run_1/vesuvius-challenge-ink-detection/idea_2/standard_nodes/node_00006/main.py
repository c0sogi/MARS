import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import InkDataset
from library.model import InkUNet
from library.train import train_one_epoch, validate
from library.utils import generate_submission, calculate_fbeta


def main():
    # --- 1. Setup ---
    Config.setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Fast baseline configuration
    # Limiting epochs to ensure completion within 2 hours while providing a meaningful baseline.
    # The dataset is small (455 train samples), so 10 epochs is very fast.
    EPOCHS = 10

    # --- 2. Data Loading ---
    # We load cached data if available for speed
    train_dataset = InkDataset(mode="train", load_cached_data=True)
    val_dataset = InkDataset(mode="val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Shuffle=False is crucial for aligning predictions with metadata during failure analysis
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- 3. Model Initialization ---
    model = InkUNet(z_dim=Config.Z_DIM).to(device)

    # Loss function with positive weight to handle class imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # --- 4. Training Loop ---
    best_score = -1.0
    best_thresh = 0.5
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate (returns loss, best threshold for this epoch, and score)
        val_loss, thresh, score = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Score: {score:.4f} | Thresh: {thresh:.2f}"
        )

        # Save best model
        if score > best_score:
            best_score = score
            best_thresh = thresh
            torch.save(model.state_dict(), best_model_path)
            print("  New best model saved.")

    # --- 5. Final Validation & Failure Analysis ---
    print("\n--- Starting Failure Analysis ---")

    # Load best model weights
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model not found, using current weights.")

    model.eval()

    # Containers for analysis
    val_errors = []
    val_means = []
    val_xs = []
    val_ys = []

    all_preds = []
    all_targets = []

    # Access metadata for correlation analysis
    val_df = val_dataset.df

    with torch.no_grad():
        batch_idx = 0
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Inference
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            # Store for global metric calculation
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            # --- Failure Analysis Metrics ---
            # Calculate Mean Absolute Error (MAE) per sample
            abs_diff = torch.abs(probs - targets)
            # Average over spatial dimensions (C, H, W) -> (B,)
            mae = abs_diff.mean(dim=(1, 2, 3)).cpu().numpy()

            # Calculate input volume mean intensity per sample -> (B,)
            vol_means = inputs.mean(dim=(1, 2, 3)).cpu().numpy()

            # Get corresponding metadata
            start_idx = batch_idx * Config.BATCH_SIZE
            end_idx = start_idx + inputs.size(0)

            # Ensure we don't go out of bounds (though loader handles this)
            current_batch_df = val_df.iloc[start_idx:end_idx]

            val_errors.extend(mae)
            val_means.extend(vol_means)
            val_xs.extend(current_batch_df["x"].values)
            val_ys.extend(current_batch_df["y"].values)

            batch_idx += 1

    # --- 6. Compute Final Metric ---
    # Flatten all predictions and targets
    y_pred_probs = np.concatenate([p.flatten() for p in all_preds])
    y_true = np.concatenate([t.flatten() for t in all_targets])

    # Apply the best threshold found during training
    y_pred_bin = y_pred_probs >= best_thresh

    final_metric = calculate_fbeta(y_pred_bin, y_true, beta=0.5)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # --- 7. Correlation Analysis ---
    df_analysis = pd.DataFrame(
        {
            "error_magnitude": val_errors,
            "mean_intensity": val_means,
            "x": val_xs,
            "y": val_ys,
        }
    )

    print("\nCorrelation between Error Magnitude and Input Features:")
    correlations = df_analysis.corr()["error_magnitude"].drop("error_magnitude")
    print(correlations)

    # --- 8. Submission ---
    THRESHOLD_SCORE = 0.41758
    if final_metric > THRESHOLD_SCORE:
        print(
            f"\nMetric ({final_metric}) > {THRESHOLD_SCORE}. Generating submission..."
        )
        generate_submission(model, device, threshold=best_thresh)
    else:
        print(f"\nMetric ({final_metric}) <= {THRESHOLD_SCORE}. Submission skipped.")


if __name__ == "__main__":
    main()
