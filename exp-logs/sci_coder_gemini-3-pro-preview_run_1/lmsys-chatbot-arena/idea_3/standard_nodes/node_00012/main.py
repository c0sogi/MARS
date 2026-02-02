import sys
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import get_dataloaders
from library.model import SiameseDebertaWithScalars
from library.train import Trainer
from library.inference import predict


def main():
    # 1. Setup and Configuration
    # Set seed for reproducibility
    seed_everything(Config.SEED)
    device = get_device()

    # Override Config for Fast Baseline Execution
    # Adjusted for ~16GB VRAM (T4) instead of A100
    # 1 Epoch is sufficient for a fast baseline on this dataset size (~41k).
    Config.NUM_EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 8
    Config.VALID_BATCH_SIZE = 16

    # 2. Training
    print("Initializing Trainer and starting training...")
    trainer = Trainer(debug=False)
    trainer.train()

    # 3. Validation and Metric Calculation
    print("Loading best model for validation analysis...")

    # Re-initialize model and load the best weights saved by Trainer
    model = SiameseDebertaWithScalars()
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Get validation dataloader
    _, val_loader, _ = get_dataloaders(debug=False)

    all_preds = []
    all_targets = []
    all_features = []

    print("Running validation inference...")
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            scalar_features = batch["scalar_features"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            logits = model(
                input_ids_a=input_ids_a,
                attention_mask_a=attention_mask_a,
                input_ids_b=input_ids_b,
                attention_mask_b=attention_mask_b,
                scalar_features=scalar_features,
            )

            # Get probabilities
            probs = torch.softmax(logits, dim=1)

            # Store results
            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            all_features.append(scalar_features.cpu().numpy())

    # Concatenate results
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)
    features = np.vstack(all_features)

    # Calculate Final Metric (Log Loss)
    # eps=1e-15 is standard for log_loss to avoid log(0)
    metric = log_loss(y_true, y_pred, eps=1e-15)

    # Print required metric line
    print(f"Final Validation Metric: {metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate per-sample loss (Cross Entropy)
    # Clip predictions for numerical stability in manual calculation
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
    per_sample_loss = -np.sum(y_true * np.log(y_pred_clipped), axis=1)

    print("Correlation between Error Magnitude and Scalar Features:")
    feature_names = Config.SCALAR_FEATURE_LIST

    correlations = []
    for i, name in enumerate(feature_names):
        feat_values = features[:, i]
        # Calculate Pearson correlation
        # Use numpy to avoid extra dependencies, though scipy is likely available
        if np.std(feat_values) > 1e-9:
            corr = np.corrcoef(per_sample_loss, feat_values)[0, 1]
        else:
            corr = 0.0
        correlations.append((name, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, corr in correlations:
        print(f"  {name}: {corr:.4f}")

    # 5. Submission Logic
    THRESHOLD = 1.0523970763522532

    if metric < THRESHOLD:
        print(
            f"\nValidation metric {metric} is below threshold {THRESHOLD}. Generating submission..."
        )
        # Run inference on test set
        predict(debug=False)
    else:
        print(
            f"\nValidation metric {metric} is NOT below threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
