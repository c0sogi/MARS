import os
import sys
import pandas as pd
import torch
import numpy as np

# Import library modules
from library import config
from library import utils
from library import dataset
from library import model as model_lib
from library import train
from library import inference


def main():
    # 1. Setup
    utils.seed_everything(config.SEED)

    # 2. Training
    # Run a fast baseline training for 5 epochs to ensure completion within time limits.
    # The library function automatically saves the best model to config.OUTPUT_DIR/best_model.pth
    print("Starting training...")
    train.run_training(epochs=5)

    # 3. Validation and Failure Analysis
    print("Starting validation and failure analysis...")
    device = config.DEVICE

    # Load the best model weights
    model_path = os.path.join(config.OUTPUT_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model = model_lib.ConvNeXtUNet(
        backbone_name=config.BACKBONE,
        pretrained=config.PRETRAINED,
        in_channels=config.MODEL_INPUT_CHANNELS,
        num_classes=1,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Get Validation DataLoader
    val_loader = dataset.get_dataloader(
        stage="validation", batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS
    )

    # Load Validation Metadata for analysis
    val_meta = pd.read_csv(config.VALIDATION_METADATA_PATH)

    # Metrics Accumulators
    intersection_sum = 0.0
    union_sum = 0.0
    sample_dices = []

    # Inference Loop
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > config.THRESHOLD).float()

            # Global Dice Calculation components
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            intersection_sum += (preds_flat * masks_flat).sum().item()
            union_sum += preds_flat.sum().item() + masks_flat.sum().item()

            # Sample-level Dice for Failure Analysis
            # Shape: (B, 1, H, W) -> flatten to (B, -1)
            B = images.size(0)
            p_flat = preds.view(B, -1)
            m_flat = masks.view(B, -1)

            inter_s = (p_flat * m_flat).sum(dim=1)
            union_s = p_flat.sum(dim=1) + m_flat.sum(dim=1)

            # Dice formula per sample. If union is 0 (empty mask and empty pred), Dice is 1.0.
            dices = (2.0 * inter_s) / (union_s + 1e-6)
            dices[union_s == 0] = 1.0

            sample_dices.extend(dices.cpu().numpy())

    # Compute and Print Final Global Metric
    final_metric = (2.0 * intersection_sum) / union_sum if union_sum > 0 else 0.0
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Align predictions with metadata
    if len(sample_dices) == len(val_meta):
        val_meta["dice"] = sample_dices
        val_meta["error"] = 1.0 - val_meta["dice"]

        # Calculate correlations
        features = ["timestamp", "row_min", "col_min"]
        print("Failure Analysis (Correlation between Error and Features):")
        for feat in features:
            if feat in val_meta.columns:
                corr = val_meta["error"].corr(val_meta[feat])
                print(f"  {feat}: {corr}")
    else:
        print(
            f"Warning: Sample count mismatch (Preds: {len(sample_dices)}, Meta: {len(val_meta)}). Skipping correlation analysis."
        )

    # 4. Submission
    # Threshold check
    THRESHOLD = 0.5910660985501295

    if final_metric > THRESHOLD:
        print(
            f"Metric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        inference.make_submission(model_path, debug=False)
    else:
        print(
            f"Metric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
