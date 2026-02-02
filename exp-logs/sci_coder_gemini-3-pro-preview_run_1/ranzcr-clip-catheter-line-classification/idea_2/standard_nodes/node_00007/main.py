import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from library.config import Config
from library.trainer import Trainer
from library.inference import predict
from library.dataset import CatheterDataset, get_transforms
from library.model import CatheterModel
from library.utils import seed_everything, get_device, calculate_metric


def main():
    # ==========================================
    # 1. Configuration for Fast Baseline
    # ==========================================
    # Override Config for a fast execution within time limits
    Config.EPOCHS = 5
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 6000  # Train on ~28% of data for speed
    Config.TRAIN_BATCH_SIZE = 8  # Conservative batch size for 768x768
    Config.VALID_BATCH_SIZE = 16

    seed_everything(Config.SEED)
    device = get_device()

    print("Configuration:")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Debug Mode: {Config.DEBUG}")
    print(f"  Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"  Device: {device}")

    # ==========================================
    # 2. Training
    # ==========================================
    print("\n=== Starting Training ===")
    trainer = Trainer(debug=Config.DEBUG)
    trainer.fit(epochs=Config.EPOCHS)

    # ==========================================
    # 3. Full Validation
    # ==========================================
    print("\n=== Starting Full Validation ===")
    # Load full validation metadata
    val_df = pd.read_csv(Config.VAL_METADATA)

    # Initialize validation dataset (force full dataset by debug=False)
    val_dataset = CatheterDataset(
        val_df, transforms=get_transforms("valid"), mode="valid", debug=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load best model weights
    model = CatheterModel(model_name=Config.MODEL_NAME, pretrained=False)
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if os.path.exists(best_model_path):
        print(f"Loading best model weights from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model weights not found. Using current model state.")
        model = trainer.model

    model.to(device)
    model.eval()

    preds_list = []
    targets_list = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)

            with autocast(enabled=Config.USE_AMP):
                logits = model(images)
                probs = torch.sigmoid(logits)

            preds_list.append(probs.cpu().numpy())
            targets_list.append(labels.numpy())

    y_pred = np.concatenate(preds_list, axis=0)
    y_true = np.concatenate(targets_list, axis=0)

    final_metric = calculate_metric(y_true, y_pred)
    # Print exactly as required
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n=== Performing Failure Analysis ===")
    # Calculate Mean Absolute Error per sample
    sample_mae = np.mean(np.abs(y_true - y_pred), axis=1)

    # Extract image features
    print("Extracting image features...")
    widths = []
    heights = []
    intensities = []

    for idx, row in val_df.iterrows():
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        # Read grayscale for speed
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            h, w = img.shape
            widths.append(w)
            heights.append(h)
            intensities.append(img.mean())
        else:
            widths.append(np.nan)
            heights.append(np.nan)
            intensities.append(np.nan)

    analysis_df = pd.DataFrame(
        {
            "error": sample_mae,
            "width": widths,
            "height": heights,
            "intensity": intensities,
        }
    )

    analysis_df["aspect_ratio"] = analysis_df["width"] / analysis_df["height"]
    analysis_df = analysis_df.dropna()

    # Calculate correlations
    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # ==========================================
    # 5. Submission
    # ==========================================
    print("\n=== Submission Check ===")
    THRESHOLD = 0.9122773173141027

    if final_metric > THRESHOLD:
        print(f"Metric {final_metric} > {THRESHOLD}. Generating submission...")
        # debug=False ensures we predict on the full test set
        predict(debug=False)
    else:
        print(f"Metric {final_metric} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
