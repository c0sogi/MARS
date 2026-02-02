import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import CameraTrapDataset
from library.train import run_training, generate_predictions


def main():
    # 1. Setup and Configuration
    Config.set_seed(Config.SEED)
    Config.make_dirs()
    device = torch.device(Config.DEVICE)

    # 2. Train Model
    # Cite solution_lesson_node_00002: Mitigate Resolution Bottlenecks in Fine-Grained Classification via Input Upscaling
    # Training on full dataset with higher resolution (448x448) and fine-tuning.
    print("Training model on full dataset...")
    model = run_training(
        sample_size=None,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        patience=2,
        load_cached_data=True,
    )

    # 3. Validation on Entire Hold-out Set
    print("Performing validation on full validation set...")
    # Load the full validation set (sample_size=None)
    val_dataset = CameraTrapDataset(split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate and print metric
    accuracy = np.mean(all_preds == all_labels)
    print(f"Final Validation Metric: {accuracy}")

    # 4. Failure Analysis
    print("Running failure analysis...")
    # Error magnitude: 1 if incorrect, 0 if correct
    errors = (all_preds != all_labels).astype(int)

    # Extract metadata features from the dataset DataFrame
    # Note: The DataLoader iterates sequentially (shuffle=False), so indices align with val_dataset.df
    df_val = val_dataset.df

    # Feature 1 & 2: Image Dimensions
    widths = df_val["width"].values
    heights = df_val["height"].values

    # Feature 3: MegaDetector Confidence
    # We need to look this up from the dataset's detection dictionary
    confidences = []
    for img_id in df_val["id"]:
        # detection info is a dict: {'bbox': [...], 'conf': float} or None
        det = val_dataset.detections.get(img_id)
        if det:
            confidences.append(det.get("conf", 0.0))
        else:
            confidences.append(0.0)
    confidences = np.array(confidences)

    # Calculate Correlations
    # We use Pearson correlation. Since Error is binary, this is equivalent to Point-Biserial.

    # Check for constant input to avoid warnings
    if np.std(errors) == 0:
        print("Model has 0% or 100% accuracy, cannot compute correlations.")
    else:
        corr_w, _ = pearsonr(errors, widths)
        corr_h, _ = pearsonr(errors, heights)
        corr_c, _ = pearsonr(errors, confidences)

        print(f"Correlation between Error and Width: {corr_w}")
        print(f"Correlation between Error and Height: {corr_h}")
        print(f"Correlation between Error and Confidence: {corr_c}")

    # 5. Generate Submission
    if accuracy > 0.6993879134457887:
        print("Generating submission for test set...")
        test_dataset = CameraTrapDataset(split="test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        generate_predictions(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"Validation accuracy {accuracy} did not meet threshold 0.6961. Skipping submission."
        )


if __name__ == "__main__":
    main()
