import os
import sys
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import provided library components
from library.config import Config
from library.utils import seed_everything
from library.dataset import load_metadata, get_transforms, PathologyDataset
from library.model import get_model
from library.trainer import ModelTrainer


def extract_features(image_path):
    """
    Extracts basic statistical features from an image file for failure analysis.
    Returns a dictionary of features or None if reading fails.
    """
    try:
        # Load image (OpenCV loads as BGR)
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Calculate features
        # 1. Color Channel Means
        mean_rgb = np.mean(img, axis=(0, 1))

        # 2. Grayscale stats (Brightness, Contrast)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)

        # 3. Sharpness (Variance of Laplacian)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()

        return {
            "red_mean": mean_rgb[0],
            "green_mean": mean_rgb[1],
            "blue_mean": mean_rgb[2],
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
        }
    except Exception:
        return None


def main():
    # --- 1. Initialization ---
    print("Initializing workflow...")
    seed_everything(Config.SEED)

    # Configuration for Full Training
    # Scaling to full dataset and 10 epochs. Cite solution_lesson_node_00003
    TRAIN_EPOCHS = 10

    # --- 2. Data Loading ---
    print("Loading metadata...")
    # Load metadata with caching enabled
    df_train = load_metadata("train", load_cached_data=True)
    df_val = load_metadata("val", load_cached_data=True)
    df_test = load_metadata("test", load_cached_data=True)

    # Using full training set as per iterative scaling lesson
    print(f"Using full training set: {len(df_train)} samples.")

    # Create Datasets
    # Note: Test set uses 'val' transforms (deterministic center crop)
    train_dataset = PathologyDataset(df_train, transform=get_transforms("train"))
    val_dataset = PathologyDataset(df_val, transform=get_transforms("val"))
    test_dataset = PathologyDataset(df_test, transform=get_transforms("val"))

    # Create DataLoaders
    # Pin memory enables faster data transfer to CUDA
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- 3. Model Training ---
    print("Setting up model and trainer...")
    model = get_model()
    trainer = ModelTrainer(model, device=Config.DEVICE)

    print(f"Starting training for {TRAIN_EPOCHS} epochs...")
    trainer.fit(train_loader, val_loader, epochs=TRAIN_EPOCHS)

    # --- 4. Validation Assessment ---
    print("Performing final validation assessment...")

    # Load best checkpoint
    if os.path.exists(Config.CHECKPOINT_PATH):
        print(f"Loading best model from {Config.CHECKPOINT_PATH}")
        trainer.model.load_state_dict(
            torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
        )

    trainer.model.eval()

    val_preds = []
    val_targets = []
    val_ids = []

    # Inference loop without gradients
    with torch.no_grad():
        for images, labels, ids in val_loader:
            images = images.to(Config.DEVICE)

            outputs = trainer.model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            val_preds.extend(probs)
            val_targets.extend(labels.numpy())
            val_ids.extend(ids)

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # Calculate and print required metric
    final_auc = roc_auc_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # --- 5. Failure Analysis ---
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(val_targets - val_preds)

    # Create a DataFrame for analysis
    analysis_df = pd.DataFrame(
        {"id": val_ids, "label": val_targets, "prob": val_preds, "error": errors}
    )

    # Merge with file paths from metadata
    analysis_df = analysis_df.merge(df_val[["id", "file_path"]], on="id", how="left")

    # Sample a subset for feature extraction to keep runtime low
    ANALYSIS_SAMPLE_SIZE = 2000
    if len(analysis_df) > ANALYSIS_SAMPLE_SIZE:
        print(
            f"Sampling {ANALYSIS_SAMPLE_SIZE} validation images for failure analysis..."
        )
        analysis_subset = analysis_df.sample(
            n=ANALYSIS_SAMPLE_SIZE, random_state=Config.SEED
        )
    else:
        analysis_subset = analysis_df

    feature_rows = []

    for idx, row in analysis_subset.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        feats = extract_features(full_path)

        if feats:
            feats["error"] = row["error"]
            feature_rows.append(feats)

    if feature_rows:
        feat_df = pd.DataFrame(feature_rows)
        # Calculate correlation
        correlations = (
            feat_df.corr()["error"].drop("error").sort_values(ascending=False)
        )
        print("Correlation between Model Error and Input Features:")
        print(correlations)
    else:
        print("Warning: No features extracted for failure analysis.")

    # --- 6. Submission Generation ---
    TARGET_METRIC = 0.9837547234728942
    if final_auc > TARGET_METRIC:
        print(
            f"\nMetric ({final_auc:.6f}) > Threshold ({TARGET_METRIC:.6f}). Generating submission..."
        )
        trainer.predict(test_loader)
    else:
        print(
            f"\nMetric ({final_auc:.6f}) <= Threshold ({TARGET_METRIC:.6f}). Skipping submission."
        )

    print("Workflow completed successfully.")


if __name__ == "__main__":
    main()
