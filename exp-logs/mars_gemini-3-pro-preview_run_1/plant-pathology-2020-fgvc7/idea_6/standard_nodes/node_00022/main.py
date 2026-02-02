import os
import sys
import pandas as pd
import numpy as np
import torch
import cv2
from sklearn.metrics import roc_auc_score

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library import train, inference, dataset, model, utils


def main():
    # ==========================================
    # 1. Configuration Override for Fast Baseline
    # ==========================================
    # Limit epochs to ensure execution within 2 hours while maintaining performance
    Config.EPOCHS = 35
    Config.DEBUG = False  # Use full dataset, but fewer epochs

    # Ensure deterministic behavior
    utils.seed_everything(Config.SEED)

    print(f"Configuration: EPOCHS={Config.EPOCHS}, DEBUG={Config.DEBUG}")

    # ==========================================
    # 2. Training Phase
    # ==========================================
    print("\n==== Starting Training Phase ====")
    # run_training handles directory setup, logging, and 5-fold CV
    train.run_training(debug=Config.DEBUG)

    # ==========================================
    # 3. Validation & Metric Calculation
    # ==========================================
    print("\n==== Starting Validation Phase ====")

    # Load Validation Metadata
    if not os.path.exists(Config.VAL_METADATA_PATH):
        print(f"Error: Validation metadata not found at {Config.VAL_METADATA_PATH}")
        return

    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Prepare DataLoader
    # We use 'valid' transforms (resize + normalize only)
    val_dataset = dataset.AppleDataset(
        val_df, transform=dataset.get_transforms("valid"), data_root=Config.INPUT_DIR
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Trained Models
    models = []
    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"resnet34_fold_{fold}.pth")
        if os.path.exists(model_path):
            m = model.AppleResNet()
            m.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
            m.to(Config.DEVICE)
            m.eval()
            models.append(m)
        else:
            print(f"Warning: Model for fold {fold} not found.")

    if not models:
        print("Error: No models available for validation.")
        return

    # Inference Loop
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(Config.DEVICE)
            targets = batch["target"].numpy()

            # Ensemble Prediction
            batch_probs = []
            for m in models:
                logits = m(images)
                probs = torch.softmax(logits, dim=1)
                batch_probs.append(probs.cpu().numpy())

            # Average across models
            avg_probs = np.mean(batch_probs, axis=0)

            all_preds.append(avg_probs)
            all_targets.append(targets)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Metric
    # Mean column-wise ROC AUC is equivalent to macro-averaged ROC AUC
    try:
        val_auc = roc_auc_score(all_targets, all_preds, average="macro")
    except Exception as e:
        print(f"Error calculating ROC AUC: {e}")
        val_auc = 0.0

    print(f"Final Validation Metric: {val_auc}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n==== Starting Failure Analysis ====")

    # Calculate error per sample: Mean Absolute Error across classes
    # Shape: (N_Samples,)
    sample_errors = np.abs(all_targets - all_preds).mean(axis=1)

    # Extract Image Features
    # We iterate through the validation dataframe to read images and get stats
    widths = []
    heights = []
    intensities = []

    print("Extracting image features for correlation analysis...")
    for idx, row in val_df.iterrows():
        path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img = cv2.imread(path)

        if img is not None:
            h, w, c = img.shape
            # Calculate mean intensity (normalized)
            # cv2 is BGR, but mean is same
            intensity = img.mean() / 255.0

            widths.append(w)
            heights.append(h)
            intensities.append(intensity)
        else:
            # Fallback
            widths.append(0)
            heights.append(0)
            intensities.append(0)

    # Create DataFrame for correlation
    analysis_df = pd.DataFrame(
        {
            "error": sample_errors,
            "width": widths,
            "height": heights,
            "intensity": intensities,
        }
    )

    # Calculate Correlations
    corr_width = analysis_df["error"].corr(analysis_df["width"])
    corr_height = analysis_df["error"].corr(analysis_df["height"])
    corr_intensity = analysis_df["error"].corr(analysis_df["intensity"])

    print("Correlation between Model Error and Input Features:")
    print(f"  Width: {corr_width:.6f}")
    print(f"  Height: {corr_height:.6f}")
    print(f"  Intensity: {corr_intensity:.6f}")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    THRESHOLD = 0.9871488489626378

    if val_auc > THRESHOLD:
        print(f"\nValidation Metric ({val_auc}) > Threshold ({THRESHOLD}).")
        print("Generating submission file...")
        inference.predict(debug=False)
    else:
        print(f"\nValidation Metric ({val_auc}) <= Threshold ({THRESHOLD}).")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
