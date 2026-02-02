import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.train import Trainer
from library.inference import predict_and_submit
from library.model import BreastCancerModel
from library.dataset import get_dataloaders
from library.utils import seed_everything, probabilistic_f1


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Override Config for a fast baseline execution
    # 2 epochs is sufficient to verify the pipeline and get a performance signal
    # within the 2-hour limit given the dataset size (~40k images).
    Config.EPOCHS = 2

    # Ensure working directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print(
        f"Configuration: Device={Config.DEVICE}, Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}"
    )

    # --------------------------------------------------------------------------
    # 2. Training
    # --------------------------------------------------------------------------
    print("\n=== Starting Training Phase ===")
    trainer = Trainer()
    trainer.fit()

    # --------------------------------------------------------------------------
    # 3. Validation & Metric Calculation
    # --------------------------------------------------------------------------
    print("\n=== Starting Validation Phase ===")

    # Load the best model saved during training
    device = torch.device(Config.DEVICE)
    model = BreastCancerModel(pretrained=False)
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if os.path.exists(checkpoint_path):
        print(f"Loading best model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(
            "WARNING: Checkpoint not found. Using current model state (likely suboptimal)."
        )

    model.to(device)
    model.eval()

    # Get Validation DataLoader
    # We use the existing helper but only need the val_loader
    _, val_loader, _ = get_dataloaders()

    # Calculate Analytical Correction Factor
    # This aligns the balanced training output (P=0.5) to the test prior (P~0.02)
    p_train = Config.P_TRAIN
    p_test = Config.P_TEST
    term_train = np.log(p_train / (1 - p_train))
    term_test = np.log(p_test / (1 - p_test))
    correction_factor = term_test - term_train

    print(f"Applying Analytical Correction Factor: {correction_factor:.4f}")

    all_probs = []
    all_labels = []

    # Inference Loop on Validation Set
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)

            # Apply Correction
            corrected_logits = logits + correction_factor
            probs = torch.sigmoid(corrected_logits)

            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    y_true = np.array(all_labels)
    y_pred = np.array(all_probs)

    # Compute Final Metric
    final_metric = probabilistic_f1(y_true, y_pred)

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # --------------------------------------------------------------------------
    # 4. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n=== Failure Analysis ===")

    # Load metadata to correlate errors with features
    df_val = pd.read_csv(Config.VAL_META_PATH)

    # Safety check for length alignment
    if len(df_val) != len(y_pred):
        print(
            f"Warning: Metadata rows ({len(df_val)}) != Prediction count ({len(y_pred)}). Truncating to match."
        )
        min_len = min(len(df_val), len(y_pred))
        df_val = df_val.iloc[:min_len].copy()
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
    else:
        df_val = df_val.copy()

    # Calculate Error Magnitude
    df_val["pred"] = y_pred
    df_val["target"] = y_true
    df_val["error"] = np.abs(df_val["target"] - df_val["pred"])

    # Features to analyze
    features_to_analyze = ["age", "machine_id", "density", "view"]

    print("Correlation between Error Magnitude and Features:")
    for feat in features_to_analyze:
        if feat in df_val.columns:
            # Prepare data: drop NaNs
            valid_mask = df_val[feat].notna() & df_val["error"].notna()

            if valid_mask.sum() < 2:
                print(f"  {feat}: Not enough valid data.")
                continue

            val_series = df_val.loc[valid_mask, feat]
            error_series = df_val.loc[valid_mask, "error"]

            # Encode categorical if necessary
            if val_series.dtype == "object":
                # Simple label encoding for correlation check
                val_series = val_series.astype("category").cat.codes

            # Calculate Pearson Correlation
            corr, _ = pearsonr(val_series, error_series)
            print(f"  Feature: {feat:<12} | Correlation: {corr:.4f}")

    # --------------------------------------------------------------------------
    # 5. Submission
    # --------------------------------------------------------------------------
    print("\n=== Submission Check ===")
    THRESHOLD = 0.06310755014419556

    if final_metric > THRESHOLD:
        print(
            f"Metric ({final_metric:.6f}) exceeds threshold ({THRESHOLD:.6f}). Generating submission..."
        )
        predict_and_submit(checkpoint_path=checkpoint_path)
    else:
        print(
            f"Metric ({final_metric:.6f}) does not exceed threshold ({THRESHOLD:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
