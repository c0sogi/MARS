import os
import sys
import cv2
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Monkey-patch tqdm to disable progress bars before importing library modules
import tqdm.auto


def no_op_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.auto.tqdm = no_op_tqdm

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, compute_qwk, ordinal_decode
from library.data import get_dataloaders
from library.model import OrdinalEfficientNet
from library.trainer import Trainer


def analyze_failures(val_df, y_true, y_pred):
    """
    Performs failure analysis by correlating error magnitude with image meta-features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate error magnitude
    errors = np.abs(np.array(y_true) - np.array(y_pred))

    # Extract meta-features for validation set
    widths = []
    heights = []
    intensities = []
    file_sizes = []

    print("Extracting meta-features from validation set...")
    for _, row in val_df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            # File size
            if os.path.exists(full_path):
                file_sizes.append(os.path.getsize(full_path))

                # Image stats
                img = cv2.imread(full_path)
                if img is not None:
                    h, w, _ = img.shape
                    widths.append(w)
                    heights.append(h)
                    intensities.append(img.mean())
                else:
                    # Fallback for corrupt images
                    widths.append(0)
                    heights.append(0)
                    intensities.append(0)
            else:
                file_sizes.append(0)
                widths.append(0)
                heights.append(0)
                intensities.append(0)

        except Exception:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)
            intensities.append(0)

    # Create DataFrame for correlation
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "width": widths,
            "height": heights,
            "mean_intensity": intensities,
            "file_size": file_sizes,
        }
    )

    # Calculate correlations
    features = ["width", "height", "mean_intensity", "file_size"]
    print("Correlation between Error Magnitude and Input Features:")
    for feat in features:
        if analysis_df[feat].std() > 0:  # Avoid division by zero
            corr, _ = pearsonr(analysis_df[feat], analysis_df["error"])
            print(f"{feat}: {corr:.4f}")
        else:
            print(f"{feat}: NaN (No variance)")


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)

    print("Starting execution...")

    # 2. Data Loading
    # Using default batch size and workers from Config
    dataloaders = get_dataloaders()

    # 3. Model Initialization
    model = OrdinalEfficientNet(pretrained=Config.PRETRAINED)

    # 4. Training
    trainer = Trainer(model)
    trainer.fit(dataloaders["train"], dataloaders["val"], epochs=Config.EPOCHS)

    # 5. Validation Assessment
    print("\nRunning Validation Assessment...")
    # Load best model
    trainer.load_checkpoint(Config.MODEL_SAVE_PATH)

    # We need predictions and true labels for the full validation set
    # The trainer.evaluate method returns loss and QWK, but we need raw preds for failure analysis.
    # We will manually iterate or modify behavior. Since we cannot modify library, we replicate inference logic.

    model.eval()
    val_preds = []
    val_targets = []

    # Use the validation loader
    val_loader = dataloaders["val"]

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(Config.DEVICE)

            # Forward
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Decode
            preds = ordinal_decode(probs)

            # Get true labels from ordinal vectors
            true_labels = torch.sum(targets, dim=1).cpu().numpy().astype(int)

            if isinstance(preds, int):
                val_preds.append(preds)
            else:
                val_preds.extend(preds)

            val_targets.extend(true_labels)

    # Compute Metric
    qwk = compute_qwk(val_targets, val_preds)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {qwk}")

    # 6. Failure Analysis
    # Load validation metadata to link images to errors
    df_val = pd.read_csv(Config.VAL_META_PATH)
    analyze_failures(df_val, val_targets, val_preds)

    # 7. Submission
    if qwk > 0.9194950903896975:
        print("\nGenerating Submission...")
        submission_df = trainer.predict(dataloaders["test"])

        # Ensure submission directory exists (handled in Config, but double check)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

        # Print head for verification
        print(submission_df.head())
    else:
        print(
            f"\nSkipping submission generation. Validation QWK ({qwk}) did not meet threshold."
        )


if __name__ == "__main__":
    main()
