import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# Import from the provided library files
from library.config import (
    TRAIN_METADATA,
    VAL_METADATA,
    TEST_METADATA,
    MODEL_SAVE_PATH,
    DEVICE,
    INPUT_DIR,
    ROI_CACHE_PATH,
)
from library.utils import seed_everything, load_checkpoint
from library.data_loader import get_dataloader, get_anchor_ratios
from library.engine import train_model, generate_submission
from library.model import GroupedEfficientNet


def main():
    # 1. Reproducibility
    seed_everything()

    # 2. Load Metadata
    if not os.path.exists(TRAIN_METADATA) or not os.path.exists(VAL_METADATA):
        print(
            "Metadata files not found. Please ensure metadata generation was successful."
        )
        return

    train_df = pd.read_csv(TRAIN_METADATA)
    val_df = pd.read_csv(VAL_METADATA)
    test_df = pd.read_csv(TEST_METADATA)

    # 3. Prepare DataLoaders
    # We use the full dataset as it is small (~500 samples), so subsampling is not necessary for speed
    # and would harm performance.
    train_loader = get_dataloader(train_df, phase="train", load_cached_data=True)
    val_loader = get_dataloader(val_df, phase="val", load_cached_data=True)

    # 4. Train Model
    # This function handles the training loop, validation, early stopping, and saving the best model.
    print("Starting training...")
    train_model(train_loader, val_loader)

    # 5. Final Validation Assessment
    print("\n--- Final Validation Assessment ---")

    # Load the best model checkpoint to ensure we evaluate the optimal state
    model = GroupedEfficientNet(pretrained=False).to(DEVICE)
    try:
        load_checkpoint(MODEL_SAVE_PATH, model, device=DEVICE)
    except FileNotFoundError:
        print("Model checkpoint not found. Training may have failed.")
        return

    model.eval()

    val_targets = []
    val_preds = []

    # Run inference on validation set
    # Note: val_loader is not shuffled (phase='val'), so order matches val_df
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            val_preds.extend(probs)
            val_targets.extend(targets.numpy().flatten())

    val_targets = np.array(val_targets)
    val_preds = np.array(val_preds)

    # Calculate Metric
    try:
        final_auc = roc_auc_score(val_targets, val_preds)
    except ValueError:
        final_auc = 0.5

    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Calculate Error Magnitude
    errors = np.abs(val_targets - val_preds)

    # Extract features for correlation analysis
    # Feature 1: Anchor Ratio (cached)
    anchor_cache = get_anchor_ratios(val_df, load_cached_data=True)

    # Feature 2: Slice Count (proxy for brain volume/scan resolution)
    slice_counts = []
    anchor_ratios_list = []

    for _, row in val_df.iterrows():
        # Get Anchor Ratio
        ratio = anchor_cache.get(row["BraTS21ID"], 0.5)
        anchor_ratios_list.append(ratio)

        # Get Slice Count (FLAIR)
        flair_path = os.path.join(INPUT_DIR, row["path_FLAIR"])
        if os.path.exists(flair_path):
            # Quick count of DICOM files
            cnt = len(
                [
                    f
                    for f in os.listdir(flair_path)
                    if f.endswith(".dcm") or f.startswith("Image-")
                ]
            )
            slice_counts.append(cnt)
        else:
            slice_counts.append(0)

    # Create Analysis DataFrame
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "anchor_ratio": anchor_ratios_list,
            "slice_count": slice_counts,
        }
    )

    # Compute Correlations
    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 7. Submission
    threshold = 0.6254545454545455
    if final_auc > threshold:
        print(
            f"\nValidation AUC ({final_auc}) exceeds threshold ({threshold}). Generating submission..."
        )
        generate_submission(test_df)
    else:
        print(
            f"\nValidation AUC ({final_auc}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
