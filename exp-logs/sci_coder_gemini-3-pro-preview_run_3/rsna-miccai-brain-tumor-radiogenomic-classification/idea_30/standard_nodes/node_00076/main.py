import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import provided library functions and configurations
from library.config import (
    VAL_META_PATH,
    TRAIN_META_PATH,
    TEST_META_PATH,
    MODEL_SAVE_PATH,
    SEED,
    DEVICE,
)
from library.utils import seed_everything, get_device
from library.train import fit
from library.inference import predict_and_submit
from library.data_loader import get_dataloaders
from library.model import HRVANet


def main():
    # 1. Setup
    seed_everything(SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Training
    # We use the provided fit function which handles the training loop,
    # validation monitoring, and saving the best model.
    print("Starting training pipeline...")
    fit(load_cached_data=True)

    # 3. Validation & Failure Analysis
    print("\nStarting validation and failure analysis...")

    # Load the validation data loader
    # We pass the paths defined in config.
    # Note: fit() likely already cached the data, so loading should be fast.
    _, val_loader, _ = get_dataloaders(
        TRAIN_META_PATH, VAL_META_PATH, TEST_META_PATH, load_cached_data=True
    )

    if val_loader is None:
        print("Error: Validation loader is None. Cannot perform validation.")
        return

    # Load the best model
    model = HRVANet().to(device)
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            state_dict = torch.load(MODEL_SAVE_PATH, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded best model from {MODEL_SAVE_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print(f"Error: Model file not found at {MODEL_SAVE_PATH}")
        return

    # Inference on Validation Set
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            # targets are already tensors in the loader

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            all_preds.extend(probs.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute Metric
    if len(np.unique(all_targets)) > 1:
        val_auc = roc_auc_score(all_targets, all_preds)
    else:
        val_auc = 0.5
        print("Warning: Only one class present in validation set.")

    # REQUIRED OUTPUT: Print Final Validation Metric
    print(f"Final Validation Metric: {val_auc}")

    # Failure Analysis
    # We correlate error magnitude with input features (slice counts per modality)
    print("\nPerforming Failure Analysis...")

    # Load metadata to get features
    if os.path.exists(VAL_META_PATH):
        val_df = pd.read_parquet(VAL_META_PATH)

        # Ensure the dataframe aligns with predictions
        # The loader is sequential (shuffle=False), so order should match
        if len(val_df) == len(all_preds):
            # Calculate Error
            val_df["pred"] = all_preds
            val_df["target"] = all_targets
            val_df["error"] = np.abs(val_df["target"] - val_df["pred"])

            # Extract Meta-Features (Slice Counts)
            modalities = ["flair", "t1w", "t1wce", "t2w"]
            correlations = {}

            print("Correlation between Error Magnitude and Modality Slice Counts:")
            for mod in modalities:
                col_name = f"{mod}_paths"
                # Count number of files in the list
                val_df[f"{mod}_count"] = val_df[col_name].apply(
                    lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
                )

                # Calculate correlation
                corr = val_df["error"].corr(val_df[f"{mod}_count"])
                correlations[mod] = corr
                print(f" - {mod.upper()} Count: {corr}")
        else:
            print(
                f"Warning: Validation DataFrame length ({len(val_df)}) does not match predictions ({len(all_preds)}). Skipping detailed analysis."
            )
    else:
        print("Warning: Validation metadata file not found. Skipping failure analysis.")

    # 4. Submission
    # Threshold check
    THRESHOLD = 0.6978181818181817

    if val_auc > THRESHOLD:
        print(
            f"\nValidation metric ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(load_cached_data=True)
    else:
        print(
            f"\nValidation metric ({val_auc}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
