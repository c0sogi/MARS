import os
import torch
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Import configuration and utilities from the provided library
from library.config import (
    VAL_META_PATH,
    MODEL_PATH,
    DEVICE,
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_EPOCHS,
)
from library.utils import seed_everything
from library.data import get_dataset
from library.model import SSBHDNetwork
from library.train import run_training
from library.predict import generate_submission


def main():
    # 1. Reproducibility
    seed_everything(SEED)

    # 2. Training
    # We use the defined number of epochs (15) which is fast enough for this dataset size.
    print("Starting training pipeline...")
    run_training(epochs=NUM_EPOCHS, load_cached_data=True)

    # 3. Validation & Metric Calculation
    print("Performing final validation...")

    # Load validation data
    val_dataset = get_dataset(
        metadata_path=VAL_META_PATH, dataset_type="val", load_cached_data=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Load the best model
    model = SSBHDNetwork().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print(f"Error: Model checkpoint not found at {MODEL_PATH}")
        return

    model.eval()

    all_targets = []
    all_probs = []

    # Inference loop
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(DEVICE)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Calculate AUC
    try:
        final_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        final_auc = 0.5

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_auc}")

    # 4. Failure Analysis
    print("\nPerforming failure analysis...")

    # Calculate error magnitude
    errors = np.abs(all_targets - all_probs)

    # Load metadata to extract features for correlation
    # We correlate error with the number of slices per modality to see if data quantity affects performance
    val_df = pd.read_parquet(VAL_META_PATH)

    # Ensure the dataframe order matches the loader (loader is sequential if shuffle=False)
    # The dataset creation in library.data iterates row by row from the dataframe.

    meta_features = {}
    modalities = ["flair", "t1w", "t1wce", "t2w"]

    for mod in modalities:
        col_name = f"{mod}_paths"
        # Count number of files in the list
        counts = (
            val_df[col_name]
            .apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
            .values
        )
        meta_features[f"{mod}_slice_count"] = counts

    print("Correlation between Error Magnitude and Input Features (Slice Counts):")
    for feature_name, feature_values in meta_features.items():
        if len(feature_values) == len(errors):
            corr, _ = stats.pearsonr(errors, feature_values)
            print(f" - {feature_name}: {corr:.4f}")
        else:
            print(
                f" - {feature_name}: Shape mismatch ({len(feature_values)} vs {len(errors)})"
            )

    # 5. Submission Generation
    # Threshold defined in the task description
    THRESHOLD = 0.6978181818181817

    if final_auc > THRESHOLD:
        print(
            f"\nValidation metric ({final_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        print(
            f"\nValidation metric ({final_auc}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
