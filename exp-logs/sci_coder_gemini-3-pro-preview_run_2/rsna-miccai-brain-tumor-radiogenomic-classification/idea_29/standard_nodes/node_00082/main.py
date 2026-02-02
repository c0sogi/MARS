import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import from the provided library files
from library.config import (
    VAL_METADATA_PATH,
    VAL_CACHE_PATH,
    VAL_LABEL_CACHE_PATH,
    MODEL_SAVE_PATH,
    DEVICE,
    INPUT_DIR,
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
)
from library.trainer import train_model, get_transforms
from library.inference import generate_submission
from library.dataset import load_dataset
from library.model import AsymmetricEfficientNet


def set_seed(seed=SEED):
    """Sets the random seed for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_pipeline():
    # Set global seeds
    set_seed()

    # --------------------------------------------------------------------------
    # 1. Training Phase
    # --------------------------------------------------------------------------
    print("--- Starting Training Pipeline ---")
    # Train the model using the configuration in library/config.py
    # This will also generate and save the data caches.
    train_model(debug_max_samples=None)

    # --------------------------------------------------------------------------
    # 2. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\n--- Starting Validation & Failure Analysis ---")

    # Load the validation dataset (leveraging the cache created during training)
    val_dataset = load_dataset(
        metadata_path=VAL_METADATA_PATH,
        cache_path_data=VAL_CACHE_PATH,
        cache_path_labels=VAL_LABEL_CACHE_PATH,
        load_cached_data=True,
        transform=get_transforms("val"),
        debug_max_samples=None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Load the best model saved during training
    model = AsymmetricEfficientNet()
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        print(f"Loaded best model from {MODEL_SAVE_PATH}")
    else:
        print(
            f"Error: Model file not found at {MODEL_SAVE_PATH}. Cannot proceed with validation."
        )
        return

    model.to(DEVICE)
    model.eval()

    # Run Inference on Validation Set
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            # Forward pass
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            val_preds.extend(probs.cpu().numpy().flatten())
            val_targets.extend(labels.numpy().flatten())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    # Compute Metric
    final_metric = roc_auc_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate absolute error
    errors = np.abs(val_targets - val_preds)

    # Extract 'FLAIR Slice Count' feature for correlation analysis
    # We read the metadata again to ensure we match the order (dataset preserves order)
    df_val = pd.read_csv(VAL_METADATA_PATH)

    slice_counts = []
    for _, row in df_val.iterrows():
        # Construct the full path to the FLAIR directory
        flair_dir = os.path.join(INPUT_DIR, row["path_FLAIR"])
        if os.path.exists(flair_dir):
            # Count files in the directory
            count = len([f for f in os.listdir(flair_dir) if f.endswith(".dcm")])
            slice_counts.append(count)
        else:
            slice_counts.append(0)

    slice_counts = np.array(slice_counts)

    # Calculate correlation
    if len(errors) == len(slice_counts):
        corr, _ = pearsonr(errors, slice_counts)
        print(f"Correlation between Error and FLAIR Slice Count: {corr}")
    else:
        print(
            "Warning: Validation set size mismatch. Skipping specific feature correlation."
        )

    # --------------------------------------------------------------------------
    # 3. Submission Generation
    # --------------------------------------------------------------------------
    threshold = 0.6303636363636363

    if final_metric > threshold:
        print(f"\nValidation metric ({final_metric}) exceeds threshold ({threshold}).")
        print("Generating submission file...")
        generate_submission(debug_max_samples=None, load_cached_data=True)
    else:
        print(
            f"\nValidation metric ({final_metric}) does not exceed threshold ({threshold})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    run_pipeline()
