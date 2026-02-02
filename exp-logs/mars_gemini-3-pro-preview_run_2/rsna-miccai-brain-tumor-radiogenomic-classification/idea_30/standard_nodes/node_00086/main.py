import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import from provided library files
from library.utils import load_cache
from library.data import get_dataloaders, compute_flair_integral_roi
from library.train import run_training, DEVICE, WORKING_DIR, BEST_MODEL_PATH
from library.model import AsymmetricEfficientNet

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 10  # Fast baseline
PATIENCE = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
NUM_WORKERS = 4
THRESHOLD_METRIC = 0.6303636363636363

METADATA_DIR = "./metadata"
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
ROI_CACHE_FILE = "./working/idea_opt/roi_cache.parquet"


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def predict(model, loader, device, use_tta=False):
    """
    Runs inference on a loader.
    If use_tta is True, averages predictions of Original, HFlip, and VFlip.
    Returns: true_labels, predicted_probs, ids
    """
    model.eval()
    all_probs = []
    all_labels = []

    # We need to track IDs to align with submission if needed,
    # but the loader order is fixed (shuffle=False).
    # The dataset __getitem__ doesn't return ID, so we rely on loader order.

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)

            # 1. Original Prediction
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            if use_tta:
                # 2. Horizontal Flip (dim 3 is width)
                inputs_h = torch.flip(inputs, dims=[3])
                logits_h = model(inputs_h)
                probs_h = torch.sigmoid(logits_h)

                # 3. Vertical Flip (dim 2 is height)
                inputs_v = torch.flip(inputs, dims=[2])
                logits_v = model(inputs_v)
                probs_v = torch.sigmoid(logits_v)

                # Average
                probs = (probs + probs_h + probs_v) / 3.0

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    return all_labels, all_probs


def perform_failure_analysis(val_df, y_true, y_pred, roi_cache):
    """
    Analyzes correlation between error and metadata features (e.g., volume depth).
    """
    print("\n=== Failure Analysis ===")

    # Calculate Error
    errors = np.abs(y_true - y_pred)

    # Extract Depth from ROI Cache
    # roi_cache is a DataFrame with BraTS21ID and sorted_ids (list)
    # We need to map BraTS21ID to Depth

    if roi_cache is not None and not roi_cache.empty:
        # Convert sorted_ids list to length (depth)
        # Ensure we handle the list correctly (it might be loaded as np array or list)
        roi_cache["depth"] = roi_cache["sorted_ids"].apply(
            lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
        )
        depth_map = dict(zip(roi_cache["BraTS21ID"], roi_cache["depth"]))

        # Map depth to validation samples
        val_depths = val_df["BraTS21ID"].map(depth_map).fillna(0).values

        # Correlation: Error vs Depth
        if len(val_depths) > 1 and np.std(val_depths) > 0:
            corr_depth, _ = pearsonr(val_depths, errors)
            print(f"Correlation (Error vs Volume Depth): {corr_depth:.4f}")
        else:
            print(
                "Correlation (Error vs Volume Depth): N/A (Constant or missing depth)"
            )

    # Correlation: Error vs Target Class
    if len(y_true) > 1 and np.std(y_true) > 0:
        corr_target, _ = pearsonr(y_true, errors)
        print(f"Correlation (Error vs Target Class): {corr_target:.4f}")
    else:
        print("Correlation (Error vs Target Class): N/A")

    print("========================\n")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    set_seed(SEED)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    print("Initializing Pipeline...")

    # 1. Train the Model
    # This function handles data loading, model init, and training loop
    # It saves the best model to BEST_MODEL_PATH
    model = run_training(
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_workers=NUM_WORKERS,
        seed=SEED,
        load_cached_roi=True,
    )

    # 2. Load Best Model for Validation
    print(f"Loading best model from {BEST_MODEL_PATH}...")
    model = AsymmetricEfficientNet(pretrained=False, dropout_rate=0.5)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # 3. Validation & Metrics
    # We need the validation loader again
    _, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, load_cached_roi=True
    )

    print("Running Validation Inference...")
    val_labels, val_preds = predict(model, val_loader, DEVICE, use_tta=False)

    # Compute Metric
    if len(np.unique(val_labels)) > 1:
        final_metric = roc_auc_score(val_labels, val_preds)
    else:
        final_metric = 0.5

    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    # Load metadata to map IDs
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    roi_cache = load_cache(ROI_CACHE_FILE)

    perform_failure_analysis(val_df, val_labels, val_preds, roi_cache)

    # 5. Submission
    if final_metric > THRESHOLD_METRIC:
        print(f"Metric {final_metric} > {THRESHOLD_METRIC}. Generating submission...")

        # Run Inference on Test Set with TTA
        # Test loader labels are placeholders (-1), we ignore them
        _, test_preds = predict(model, test_loader, DEVICE, use_tta=True)

        # Load test metadata to get IDs
        test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"BraTS21ID": test_df["BraTS21ID"], "MGMT_value": test_preds}
        )

        # Save
        submission_df.to_csv(SUBMISSION_FILE, index=False)
        print(f"Submission saved to {SUBMISSION_FILE}")

        # Verify
        print("Submission Head:")
        print(submission_df.head())
    else:
        print(
            f"Metric {final_metric} <= {THRESHOLD_METRIC}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
