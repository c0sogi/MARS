import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, weighted_log_loss, load_or_generate_cache
from library.data import get_loaders, get_study_paths
from library.engine import Trainer, inference


# -----------------------------------------------------------------------------
# Configuration Override for Fast Baseline
# -----------------------------------------------------------------------------
class FastConfig(Config):
    """
    Overrides Config to ensure the task completes quickly within the time limit.
    """

    # Reduce epochs for a fast baseline
    EPOCHS = 5

    # Ensure we use the best model for validation/inference
    LOAD_CACHED_DATA = True

    # Ensure deterministic behavior
    SEED = 42


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def calculate_study_level_loss(y_true, y_pred, weights):
    """
    Calculates the weighted log loss for each study individually.
    Returns a numpy array of losses.
    """
    # Clip predictions
    epsilon = 1e-7
    y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)

    # Calculate BCE per element: -(y*log(p) + (1-y)*log(1-p))
    loss_per_element = -(
        y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)
    )

    # Apply class weights
    weighted_loss = loss_per_element * weights

    # Mean over classes (rows) for each study
    study_losses = weighted_loss.mean(dim=1)

    return study_losses.cpu().numpy()


def perform_failure_analysis(val_df, y_true, y_pred, config):
    """
    Analyzes the correlation between model error and input features (slice count).
    """
    print("\n=== Failure Analysis ===")

    # 1. Calculate Loss per Study
    metric_weights = torch.ones(len(config.TARGET_COLS), device=y_true.device)
    if "patient_overall" in config.TARGET_COLS:
        idx = config.TARGET_COLS.index("patient_overall")
        metric_weights[idx] = 7.0

    losses = calculate_study_level_loss(y_true, y_pred, metric_weights)

    # 2. Get Metadata Features (Slice Count)
    # Load the validation paths cache to get slice counts efficiently
    cache_path = os.path.join(config.CACHE_DIR, "val_paths_cache.parquet")

    if os.path.exists(cache_path):
        paths_df = pd.read_parquet(cache_path)
        # Count slices per study
        slice_counts = (
            paths_df.groupby("StudyInstanceUID").size().reset_index(name="slice_count")
        )
    else:
        # Fallback if cache doesn't exist (should not happen given pipeline)
        slice_counts = pd.DataFrame(
            {
                "StudyInstanceUID": val_df["StudyInstanceUID"].unique(),
                "slice_count": [0] * len(val_df["StudyInstanceUID"].unique()),
            }
        )

    # 3. Merge Loss and Features
    # val_df order matches y_true/y_pred order from the loader (shuffle=False)
    analysis_df = val_df[["StudyInstanceUID"]].copy()
    analysis_df["error_magnitude"] = losses

    analysis_df = analysis_df.merge(slice_counts, on="StudyInstanceUID", how="left")
    analysis_df["slice_count"] = analysis_df["slice_count"].fillna(0)

    # 4. Calculate Correlation
    if len(analysis_df) > 1:
        correlation = analysis_df["error_magnitude"].corr(analysis_df["slice_count"])
        print(f"Correlation (Error Magnitude vs Slice Count): {correlation:.6f}")

        # Additional stats
        high_error_cutoff = np.percentile(losses, 90)
        print(f"90th Percentile Error Threshold: {high_error_cutoff:.6f}")
    else:
        print("Insufficient validation data for correlation analysis.")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------


def main():
    # 1. Setup
    config = FastConfig()
    seed_everything(config.SEED)

    # 2. Training
    # Trainer handles model init, optimizer, loop, and saving best_model.pth
    trainer = Trainer(config)
    best_loss = trainer.fit()

    # 3. Validation Assessment
    # Reload best model for final validation metric calculation
    # (Trainer.fit loads it at the end, but we ensure device placement and eval mode)
    device = torch.device(config.DEVICE)
    model = trainer.model
    model.eval()

    _, val_loader = get_loaders()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            targets = batch["label_study"].to(device)

            # Inference (no gradient, mixed precision if enabled in engine but simple here)
            outputs = model(images)
            preds = torch.sigmoid(outputs["study_logits"])

            all_preds.append(preds)
            all_targets.append(targets)

    if not all_preds:
        print("Error: Validation set empty.")
        return

    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)

    # Calculate Competition Metric
    metric_weights = torch.ones(len(config.TARGET_COLS), device=device)
    if "patient_overall" in config.TARGET_COLS:
        idx = config.TARGET_COLS.index("patient_overall")
        metric_weights[idx] = 7.0

    final_metric = weighted_log_loss(y_true, y_pred, weights=metric_weights).item()

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    # Load validation metadata to map UIDs
    val_df = pd.read_csv(config.VAL_METADATA)
    perform_failure_analysis(val_df, y_true, y_pred, config)

    # 5. Submission
    # Threshold check
    THRESHOLD = 0.15364714496434773

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        inference(config)
    else:
        print(
            f"\nMetric ({final_metric:.6f}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
