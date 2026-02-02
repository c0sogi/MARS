import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

# Import library modules
from library.config import Config
from library.train import run_training
from library.inference import predict_test_set
from library.utils import seed_everything, KLDivLossWithLogits
from library.data_loader import get_dataloaders
from library.models import DualStreamNetwork


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Configure for Fast Baseline Execution (Target < 2 hours)
    # We use a subset of data to ensure preprocessing and training fits in time.
    # 20,000 samples is approximately 25% of the dataset, sufficient for a strong baseline
    # while ensuring preprocessing and 5 epochs of training complete rapidly.
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20000
    Config.EPOCHS = 5
    Config.BATCH_SIZE = 32
    Config.NUM_WORKERS = 12  # Utilize all available vCPUs

    # Use a specific cache directory to avoid conflicts with existing full-dataset caches
    # and ensure we process exactly the subset size we defined.
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "fast_run_cache")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print("=== Configuration ===")
    print(f"Device: {Config.DEVICE}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Subset Size: {Config.DEBUG_SUBSET_SIZE}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Cache Dir: {Config.CACHE_DIR}")
    print("=====================")

    # ==========================================
    # 2. Training
    # ==========================================
    print("\nStarting Training Pipeline...")
    # run_training encapsulates the training loop, validation monitoring, and model saving.
    # It will use the Config settings modified above.
    run_training()

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("\nStarting Validation & Failure Analysis...")
    device = torch.device(Config.DEVICE)

    # Load the validation dataloader
    # (This uses the cache generated during the training setup, so it's fast)
    # get_dataloaders returns (train, val, test) -> we take index 1
    _, val_loader, _ = get_dataloaders(debug=Config.DEBUG)

    # Load the best saved model
    model = DualStreamNetwork(num_classes=Config.N_CLASSES, pretrained=False)
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {Config.MODEL_PATH}")

    state_dict = torch.load(Config.MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Run Inference on Validation Set
    all_logits = []
    all_targets = []

    print(f"Evaluating on {len(val_loader.dataset)} validation samples...")
    with torch.no_grad():
        for eeg, spec, targets in val_loader:
            eeg = eeg.to(device)
            spec = spec.to(device)
            targets = targets.to(device)

            logits = model(eeg, spec)

            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    # Concatenate results
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute Final Validation Metric (KL Divergence)
    # The competition metric is KL Divergence.
    # We use F.log_softmax on logits and then KLDivLoss with reduction='batchmean'.
    log_probs = F.log_softmax(all_logits, dim=1)
    kl_criterion = torch.nn.KLDivLoss(reduction="batchmean")
    final_metric = kl_criterion(log_probs, all_targets).item()

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\nPerforming Failure Analysis...")
    # Compute per-sample loss (sum over classes) to identify hard samples
    kl_none = torch.nn.KLDivLoss(reduction="none")
    # Loss shape: (N, Classes) -> Sum -> (N,)
    per_sample_loss = kl_none(log_probs, all_targets).sum(dim=1).numpy()

    # Load validation metadata to correlate features with error
    df_val = pd.read_csv(Config.VAL_CSV)
    if Config.DEBUG:
        df_val = df_val.head(Config.DEBUG_SUBSET_SIZE)

    # Align lengths (handle potential mismatches if data loader dropped samples)
    n_samples = len(per_sample_loss)
    if len(df_val) > n_samples:
        df_val = df_val.iloc[:n_samples]
    elif len(df_val) < n_samples:
        per_sample_loss = per_sample_loss[: len(df_val)]

    df_val["error_magnitude"] = per_sample_loss

    # Calculate correlations with numeric metadata
    numeric_cols = df_val.select_dtypes(include=[np.number]).columns
    # Exclude the error column itself and target probabilities
    exclude_cols = ["error_magnitude"] + Config.TARGET_COLS
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    correlations = (
        df_val[feature_cols]
        .corrwith(df_val["error_magnitude"])
        .sort_values(key=abs, ascending=False)
    )

    print("Top 5 Features correlated with Error Magnitude:")
    print(correlations.head(5))

    # ==========================================
    # 4. Submission Generation
    # ==========================================
    TARGET_THRESHOLD = 1.0081

    if final_metric < TARGET_THRESHOLD:
        print(
            f"\nMetric {final_metric:.4f} meets threshold {TARGET_THRESHOLD}. Generating submission..."
        )
        # predict_test_set handles loading test data, model, and saving CSV
        # We enable caching to speed up the process if run multiple times
        predict_test_set(load_cached_data=True)
    else:
        print(
            f"\nMetric {final_metric:.4f} does not meet threshold {TARGET_THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
