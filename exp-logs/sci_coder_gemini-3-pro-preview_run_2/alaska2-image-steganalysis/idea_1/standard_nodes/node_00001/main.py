import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.trainer import Trainer
from library.utils import seed_everything, weighted_auc_score


def run_failure_analysis(trainer, val_loader):
    """
    Performs validation inference, calculates the final metric, and runs failure analysis.
    """
    print("\n--- Starting Failure Analysis ---")

    device = trainer.device
    model = trainer.model
    model.eval()

    all_preds = []
    all_targets = []
    all_means = []
    all_sizes = []

    # Ensure we access the samples list to get file paths for metadata analysis
    # The val_loader is created with shuffle=False, so indices align with iteration
    dataset_samples = val_loader.dataset.samples

    batch_size = val_loader.batch_size
    current_idx = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)

            # Forward pass (Mixed Precision)
            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                outputs = model(images)
                preds = torch.sigmoid(outputs)

            # Store predictions and targets
            all_preds.extend(preds.cpu().numpy().ravel())
            all_targets.extend(labels.numpy())

            # Feature 1: Mean Pixel Intensity (computed on GPU for speed)
            # images shape: (B, 3, H, W) -> mean over dims 1,2,3
            batch_means = images.mean(dim=(1, 2, 3)).cpu().numpy()
            all_means.extend(batch_means)

            # Feature 2: File Size (read from disk metadata)
            batch_len = images.size(0)
            batch_samples = dataset_samples[current_idx : current_idx + batch_len]

            # Construct full paths and get size
            sizes = []
            for sample in batch_samples:
                full_path = os.path.join(Config.INPUT_DIR, sample["file_path"])
                try:
                    sizes.append(os.path.getsize(full_path))
                except OSError:
                    sizes.append(0)
            all_sizes.extend(sizes)

            current_idx += batch_len

    # Convert to numpy arrays
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # 1. Calculate and Print Final Metric
    final_metric = weighted_auc_score(
        y_true, y_pred, tpr_thresholds=Config.TPR_THRESHOLDS, weights=Config.TPR_WEIGHTS
    )
    print(f"Final Validation Metric: {final_metric}")

    # 2. Failure Analysis
    # Calculate absolute error
    errors = np.abs(y_true - y_pred)

    # Create DataFrame for correlation analysis
    df_analysis = pd.DataFrame(
        {"error": errors, "mean_intensity": all_means, "file_size": all_sizes}
    )

    print("\nCorrelation between Error Magnitude and Input Features:")
    correlations = df_analysis.corr()["error"].drop("error")
    print(correlations)

    return final_metric


def main():
    # --- Configuration Override for Fast Baseline ---
    # Limit epochs to 1 for a quick baseline run within time limits
    Config.EPOCHS = 1
    # Increase batch size to utilize A100 memory
    Config.BATCH_SIZE = 64
    # Ensure workers are set
    Config.NUM_WORKERS = 12

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    print("Initializing Trainer...")
    trainer = Trainer()

    # --- Training ---
    print("Starting Training...")
    trainer.fit()

    # --- Validation & Failure Analysis ---
    # Load validation data explicitly for analysis
    val_loader = trainer.get_dataloader("val")
    run_failure_analysis(trainer, val_loader)

    # --- Submission ---
    print("\nGenerating Submission...")
    trainer.predict()
    print("Process Complete.")


if __name__ == "__main__":
    main()
