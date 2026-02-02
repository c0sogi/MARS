import sys
import os
import pandas as pd
import numpy as np
import torch

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.trainer import Trainer
from library.dataset import get_dataloaders
from library.utils import seed_everything


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for Fast Baseline Execution
    # EfficientNet-B0 on A100 is fast, but we limit epochs to ensure < 2h runtime
    Config.EPOCHS = 10
    Config.BATCH_SIZE = 32  # Decrease batch size to avoid OOM with large inputs
    Config.NUM_WORKERS = 8  # Utilize available vCPUs

    # Ensure reproducibility
    seed_everything(Config.SEED)

    print(
        f"Starting execution with {Config.EPOCHS} epochs and batch size {Config.BATCH_SIZE}..."
    )

    # ==========================================
    # 2. Training
    # ==========================================
    trainer = Trainer()

    # Run training
    # fit() returns the test_loader which we will use for prediction later
    # load_cached_data=False forces reprocessing to ensure full dataset is used (fixes stale debug cache)
    test_loader = trainer.fit(load_cached_data=False)

    # ==========================================
    # 3. Validation & Metric Calculation
    # ==========================================
    # Retrieve validation loader for final metric and failure analysis
    # We call get_dataloaders again; since caching is enabled, this is fast.
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Calculate metric on the best model (loaded automatically at end of fit)
    print("Calculating final validation metric...")
    val_loss, val_lwlrap = trainer.validate(val_loader)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_lwlrap}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("Performing failure analysis...")
    trainer.model.eval()

    # Use BCE loss as a proxy for error magnitude per sample
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    errors = []
    num_labels_list = []
    file_sizes = []

    # Map fnames to filepaths for metadata extraction
    val_df = pd.read_csv(Config.VAL_CSV)
    fname_to_path = pd.Series(val_df.filepath.values, index=val_df.fname).to_dict()

    device = Config.DEVICE

    with torch.no_grad():
        for data, target, fnames in val_loader:
            data = data.to(device)
            target = target.to(device)

            # Forward pass
            logits = trainer.model(data)

            # Calculate loss per sample (average over classes)
            # Shape: (Batch, Num_Classes) -> (Batch,)
            loss_per_sample = criterion(logits, target).mean(dim=1)
            errors.extend(loss_per_sample.cpu().numpy())

            # Feature 1: Number of Labels (Complexity)
            # Sum of ground truth vector
            n_labels = target.sum(dim=1)
            num_labels_list.extend(n_labels.cpu().numpy())

            # Feature 2: File Size (Proxy for Duration/Information Content)
            for fname in fnames:
                rel_path = fname_to_path.get(fname)
                if rel_path:
                    full_path = os.path.join(Config.INPUT_ROOT, rel_path)
                    if os.path.exists(full_path):
                        file_sizes.append(os.path.getsize(full_path))
                    else:
                        file_sizes.append(0)
                else:
                    file_sizes.append(0)

    # Convert to numpy arrays for correlation calculation
    errors = np.array(errors)
    num_labels_list = np.array(num_labels_list)
    file_sizes = np.array(file_sizes)

    # Calculate Correlations using NumPy
    if len(errors) > 1 and np.std(num_labels_list) > 0:
        corr_labels = np.corrcoef(errors, num_labels_list)[0, 1]
        print(f"Correlation (Error vs Num Labels): {corr_labels}")
    else:
        print("Correlation (Error vs Num Labels): Undefined (insufficient variance)")

    if len(errors) > 1 and np.std(file_sizes) > 0:
        corr_size = np.corrcoef(errors, file_sizes)[0, 1]
        print(f"Correlation (Error vs File Size): {corr_size}")
    else:
        print("Correlation (Error vs File Size): Undefined (insufficient variance)")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    THRESHOLD = 0.7117108825122853

    if val_lwlrap > THRESHOLD:
        print(
            f"Validation Metric ({val_lwlrap}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict(test_loader)
    else:
        print(
            f"Validation Metric ({val_lwlrap}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
