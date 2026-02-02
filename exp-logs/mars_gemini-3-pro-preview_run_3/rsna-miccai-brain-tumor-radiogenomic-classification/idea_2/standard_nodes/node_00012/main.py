import sys
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_train_val_datasets, get_test_dataset
from library.model import BraTS25DNet
from library.engine import train_model, generate_submission


def run():
    # 1. Setup and Initialization
    # Cite debug_lesson_2: Force update of Config parameters to handle persistent environment caching
    Config.BATCH_SIZE = 4
    Config.NUM_SLICES = 20

    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # Use cached data if available to save time
    train_ds, val_ds = get_train_val_datasets(load_cached_data=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    # Adapt 5D input (Batch, Slices, Channels, H, W) to 2.5D (Batch, Slices*Channels, H, W)
    model = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=1, end_dim=2),
        BraTS25DNet(in_channels=Config.NUM_SLICES * Config.IN_CHANNELS),
    ).to(device)

    # 4. Training
    # The engine handles the training loop, validation per epoch, and early stopping
    train_model(model, train_loader, val_loader, device)

    # 5. Final Validation Assessment
    # Load the best model checkpoint saved during training
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    model.eval()

    all_targets = []
    all_preds = []

    # Run inference on validation set
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            # Forward pass
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            targets_np = targets.numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targets_np)

    # Calculate and print the required metric
    val_auc = roc_auc_score(all_targets, all_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Load validation metadata to access feature information (slice counts)
    val_df = pd.read_parquet(Config.VAL_META_PATH)

    # Calculate error magnitude for each sample
    errors = np.abs(np.array(all_targets) - np.array(all_preds))

    # Create a DataFrame for correlation analysis
    # We use the length of the path lists as a proxy for the number of slices available
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "flair_count": val_df["flair_paths"].apply(
                lambda x: len(x) if x is not None else 0
            ),
            "t1w_count": val_df["t1w_paths"].apply(
                lambda x: len(x) if x is not None else 0
            ),
            "t1wce_count": val_df["t1wce_paths"].apply(
                lambda x: len(x) if x is not None else 0
            ),
            "t2w_count": val_df["t2w_paths"].apply(
                lambda x: len(x) if x is not None else 0
            ),
        }
    )

    # Calculate correlation between error and slice counts
    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features (Slice Counts):")
    print(correlations)

    # 7. Submission Generation
    threshold = 0.6818181818181819

    if val_auc > threshold:
        print(f"\nValidation metric {val_auc} > {threshold}. Generating submission...")

        # Load test dataset
        test_ds = get_test_dataset(load_cached_data=True)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Generate and save predictions
        generate_submission(model, test_loader, device)
    else:
        print(f"\nValidation metric {val_auc} <= {threshold}. Submission skipped.")


if __name__ == "__main__":
    run()
