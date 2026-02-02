import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from provided libraries
from library.config import Config
from library.utils import set_seed, calculate_lwlrap
from library.dataset import get_datasets
from library.model import ConvNeXtAudio
from library.trainer import Trainer


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    # Override Config for a fast baseline execution
    Config.epochs = 10  # Reduced to ensure quick execution within time limits

    # Set random seed for reproducibility
    set_seed(Config.seed)

    print(f"Configuration:")
    print(f"  Device: {Config.device}")
    print(f"  Epochs: {Config.epochs}")
    print(f"  Batch Size: {Config.batch_size}")
    print(f"  Backbone: {Config.backbone}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\nInitializing Datasets...")
    # Load datasets with caching enabled to speed up startup
    train_ds, val_ds, test_ds = get_datasets(load_cached_data=True)

    # Create DataLoaders
    # Pin memory for faster host-to-device transfer
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    print(f"DataLoaders created.")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\nInitializing Model...")
    model = ConvNeXtAudio(Config)
    model = model.to(Config.device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.learning_rate,
        epochs=Config.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.pct_start,
        div_factor=Config.div_factor,
        final_div_factor=Config.final_div_factor,
    )

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("\nStarting Training...")
    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler)
    trainer.fit()

    # -------------------------------------------------------------------------
    # 5. Evaluation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\nEvaluating Best Model...")
    # Load best checkpoint
    if os.path.exists(Config.checkpoint_path):
        checkpoint = torch.load(Config.checkpoint_path, map_location=Config.device)
        model.load_state_dict(checkpoint)
    else:
        print("Warning: Checkpoint not found. Using current model state.")

    model.eval()

    val_preds = []
    val_targets = []

    # Inference on Validation Set
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(Config.device)
            # Forward pass
            outputs = model(images)
            preds = torch.sigmoid(outputs)

            val_preds.append(preds.cpu().numpy())
            val_targets.append(targets.numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Compute Final Metric
    final_metric = calculate_lwlrap(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Metric: Mean BCE Loss per sample
    # Clip predictions for numerical stability in log
    eps = 1e-7
    preds_clipped = np.clip(val_preds, eps, 1 - eps)

    # Compute binary cross entropy per element
    bce_matrix = -(
        val_targets * np.log(preds_clipped)
        + (1 - val_targets) * np.log(1 - preds_clipped)
    )

    # Average error per sample (across classes)
    error_per_sample = np.mean(bce_matrix, axis=1)

    # Correlate with Number of Labels (Polyphony)
    # Load validation metadata to get labels
    val_df = pd.read_csv(Config.val_csv)

    # Ensure alignment: The dataset loader loads files in the order of the CSV (assuming no shuffle in loader)
    # val_loader was created with shuffle=False.
    # val_ds.fnames should match val_df['fname'] if loaded sequentially.
    if not np.array_equal(val_ds.fnames, val_df["fname"].values):
        print("Warning: Validation dataset order mismatch. Attempting re-alignment.")
        val_df = val_df.set_index("fname").reindex(val_ds.fnames).reset_index()

    # Calculate number of labels per sample
    val_df["num_labels"] = val_df["labels"].apply(
        lambda x: len(str(x).split(",")) if pd.notna(x) else 0
    )

    # Calculate correlation
    correlation = np.corrcoef(val_df["num_labels"].values, error_per_sample)[0, 1]
    print(
        f"Correlation between Error Magnitude and Number of Labels: {correlation:.4f}"
    )

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.7117108825122853

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        test_preds = []

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(Config.device)
                outputs = model(images)
                preds = torch.sigmoid(outputs)
                test_preds.append(preds.cpu().numpy())

        test_preds = np.concatenate(test_preds, axis=0)

        # Prepare Submission DataFrame
        sample_sub = pd.read_csv(Config.sample_submission)

        # Verify shape
        if test_preds.shape[0] != len(sample_sub):
            print(
                f"Error: Prediction count {test_preds.shape[0]} != Sample submission count {len(sample_sub)}"
            )

        # Create dataframe with fnames from test_ds (which comes from test.csv / sample_sub)
        # The columns should match the sample submission (fname, label1, label2...)
        label_columns = sample_sub.columns[1:]

        submission_df = pd.DataFrame(test_preds, columns=label_columns)
        submission_df.insert(0, "fname", test_ds.fnames)

        # Ensure sorting matches sample_submission
        submission_df = (
            submission_df.set_index("fname").reindex(sample_sub["fname"]).reset_index()
        )

        # Save
        submission_df.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
