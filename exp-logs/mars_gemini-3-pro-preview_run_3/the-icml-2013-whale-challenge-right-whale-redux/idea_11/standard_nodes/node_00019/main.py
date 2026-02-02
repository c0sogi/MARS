import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import components from the provided library files
from library.config import Config
from library.utils import seed_everything, get_logger, calculate_auc
from library.dataset import WhaleDataset, prepare_data
from library.model import DualStreamEfficientNet
from library.trainer import Trainer


def main():
    # 1. Setup and Configuration
    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    # Initialize logger
    logger = get_logger("RunFile")

    # Modify Config for a Fast Baseline
    # Reducing epochs and limiting training samples to satisfy the "fast baseline" requirement
    Config.EPOCHS = 5
    TRAIN_SAMPLE_LIMIT = 5000

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")

    # 2. Data Loading
    logger.info("Loading metadata...")
    train_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))

    # Load Training Data (utilizing cache if available)
    logger.info("Preparing training data...")
    # We load the full cached data first to avoid recomputing features, then slice it
    X1_train_full, X2_train_full, Y_train_full, _ = prepare_data(
        train_df, Config, cache_name="train", load_cached_data=True
    )

    # Limit training samples for speed
    if len(Y_train_full) > TRAIN_SAMPLE_LIMIT:
        logger.info(f"Subsampling training data to {TRAIN_SAMPLE_LIMIT} samples.")
        # Use fixed seed for subsampling reproducibility
        rng = np.random.RandomState(Config.SEED)
        indices = rng.choice(len(Y_train_full), TRAIN_SAMPLE_LIMIT, replace=False)
        X1_train = X1_train_full[indices]
        X2_train = X2_train_full[indices]
        Y_train = Y_train_full[indices]
    else:
        X1_train, X2_train, Y_train = X1_train_full, X2_train_full, Y_train_full

    # Load Validation Data (Full set required for valid metric)
    logger.info("Preparing validation data...")
    X1_val, X2_val, Y_val, _ = prepare_data(
        val_df, Config, cache_name="val", load_cached_data=True
    )

    # Create Datasets
    train_dataset = WhaleDataset(X1_train, X2_train, Y_train, augment=True)
    val_dataset = WhaleDataset(X1_val, X2_val, Y_val, augment=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = DualStreamEfficientNet(Config).to(device)

    # Loss Function (Weighted BCE)
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # 4. Training
    logger.info("Starting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=Config,
    )

    best_model_path = trainer.fit()

    # 5. Final Validation & Failure Analysis
    logger.info("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    # Inference on Validation Set
    with torch.no_grad():
        for x1, x2, y in val_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            outputs = model(x1, x2)
            preds = torch.sigmoid(outputs).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(y.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets)

    # Calculate and Print Metric
    val_auc = calculate_auc(all_targets, all_preds)
    print(f"Final Validation Metric: {val_auc}")

    # Failure Analysis
    logger.info("Performing failure analysis...")
    errors = np.abs(all_targets - all_preds)

    # Compute aggregate statistics of input spectrograms for correlation
    # X1_val shape: (N, 1, F, T)
    # We use the Spectral Stream (X1) for analysis
    spec_means = X1_val.mean(axis=(1, 2, 3))
    spec_stds = X1_val.std(axis=(1, 2, 3))

    df_analysis = pd.DataFrame(
        {"error": errors, "spec_mean": spec_means, "spec_std": spec_stds}
    )

    corr_mean = df_analysis["error"].corr(df_analysis["spec_mean"])
    corr_std = df_analysis["error"].corr(df_analysis["spec_std"])

    print(f"Correlation between Error and Spectrogram Mean: {corr_mean}")
    print(f"Correlation between Error and Spectrogram Std: {corr_std}")

    # 6. Submission Generation
    SUBMISSION_THRESHOLD = 0.9960914834372254

    if val_auc > SUBMISSION_THRESHOLD:
        logger.info(
            f"Validation AUC ({val_auc}) exceeds threshold. Generating submission..."
        )

        # Load Test Data
        test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))
        X1_test, X2_test, Y_test, clips = prepare_data(
            test_df, Config, cache_name="test", load_cached_data=True
        )

        test_dataset = WhaleDataset(X1_test, X2_test, Y_test, augment=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Inference on Test Set
        test_preds = []
        with torch.no_grad():
            for x1, x2, _ in test_loader:
                x1 = x1.to(device)
                x2 = x2.to(device)

                outputs = model(x1, x2)
                preds = torch.sigmoid(outputs).cpu().numpy()
                test_preds.extend(preds)

        test_preds = np.array(test_preds).flatten()

        # Save Submission
        submission = pd.DataFrame({"clip": clips, "probability": test_preds})
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.info(
            f"Validation AUC ({val_auc}) did not meet threshold ({SUBMISSION_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
