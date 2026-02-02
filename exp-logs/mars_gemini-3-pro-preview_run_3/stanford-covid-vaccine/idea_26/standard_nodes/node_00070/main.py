import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library modules
from library import config, utils, data, model, train


def main():
    # =========================================================================
    # 1. Setup & Configuration
    # =========================================================================
    # Override config for Fast Baseline execution
    config.EPOCHS = 10
    TRAIN_SUBSET_SIZE = 1200  # Limit training samples for speed

    # Set seeds for reproducibility
    utils.set_seed(config.SEED)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("Loading and processing data...")

    # Load raw data dictionaries using library function
    # We load cached data if available to save time
    train_dict = data.load_or_process_data(
        config.TRAIN_PATH, load_cached_data=True, is_test=False
    )
    val_dict = data.load_or_process_data(
        config.VAL_PATH, load_cached_data=True, is_test=False
    )
    test_dict = data.load_or_process_data(
        config.TEST_PATH, load_cached_data=True, is_test=True
    )

    # Manually slice Training Data to satisfy "Limit maximum number of training samples"
    # We do NOT slice Validation data to ensure the metric is computed on the full set.
    print(f"Slicing training data to {TRAIN_SUBSET_SIZE} samples.")
    for k in train_dict.keys():
        train_dict[k] = train_dict[k][:TRAIN_SUBSET_SIZE]

    # Create Datasets
    train_dataset = data.RNADataset(train_dict)
    val_dataset = data.RNADataset(val_dict)
    test_dataset = data.RNADataset(test_dict)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    print("Initializing model...")
    net = model.DeepPostNormBiGRU().to(device)

    # =========================================================================
    # 4. Training Loop
    # =========================================================================
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    best_score = float("inf")
    best_model_path = os.path.join(config.WORKING_DIR, "best_model_runfile.pth")

    print("Starting training...")
    for epoch in range(config.EPOCHS):
        # Train
        train_loss = train.train_one_epoch(net, train_loader, optimizer, device)

        # Validate
        val_score = train.validate(net, val_loader, device)

        # Scheduler Step
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.4f} | Val MCRMSE: {val_score:.6f}"
        )

        # Checkpoint
        if val_score < best_score:
            best_score = val_score
            torch.save(net.state_dict(), best_model_path)

    print(f"Training complete. Best Val Score: {best_score}")

    # =========================================================================
    # 5. Final Validation & Metric
    # =========================================================================
    # Load best model for final evaluation
    net.load_state_dict(torch.load(best_model_path, map_location=device))
    net.eval()

    # Compute metric on full validation set
    final_metric = train.validate(net, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 6. Failure Analysis
    # =========================================================================
    print("\nRunning Failure Analysis...")
    all_ids = []
    all_errors = []

    # Compute per-sample error magnitude
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["targets"]  # Targets are typically on CPU from Dataset
            ids = batch["ids"]

            # Inference
            preds = net(inputs, pair_indices, pair_mask).cpu()

            # Slice to scored region (first 68 positions)
            preds_sliced = preds[:, : config.SEQ_SCORED, :]
            targets_sliced = targets[:, : config.SEQ_SCORED, :]

            # Calculate RMSE per sample (averaged over sequence and targets)
            # Shape: (Batch, Seq, Targets) -> (Batch,)
            mse_per_sample = torch.mean(
                (preds_sliced - targets_sliced) ** 2, dim=(1, 2)
            )
            rmse_per_sample = torch.sqrt(mse_per_sample)

            all_errors.extend(rmse_per_sample.numpy())
            all_ids.extend(ids)

    # Load Validation Metadata
    df_val = pd.read_parquet(config.VAL_PATH)

    # Map calculated errors to the dataframe
    error_map = dict(zip(all_ids, all_errors))
    df_val["model_error"] = df_val["id"].map(error_map)

    # Calculate correlations with numeric features
    numeric_cols = df_val.select_dtypes(include=[np.number]).columns
    correlations = (
        df_val[numeric_cols]
        .corrwith(df_val["model_error"])
        .sort_values(ascending=False)
    )

    print("Correlation of Input Features with Model Error:")
    print(correlations)

    # =========================================================================
    # 7. Submission
    # =========================================================================
    THRESHOLD = 0.5978901386

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        # Generate predictions
        preds, ids = train.predict_test(net, test_loader, device)

        # Format submission
        submission_data = []
        for i, sample_id in enumerate(ids):
            sample_preds = preds[i]  # Shape: (107, 5)

            for seq_pos in range(config.SEQ_LEN):
                row_id = f"{sample_id}_{seq_pos}"
                row_values = sample_preds[seq_pos].tolist()

                row_dict = {"id_seqpos": row_id}
                for col_idx, col_name in enumerate(config.TARGET_COLS):
                    row_dict[col_name] = row_values[col_idx]

                submission_data.append(row_dict)

        submission_df = pd.DataFrame(submission_data)

        # Ensure correct column order
        cols = ["id_seqpos"] + config.TARGET_COLS
        submission_df = submission_df[cols]

        # Save
        os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(config.SUBMISSION_FILE_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_FILE_PATH}")
        print(f"Submission shape: {submission_df.shape}")

    else:
        print(
            f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
