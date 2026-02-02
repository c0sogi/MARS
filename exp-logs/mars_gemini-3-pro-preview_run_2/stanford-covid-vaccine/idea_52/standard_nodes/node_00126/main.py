import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import scipy.stats as stats
from torch.utils.data import DataLoader

# Import from provided libraries
from library.config import Config
from library.utils import set_seed, MCRMSELoss, get_global_rmse
from library.data import preprocess_data, RNADataset
from library.model import SSPFN
from library.train import train_one_epoch, validate


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Override Config for fast baseline execution
    Config.EPOCHS = 15  # Reduce epochs to ensure completion within time limit

    print(f"Execution Device: {device}")
    print(f"Training for {Config.EPOCHS} epochs.")

    # 2. Data Loading
    print("Loading Data...")
    # Load Train
    train_feats, train_p_idx, train_targets, train_ids = preprocess_data(
        Config.TRAIN_FILE, Config.TRAIN_CACHE, load_cached_data=True, is_test=False
    )
    # Load Val
    val_feats, val_p_idx, val_targets, val_ids = preprocess_data(
        Config.VAL_FILE, Config.VAL_CACHE, load_cached_data=True, is_test=False
    )

    # Create Datasets and Loaders
    train_dataset = RNADataset(train_feats, train_p_idx, train_targets, train_ids)
    val_dataset = RNADataset(val_feats, val_p_idx, val_targets, val_ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = SSPFN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = MCRMSELoss()

    # 4. Training Loop
    best_score = float("inf")

    # Ensure working directory exists for model saving
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        start_t = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_score = validate(model, val_loader, criterion, device)

        scheduler.step(val_score)

        elapsed = time.time() - start_t
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val MCRMSE: {val_score:.5f} | Time: {elapsed:.1f}s"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    print(f"Training Complete. Best Validation Score: {best_score}")

    # 5. Final Validation & Metrics
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []
    all_ids_list = []

    with torch.no_grad():
        for features, partner_indices, targets in val_loader:
            features = features.to(device)
            partner_indices = partner_indices.to(device)

            # Pass 1
            pred1 = model(features, partner_indices, feedback_input=None)
            # Pass 2
            pred2 = model(features, partner_indices, feedback_input=pred1)

            all_preds.append(pred2.cpu().numpy())
            all_targets.append(targets.numpy())

    # Reconstruct IDs from loader (dataset order is preserved if shuffle=False)
    # The loader iterates sequentially over val_dataset which was built from val_ids
    all_ids_list = val_ids

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    final_metric = get_global_rmse(all_preds, all_targets)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n==== Failure Analysis ====")
    # Calculate RMSE per sample
    # Scored columns indices in the 5-channel output
    scored_indices = [
        i for i, col in enumerate(Config.ALL_TARGET_COLS) if col in Config.TARGET_COLS
    ]
    seq_scored = Config.SCORED_SEQ_LENGTH

    # Filter preds and targets
    p_filtered = all_preds[:, :seq_scored, scored_indices]
    t_filtered = all_targets[:, :seq_scored, scored_indices]

    # MSE per sample: mean over (sequence_length, channels)
    sample_mse = np.mean((p_filtered - t_filtered) ** 2, axis=(1, 2))
    sample_rmse = np.sqrt(sample_mse)

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame({"id": all_ids_list, "rmse": sample_rmse})

    # Load metadata
    if os.path.exists(Config.VAL_FILE):
        val_meta = pd.read_csv(Config.VAL_FILE)
        # Merge on ID
        analysis_df = analysis_df.merge(val_meta, on="id", how="left")

        # Calculate correlations
        # Check for numeric columns of interest
        cols_to_check = [
            "signal_to_noise",
            "mean_reactivity",
            "seq_length",
            "SN_filter",
        ]

        print("Correlation between Sample RMSE and Features:")
        for col in cols_to_check:
            if col in analysis_df.columns:
                # Drop NaNs just in case
                valid_data = analysis_df[[col, "rmse"]].dropna()
                if len(valid_data) > 1:
                    corr, _ = stats.pearsonr(valid_data[col], valid_data["rmse"])
                    print(f"  {col}: {corr:.4f}")
    else:
        print("Validation metadata file not found. Skipping detailed failure analysis.")

    # 7. Submission
    THRESHOLD = 0.47142532743789534

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        # Load Test Data
        test_feats, test_p_idx, test_targets, test_ids = preprocess_data(
            Config.TEST_FILE, Config.TEST_CACHE, load_cached_data=True, is_test=True
        )

        test_dataset = RNADataset(test_feats, test_p_idx, test_targets, test_ids)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        test_preds = []
        with torch.no_grad():
            for features, partner_indices, _ in test_loader:
                features = features.to(device)
                partner_indices = partner_indices.to(device)

                # Pass 1
                pred1 = model(features, partner_indices, feedback_input=None)
                # Pass 2
                pred2 = model(features, partner_indices, feedback_input=pred1)

                test_preds.append(pred2.cpu().numpy())

        test_preds = np.concatenate(test_preds, axis=0)  # (N, 107, 5)

        # Format Submission
        submission_data = []
        for i, sample_id in enumerate(test_ids):
            sample_p = test_preds[i]
            for seqpos in range(Config.SEQ_LENGTH):
                row_id = f"{sample_id}_{seqpos}"
                row_vals = sample_p[seqpos]
                submission_data.append(
                    {
                        "id_seqpos": row_id,
                        "reactivity": row_vals[0],
                        "deg_Mg_pH10": row_vals[1],
                        "deg_pH10": row_vals[2],
                        "deg_Mg_50C": row_vals[3],
                        "deg_50C": row_vals[4],
                    }
                )

        submission_df = pd.DataFrame(submission_data)
        # Order columns
        cols = ["id_seqpos"] + Config.ALL_TARGET_COLS
        submission_df = submission_df[cols]

        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
