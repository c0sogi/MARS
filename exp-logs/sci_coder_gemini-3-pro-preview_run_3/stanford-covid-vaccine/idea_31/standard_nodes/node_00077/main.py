import sys
import os
import torch
import pandas as pd
import numpy as np

# Ensure local library is importable
sys.path.append(os.getcwd())

from library.config import Config
from library.dataset import RNADataset
from library.model import SDCGBiGRU
from library.loss import MCRMSELoss
from library.engine import set_seed, train_fn, eval_fn

# Override Config for Fast Baseline constraints and Path requirements
Config.EPOCHS = (
    25  # Sufficient for convergence on small dataset, fast enough for baseline
)
Config.SUBMISSION_PATH = "./submission/submission.csv"


def run_pipeline():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # 2. Data Loading
    print("Loading datasets...")
    train_dataset = RNADataset(split="train")
    val_dataset = RNADataset(split="val")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print(f"Initializing model on {device}...")
    model = SDCGBiGRU().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )
    criterion = MCRMSELoss()

    # 4. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_score = float("inf")

    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, criterion, device)
        val_score = eval_fn(model, val_loader, device)
        scheduler.step()

        # Save best model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    print(f"Training finished. Best Val Score: {best_score}")

    # 5. Final Validation & Failure Analysis
    print("Performing validation and failure analysis...")

    # Load best model
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    model.eval()

    # Compute Final Metric on full validation set
    final_metric = eval_fn(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation of Error with Metadata
    # Collect predictions and targets
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            targets = batch["target"].to(device)
            ids = batch["id"]

            outputs = model(inputs, pair_indices)
            outputs_sliced = outputs[:, : Config.PRED_LEN, :]

            all_preds.append(outputs_sliced.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(ids)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Filter to scored columns for error analysis
    target_cols = Config.TARGET_COLS
    scored_cols = Config.SCORED_COLS
    scored_indices = [i for i, col in enumerate(target_cols) if col in scored_cols]

    filtered_preds = all_preds[:, :, scored_indices]
    filtered_targets = all_targets[:, :, scored_indices]

    # Calculate RMSE per sample (averaged over scored columns and sequence length)
    sample_mse = torch.mean((filtered_preds - filtered_targets) ** 2, dim=(1, 2))
    sample_rmse = torch.sqrt(sample_mse).numpy()

    # Load Validation Metadata
    val_df = pd.read_parquet(Config.VAL_PATH)
    val_df.set_index("id", inplace=True)

    analysis_data = []
    for i, sample_id in enumerate(all_ids):
        if sample_id in val_df.index:
            row = val_df.loc[sample_id]
            error = sample_rmse[i]

            # Extract features
            sn_ratio = row["signal_to_noise"] if "signal_to_noise" in row else 0.0
            sn_filter = row["SN_filter"] if "SN_filter" in row else 0
            seq_len = len(row["sequence"])

            analysis_data.append(
                {
                    "error": error,
                    "signal_to_noise": sn_ratio,
                    "SN_filter": sn_filter,
                    "seq_length": seq_len,
                }
            )

    analysis_df = pd.DataFrame(analysis_data)
    if not analysis_df.empty:
        correlations = analysis_df.corr()["error"].drop("error")
        print("Failure Analysis (Correlation with Error):")
        print(correlations)

    # 6. Submission
    # Threshold condition
    THRESHOLD = 0.5978901386

    if final_metric < THRESHOLD:
        print(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")
        generate_submission_file(model, device)
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Skipping submission.")


def generate_submission_file(model, device):
    test_dataset = RNADataset(split="test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()
    preds_map = {}

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            ids = batch["id"]

            # Forward pass (Full length 107)
            outputs = model(inputs, pair_indices)
            outputs = outputs.cpu().numpy()

            for i, sample_id in enumerate(ids):
                preds_map[sample_id] = outputs[i]

    submission_data = []
    target_cols = Config.TARGET_COLS

    # Iterate through test dataset IDs to maintain order
    for sample_id in test_dataset.ids:
        pred_matrix = preds_map[sample_id]

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_preds = pred_matrix[seqpos]

            row_dict = {"id_seqpos": row_id}
            for idx, col in enumerate(target_cols):
                row_dict[col] = float(row_preds[idx])

            submission_data.append(row_dict)

    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    run_pipeline()
