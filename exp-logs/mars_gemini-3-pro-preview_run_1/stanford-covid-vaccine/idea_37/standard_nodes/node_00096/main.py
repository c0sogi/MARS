import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

from library.config import (
    WORKING_DIR,
    MODEL_PATH,
    SUBMISSION_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SEQ_SCORED,
    SEQ_LEN,
    SUBMISSION_COLS,
    SEED,
    VAL_METADATA,
)
from library.utils import seed_everything, calculate_mcrmse
from library.data import get_dataset
from library.model import RNANet, HomoscedasticLoss
from library.train import train_one_epoch, predict

# Configuration for this run
EPOCHS = 15  # Reduced for fast baseline execution


def get_val_preds_and_targets(model, loader, device):
    """
    Runs inference on validation set and returns predictions and targets
    for analysis.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["pair_dist"].to(device)
            target = batch["target"].cpu().numpy()

            # Forward pass
            pred_val, _ = model(seq, loop, dist)

            # Slice to scored positions
            pred_val_scored = pred_val[:, :SEQ_SCORED, :].cpu().numpy()

            all_preds.append(pred_val_scored)
            all_targets.append(target)

    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)


def perform_failure_analysis(val_preds, val_targets, val_ids):
    """
    Analyzes correlation between model error and input features.
    """
    print("\n=== Failure Analysis ===")

    # 1. Calculate Error per Sample (RMSE averaged over columns)
    # val_preds: [N, 68, 3], val_targets: [N, 68, 3]
    mse_per_sample = np.mean((val_preds - val_targets) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # 2. Load Metadata for features
    df_val = pd.read_parquet(VAL_METADATA)
    # Ensure alignment
    df_val = df_val.set_index("id").loc[val_ids].reset_index()

    # 3. Extract Features
    features = {}

    # Signal to Noise
    if "signal_to_noise" in df_val.columns:
        features["Signal_to_Noise"] = df_val["signal_to_noise"].values

    # Sequence Length (Constant 107, but good to check)
    features["Seq_Length"] = df_val["sequence"].apply(len).values

    # GC Content
    features["GC_Content"] = (
        df_val["sequence"]
        .apply(lambda s: (s.count("G") + s.count("C")) / len(s))
        .values
    )

    # A Content
    features["A_Content"] = (
        df_val["sequence"].apply(lambda s: s.count("A") / len(s)).values
    )

    # Unpaired bases count (dots in structure)
    features["Unpaired_Ratio"] = (
        df_val["structure"].apply(lambda s: s.count(".") / len(s)).values
    )

    # 4. Compute Correlations
    print(f"{'Feature':<20} | {'Correlation with Error':<20}")
    print("-" * 45)
    for name, values in features.items():
        if len(np.unique(values)) > 1:
            corr, _ = pearsonr(rmse_per_sample, values)
            print(f"{name:<20} | {corr:.4f}")
        else:
            print(f"{name:<20} | N/A (Constant)")
    print("-" * 45)


def main():
    # 1. Setup
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # 2. Data Loading
    print("Loading datasets...")
    train_ds = get_dataset("train", load_cached_data=True)
    val_ds = get_dataset("val", load_cached_data=True)
    test_ds = get_dataset("test", load_cached_data=True)

    # Pin memory for faster transfer to GPU
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    # 3. Model & Optimizer
    model = RNANet().to(device)
    criterion = HomoscedasticLoss().to(device)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 4. Training Loop
    best_mcrmse = float("inf")
    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        train_loss, train_mse_val, train_mse_unc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validation
        val_preds, val_targets = get_val_preds_and_targets(model, val_loader, device)

        # Calculate MCRMSE
        # Flatten for calculation: [N*68, 3]
        flat_preds = val_preds.reshape(-1, 3)
        flat_targets = val_targets.reshape(-1, 3)
        val_mcrmse = calculate_mcrmse(flat_targets, flat_preds)

        scheduler.step()

        print(
            f"Epoch {epoch+1:02d} | Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"Training complete.")

    # 5. Final Evaluation
    print("Loading best model for evaluation...")
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    val_preds, val_targets = get_val_preds_and_targets(model, val_loader, device)
    flat_preds = val_preds.reshape(-1, 3)
    flat_targets = val_targets.reshape(-1, 3)
    final_metric = calculate_mcrmse(flat_targets, flat_preds)

    print(f"Final Validation Metric: {final_metric:.12f}")

    # 6. Failure Analysis
    perform_failure_analysis(val_preds, val_targets, val_ds.ids)

    # 7. Submission
    THRESHOLD = 0.6199890971183777
    if final_metric < THRESHOLD:
        print(
            f"Validation metric {final_metric:.6f} < {THRESHOLD}. Generating submission..."
        )

        predictions = predict(model, test_loader, device)  # [N, 107, 5]

        submission_rows = []
        test_ids = test_ds.ids

        for i, sample_id in enumerate(test_ids):
            sample_pred = predictions[i]
            for seq_pos in range(SEQ_LEN):
                row_id = f"{sample_id}_{seq_pos}"
                row_values = sample_pred[seq_pos].tolist()
                submission_rows.append([row_id] + row_values)

        submission_df = pd.DataFrame(
            submission_rows, columns=["id_seqpos"] + SUBMISSION_COLS
        )
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {final_metric:.6f} >= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
