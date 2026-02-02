import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

# 1. Setup & Configuration
# Suppress warnings and progress bars for clean output
warnings.filterwarnings("ignore")
# Disable tqdm globally to satisfy "No progress bars" requirement
tqdm.disable = True

# Import provided library components
from library.config import Config
from library.utils import seed_everything, scored_mcrmse
from library.dataset import get_loaders
from library.model import SDIN_CG_BiGRU
from library.train import train_fn, eval_fn, inference


def main():
    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Ensure submission directory exists
    os.makedirs("./submission", exist_ok=True)

    # 2. Data Loading
    # Load cached data. We use the full dataset as it is small enough for fast training.
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Model Initialization
    model = SDIN_CG_BiGRU().to(device)

    # 4. Optimization Setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # Loss Function (MSE)
    criterion = torch.nn.MSELoss()

    # 5. Training Loop
    best_mcrmse = float("inf")

    # Train for the configured number of epochs (50)
    for epoch in range(Config.EPOCHS):
        # Execute one training epoch
        train_loss = train_fn(
            model, train_loader, optimizer, criterion, device, scheduler
        )

        # Evaluate on validation set
        val_mcrmse = eval_fn(model, val_loader, device)

        # Step the scheduler
        scheduler.step()

        # Save the best model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # 6. Final Evaluation
    # Load the best model checkpoint
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Compute final metric on the full validation set
    final_val_metric = eval_fn(model, val_loader, device)
    print(f"Final Validation Metric: {final_val_metric}")

    # 7. Failure Analysis
    # Calculate per-sample error and correlate with metadata
    model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            indices = batch["indices"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            outputs = model(features, indices, mask)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(ids)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Slice to scored columns and positions for error calculation
    target_cols = Config.TARGET_COLS
    scored_cols = Config.SCORED_COLS
    scored_indices = [i for i, col in enumerate(target_cols) if col in scored_cols]

    # Slice: (N, 68, 3)
    p_sliced = all_preds[:, : Config.SEQ_SCORED, scored_indices]
    t_sliced = all_targets[:, : Config.SEQ_SCORED, scored_indices]

    # Compute RMSE per sample (averaged over sequence and channels)
    mse_per_sample = torch.mean((p_sliced - t_sliced) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # Load validation metadata
    val_df = pd.read_parquet(Config.VAL_DATA_PATH)

    # Align metadata with predictions using IDs
    if not np.array_equal(val_df["id"].values, np.array(all_ids)):
        val_df = val_df.set_index("id").loc[all_ids].reset_index()

    val_df["error"] = rmse_per_sample

    # Calculate feature correlations
    # Nucleotide content features
    val_df["pct_A"] = val_df["sequence"].apply(lambda x: x.count("A") / len(x))
    val_df["pct_G"] = val_df["sequence"].apply(lambda x: x.count("G") / len(x))
    val_df["pct_C"] = val_df["sequence"].apply(lambda x: x.count("C") / len(x))
    val_df["pct_U"] = val_df["sequence"].apply(lambda x: x.count("U") / len(x))

    corr_cols = ["signal_to_noise", "SN_filter", "pct_A", "pct_G", "pct_C", "pct_U"]
    correlations = val_df[corr_cols + ["error"]].corr()["error"].drop("error")

    print("Error Correlations:")
    print(correlations)

    # 8. Conditional Submission
    threshold = 0.5978901386
    if final_val_metric < threshold:
        # Generate predictions for test set
        preds, ids = inference(model, test_loader, device)

        submission_rows = []
        # preds shape: (N_samples, 107, 5)
        for i, sample_id in enumerate(ids):
            sample_preds = preds[i]
            for seqpos in range(Config.SEQ_LEN):
                row_id = f"{sample_id}_{seqpos}"
                row_values = sample_preds[seqpos].tolist()

                row_dict = {"id_seqpos": row_id}
                for col_idx, col_name in enumerate(Config.TARGET_COLS):
                    row_dict[col_name] = row_values[col_idx]

                submission_rows.append(row_dict)

        sub_df = pd.DataFrame(submission_rows)
        sub_path = "./submission/submission.csv"
        sub_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")


if __name__ == "__main__":
    main()
