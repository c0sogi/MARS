import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from provided library
from library.config import Config
from library.utils import seed_everything
from library.data import get_train_val_datasets, get_test_dataset
from library.model import RNANet
from library.engine import train_model, validate, generate_submission


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # We use the full dataset (debug=False) to ensure we meet the metric threshold.
    # The dataset is small (1.7k), so this fits within the "fast baseline" requirement.
    train_dataset, val_dataset = get_train_val_datasets(
        load_cached_data=True, debug=False
    )

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
    model = RNANet().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 5. Training
    # train_model handles the training loop, validation per epoch, and saving the best model
    best_score_from_train = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        num_epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
    )

    # 6. Final Validation & Metric
    # Load the best model weights
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Calculate final metric on the full validation set
    final_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\nRunning Failure Analysis...")
    model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    # Inference loop for analysis to get raw predictions
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            adj_map = batch["adj_map"].to(device)
            targets = batch["targets"]  # Keep on CPU
            ids = batch["id"]

            outputs = model(inputs, adj_map)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
            all_ids.extend(ids)

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Calculate per-sample MCRMSE
    # Slice to scored sequence length and filter for scored columns
    seq_scored = Config.SEQ_SCORED
    scored_indices = [0, 1, 3]  # reactivity, deg_Mg_pH10, deg_Mg_50C

    p_sliced = preds[:, :seq_scored, scored_indices]
    t_sliced = targets[:, :, scored_indices]

    # RMSE per column per sample: (N, 3)
    rmse_per_col = np.sqrt(np.mean((p_sliced - t_sliced) ** 2, axis=1))

    # MCRMSE per sample: Mean across columns (N,)
    sample_errors = np.mean(rmse_per_col, axis=1)

    # Load metadata
    val_df = pd.read_parquet(Config.VAL_METADATA)

    # Create error dataframe
    error_df = pd.DataFrame({"id": all_ids, "error": sample_errors})

    # Merge with metadata
    analysis_df = val_df.merge(error_df, on="id")

    # Feature Engineering for correlation
    analysis_df["pct_A"] = analysis_df["sequence"].apply(
        lambda x: x.count("A") / len(x)
    )
    analysis_df["pct_U"] = analysis_df["sequence"].apply(
        lambda x: x.count("U") / len(x)
    )
    analysis_df["pct_G"] = analysis_df["sequence"].apply(
        lambda x: x.count("G") / len(x)
    )
    analysis_df["pct_C"] = analysis_df["sequence"].apply(
        lambda x: x.count("C") / len(x)
    )

    # Calculate correlations
    features = ["signal_to_noise", "SN_filter", "pct_A", "pct_U", "pct_G", "pct_C"]
    correlations = analysis_df[features].corrwith(analysis_df["error"])

    print("Correlation between Error and Input Features:")
    print(correlations)

    # 8. Submission
    THRESHOLD = 0.5978901386
    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")
        test_dataset = get_test_dataset(load_cached_data=True, debug=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )
        generate_submission(model, test_loader, device)
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
