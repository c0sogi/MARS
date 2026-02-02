import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library
from library.config import Config
from library import utils, data, model, train

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    utils.seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = data.get_dataloaders(
        load_cached_data=True, debug=Config.DEBUG
    )

    # 3. Model Initialization
    print("Initializing model...")
    net = model.HC_SDBR_BiGRU().to(device)

    # 4. Optimization Setup
    criterion = utils.MCRMSELoss()
    optimizer = optim.AdamW(
        net.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 5. Training
    print("Starting training...")
    trainer = train.Trainer(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    trainer.fit(epochs=Config.EPOCHS)

    # 6. Evaluation & Failure Analysis
    print("\n==== Evaluation & Failure Analysis ====")

    # Load best model for analysis
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading best model from {Config.BEST_MODEL_PATH}")
        net.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model not found. Using current weights.")

    net.eval()

    # Collect all validation predictions and targets
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            outputs = net(inputs, pair_indices, pair_mask)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(ids)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate Final Metric
    val_metric = utils.calculate_metric(all_preds, all_targets)
    print(f"Final Validation Metric: {val_metric}")

    # Failure Analysis: Correlation of Error with Features
    # 1. Calculate RMSE per sample (on scored columns/positions)
    seq_scored = Config.SEQ_SCORED
    scored_cols_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_TARGETS
    ]

    preds_sliced = all_preds[:, :seq_scored, scored_cols_indices]
    targets_sliced = all_targets[:, :seq_scored, scored_cols_indices]

    # MSE per sample: Average over length and columns
    mse_per_sample = torch.mean((preds_sliced - targets_sliced) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # 2. Load Metadata features
    try:
        val_df = pd.read_parquet(Config.VAL_METADATA)
        # Align dataframe with the order of IDs in the loader
        val_df = val_df.set_index("id").loc[all_ids].reset_index()

        # 3. Construct Analysis DataFrame
        analysis_df = pd.DataFrame(
            {
                "rmse": rmse_per_sample,
                "signal_to_noise": val_df["signal_to_noise"].values,
                "SN_filter": val_df["SN_filter"].values,
            }
        )

        # Add sequence composition features
        analysis_df["pct_A"] = val_df["sequence"].apply(lambda s: s.count("A") / len(s))
        analysis_df["pct_G"] = val_df["sequence"].apply(lambda s: s.count("G") / len(s))
        analysis_df["pct_C"] = val_df["sequence"].apply(lambda s: s.count("C") / len(s))
        analysis_df["pct_U"] = val_df["sequence"].apply(lambda s: s.count("U") / len(s))

        # 4. Compute Correlations
        correlations = analysis_df.corr()["rmse"].sort_values(ascending=False)
        print("\nCorrelation of RMSE with features:")
        print(correlations)

    except Exception as e:
        print(f"Failure analysis skipped due to error: {e}")

    # 7. Conditional Submission
    THRESHOLD = 0.5884495377540588

    if val_metric < THRESHOLD:
        print(
            f"\nValidation metric {val_metric} < {THRESHOLD}. Generating submission..."
        )
        # generate_submission handles loading the best model internally, but we already loaded it.
        # It's fine, it will reload or we can pass the loaded net.
        # The function signature is generate_submission(model_instance, test_loader, device)
        train.generate_submission(net, test_loader, device)
    else:
        print(f"\nValidation metric {val_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
