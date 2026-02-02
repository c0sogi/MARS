import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import provided library components
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import train_epoch, validate, inference

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for fast baseline execution
    Config.EPOCHS = 20

    # Set seeds and device
    seed_everything(Config.SEED)
    device = get_device()

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    # Load dataloaders with caching enabled
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    model = RNAModel().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    # =========================================================================
    # 4. Training Loop
    # =========================================================================
    best_score = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model_runfile.pth")

    for epoch in range(Config.EPOCHS):
        # Train
        _ = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Checkpointing
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

    # =========================================================================
    # 5. Validation Reporting
    # =========================================================================
    print(f"Final Validation Metric: {best_score}")

    # =========================================================================
    # 6. Failure Analysis
    # =========================================================================
    # Load best model for analysis
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Collect predictions and targets for the validation set
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["id"]

            preds = model(inputs, pair_indices)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_ids.extend(ids)

    global_preds = torch.cat(all_preds, dim=0)
    global_targets = torch.cat(all_targets, dim=0)

    # Calculate RMSE per sample on scored columns
    # Config.SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    # Indices in target (5 cols): 0, 1, 3
    target_cols = Config.TARGET_COLS
    scored_cols = Config.SCORED_COLS
    col_indices = [i for i, col in enumerate(target_cols) if col in scored_cols]

    # Slice to scored length (68)
    scored_len = Config.PRED_LEN
    p_scored = global_preds[:, :scored_len, col_indices]
    t_scored = global_targets[:, :scored_len, col_indices]

    # Compute Mean Squared Error per sample (averaged over positions and columns)
    mse_per_sample = torch.mean((p_scored - t_scored) ** 2, dim=(1, 2))
    rmse_per_sample = torch.sqrt(mse_per_sample).numpy()

    # Load Metadata to get features
    val_df = pd.read_parquet(Config.VAL_PATH)
    # Align dataframe with the order in DataLoader
    val_df = val_df.set_index("id").loc[all_ids].reset_index()

    # Construct Analysis DataFrame
    analysis_df = pd.DataFrame(
        {
            "error": rmse_per_sample,
            "signal_to_noise": val_df["signal_to_noise"].values,
            "SN_filter": val_df["SN_filter"].values,
        }
    )

    # Add Sequence Features
    # Nucleotide content
    for char in ["A", "G", "U", "C"]:
        analysis_df[f"pct_{char}"] = (
            val_df["sequence"].apply(lambda s: s.count(char) / len(s)).values
        )

    # Structural content (Paired ratio)
    analysis_df["pct_paired"] = (
        val_df["structure"]
        .apply(lambda s: (s.count("(") + s.count(")")) / len(s))
        .values
    )

    # Calculate Correlations
    corrs = analysis_df.corr()["error"].drop("error").sort_values(ascending=False)

    print("Failure Analysis (Correlation with Error Magnitude):")
    print(corrs)

    # =========================================================================
    # 7. Conditional Submission
    # =========================================================================
    THRESHOLD = 0.5884495377540588

    if best_score < THRESHOLD:
        inference(model, test_loader, device)
    else:
        pass  # Skip submission if threshold not met


if __name__ == "__main__":
    main()
