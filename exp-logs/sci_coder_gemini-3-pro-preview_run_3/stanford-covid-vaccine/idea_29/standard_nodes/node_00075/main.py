import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import RNARegressor
from library.train import MCRMSELoss, train_one_epoch, validate, generate_submission


def main():
    # 1. Configuration & Setup
    # Using 15 epochs for a fast baseline as requested
    config = Config(num_epochs=15)
    set_seed(config.seed)

    # Ensure submission directory exists
    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=True
    )

    # 3. Model Initialization
    device = config.device
    model = RNARegressor(config).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    criterion = MCRMSELoss()

    # 4. Training Loop
    best_val_score = float("inf")

    # Silent training loop as per requirements (no progress bars)
    for epoch in range(config.num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config
        )
        val_score = validate(model, val_loader, device, config)
        scheduler.step()

        if val_score < best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), config.model_path)

    # 5. Final Validation
    # Load best model
    model.load_state_dict(torch.load(config.model_path, map_location=device))

    # Compute final metric on full validation set
    final_val_score = validate(model, val_loader, device, config)
    print(f"Final Validation Metric: {final_val_score}")

    # 6. Failure Analysis
    # Load validation metadata for correlation analysis
    val_df = pd.read_parquet(config.val_file)

    # Generate predictions on validation set for analysis
    model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            bpp_mask = batch["bpp_mask"].to(device)
            targets = batch["targets"].to(device)
            ids = batch["ids"]

            outputs = model(inputs, bpp_indices, bpp_mask)

            # Slice to scored length
            outputs = outputs[:, : config.seq_scored, :]
            targets = targets[:, : config.seq_scored, :]

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_ids.extend(ids)

    y_pred = np.concatenate(all_preds, axis=0)  # (N, 68, 5)
    y_true = np.concatenate(all_targets, axis=0)  # (N, 68, 5)

    # Calculate error per sample (Mean RMSE across 5 targets)
    # Shape: (N, 68, 5) -> (N, 5) -> (N,)
    mse_per_sample = np.mean((y_true - y_pred) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    analysis_df = pd.DataFrame({"id": all_ids, "error": rmse_per_sample})

    # Merge with metadata
    analysis_df = analysis_df.merge(val_df, on="id", how="left")

    # Feature Engineering
    analysis_df["gc_content"] = analysis_df["sequence"].apply(
        lambda s: (s.count("G") + s.count("C")) / len(s)
    )
    analysis_df["paired_pct"] = analysis_df["structure"].apply(
        lambda s: 1.0 - (s.count(".") / len(s))
    )

    # Calculate Correlations
    print("Failure Analysis (Correlation with Error):")
    features = ["signal_to_noise", "SN_filter", "gc_content", "paired_pct"]
    for feat in features:
        if feat in analysis_df.columns:
            corr = analysis_df["error"].corr(analysis_df[feat])
            print(f"{feat}: {corr:.4f}")

    # 7. Conditional Submission
    threshold = 0.5978901386
    if final_val_score < threshold:
        submission_df = generate_submission(model, test_loader, device, config)
        submission_df.to_csv(submission_path, index=False)


if __name__ == "__main__":
    main()
