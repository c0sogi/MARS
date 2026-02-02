import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import logging

# Import provided library modules
from library.config import Config
from library.data import get_dataloaders
from library.model import Net
from library.loss import MaskedMSELoss
from library.train import Trainer
from library.utils import seed_everything, get_logger, load_checkpoint
from library.predict import run_prediction


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Override Config for a fast baseline run to meet time constraints
    Config.epochs = 25
    Config.es_patience = 5
    Config.scheduler_patience = 3

    # Ensure reproducibility
    seed_everything(Config.seed)

    # Setup Logger
    # We use a specific logger for the runfile to control output
    logger = get_logger("RunFile", log_file=os.path.join(Config.working_dir, "run.log"))

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    # Load cached data for speed
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    model = Net().to(Config.device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=Config.scheduler_mode,
        factor=Config.scheduler_factor,
        patience=Config.scheduler_patience,
        min_lr=Config.min_lr,
    )

    criterion = MaskedMSELoss()

    # ---------------------------------------------------------
    # 4. Training
    # ---------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=Config.device,
        logger=logger,
    )

    # Train the model
    trainer.fit(epochs=Config.epochs, patience=Config.es_patience)

    # ---------------------------------------------------------
    # 5. Validation & Metric Calculation
    # ---------------------------------------------------------
    # Load the best model saved by the trainer
    load_checkpoint(Config.model_save_path, model, device=Config.device)
    model.eval()

    all_preds = []
    all_targets = []
    all_masks = []

    # Inference on Validation Set
    with torch.no_grad():
        for batch in val_loader:
            seq = batch["sequence"].to(Config.device)
            loop = batch["loop_type"].to(Config.device)
            dist = batch["distance"].to(Config.device)
            tgt = batch["target"].to(Config.device)
            msk = batch["mask"].to(Config.device)

            out = model(seq, loop, dist)

            all_preds.append(out.cpu())
            all_targets.append(tgt.cpu())
            all_masks.append(msk.cpu())

    # Concatenate results
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    masks = torch.cat(all_masks, dim=0)

    # Compute MCRMSE on valid positions
    valid_preds = preds[masks]
    valid_targets = targets[masks]

    # MSE per column (averaging over all valid pixels)
    mse_per_col = torch.mean((valid_preds - valid_targets) ** 2, dim=0)
    # RMSE per column
    rmse_per_col = torch.sqrt(mse_per_col)
    # MCRMSE (Mean of RMSEs)
    mcrmse = torch.mean(rmse_per_col).item()

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {mcrmse}")

    # ---------------------------------------------------------
    # 6. Failure Analysis
    # ---------------------------------------------------------
    # Load validation metadata to access features
    df_val = pd.read_parquet(Config.val_file)

    # Calculate error magnitude per sample (Mean Squared Error of valid positions)
    sample_errors = []
    for i in range(len(preds)):
        p_sample = preds[i]
        t_sample = targets[i]
        m_sample = masks[i]

        if m_sample.sum() > 0:
            diff = p_sample[m_sample] - t_sample[m_sample]
            mse = torch.mean(diff**2).item()
            sample_errors.append(mse)
        else:
            sample_errors.append(0.0)

    df_val["error_magnitude"] = sample_errors

    # Feature Engineering for Correlation Analysis
    df_val["len_A"] = df_val["sequence"].apply(lambda x: x.count("A"))
    df_val["len_G"] = df_val["sequence"].apply(lambda x: x.count("G"))
    df_val["len_C"] = df_val["sequence"].apply(lambda x: x.count("C"))
    df_val["len_U"] = df_val["sequence"].apply(lambda x: x.count("U"))

    features_to_check = [
        "signal_to_noise",
        "SN_filter",
        "len_A",
        "len_G",
        "len_C",
        "len_U",
    ]

    print("Failure Analysis (Correlation with Error Magnitude):")
    for feat in features_to_check:
        if feat in df_val.columns:
            # Drop NaNs to ensure valid correlation calculation
            tmp = df_val[[feat, "error_magnitude"]].dropna()
            if len(tmp) > 0:
                corr = tmp[feat].corr(tmp["error_magnitude"])
                print(f"Correlation between {feat} and error: {corr:.4f}")

    # ---------------------------------------------------------
    # 7. Conditional Submission
    # ---------------------------------------------------------
    threshold = 0.6209375959946717

    if mcrmse < threshold:
        # Generate submission if performance is sufficient
        # run_prediction handles loading model, inference on test, and saving csv
        run_prediction(load_cached_data=True)
    else:
        logger.info(f"Metric {mcrmse} >= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
