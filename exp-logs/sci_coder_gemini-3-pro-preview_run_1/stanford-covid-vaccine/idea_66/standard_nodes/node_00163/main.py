import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy import stats
from library.config import Config
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import train_one_epoch, validate, generate_submission
from library.utils import seed_everything, mcrmse_metric


def run():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    Config.initialize()

    # Override Config for fast execution within time limits
    Config.EPOCHS = (
        15  # Increased slightly to ensure convergence (Cite Lesson 131 context)
    )

    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # Uses cached data if available for speed
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = RNAModel().to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # 5. Training Loop
    best_score = float("inf")
    best_model_path = os.path.join(Config.SUBMISSION_DIR, "best_model.pth")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_score = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

    # 6. Final Evaluation & Failure Analysis
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Run inference on Validation set to gather predictions for analysis
    val_preds_list = []
    val_targets_list = []
    val_ids_list = []

    with torch.no_grad():
        for batch in val_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            struct = batch["struct"].to(device)
            target = batch["target"].to(device)
            ids = batch["id"]

            preds = model(seq, loop, struct)

            val_preds_list.append(preds.cpu())
            val_targets_list.append(target.cpu())
            val_ids_list.extend(ids)

    val_preds = torch.cat(val_preds_list, dim=0)
    val_targets = torch.cat(val_targets_list, dim=0)

    # Compute and print Final Validation Metric
    final_metric = mcrmse_metric(val_preds, val_targets, num_scored=Config.PRED_LEN)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    # Calculate Mean Squared Error per sample (averaged over positions and channels)
    # Slice to scored length first
    p_scored = val_preds[:, : Config.PRED_LEN, :].numpy()
    t_scored = val_targets[:, : Config.PRED_LEN, :].numpy()

    # MSE per sample
    sample_mse = np.mean((p_scored - t_scored) ** 2, axis=(1, 2))

    # Load validation metadata to get features
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.parquet")
    if os.path.exists(val_meta_path):
        df_val = pd.read_parquet(val_meta_path)

        # Map calculated errors to the dataframe
        error_map = dict(zip(val_ids_list, sample_mse))
        df_val["model_error"] = df_val["id"].map(error_map)

        # Drop rows where mapping might have failed (though it shouldn't)
        df_val = df_val.dropna(subset=["model_error"])

        # Correlation 1: Signal to Noise
        if "signal_to_noise" in df_val.columns:
            corr_sn, _ = stats.pearsonr(
                df_val["model_error"], df_val["signal_to_noise"]
            )
            print(f"Correlation (Error vs Signal_to_Noise): {corr_sn}")

        # Correlation 2: GC Content
        # Calculate GC content from sequence
        df_val["gc_content"] = df_val["sequence"].apply(
            lambda x: (x.count("G") + x.count("C")) / len(x) if len(x) > 0 else 0
        )
        corr_gc, _ = stats.pearsonr(df_val["model_error"], df_val["gc_content"])
        print(f"Correlation (Error vs GC Content): {corr_gc}")

    # 7. Submission Generation
    # Threshold check
    THRESHOLD = 0.6176461577

    if final_metric < THRESHOLD:
        generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"Validation metric {final_metric} is not lower than threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    run()
