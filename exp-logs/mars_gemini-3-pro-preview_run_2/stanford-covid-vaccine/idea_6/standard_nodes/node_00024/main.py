import os
import numpy as np
import pandas as pd
import torch
import scipy.stats
from torch.utils.data import DataLoader

from library.config import Config
from library.data import process_data, RNADataset
from library.model import OptimizedHybridNet
from library.train import train_one_epoch, validate, generate_submission, set_seed
from library.utils import mcrmse_loss


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Override Config for fast baseline if necessary, but defaults (25 epochs) are fine for this dataset size.
    # We will use 20 epochs to ensure it finishes well within the time limit while converging.
    EPOCHS = 20
    BATCH_SIZE = Config.BATCH_SIZE

    print(f"Running on device: {device}")

    # 2. Data Loading
    # Load processed data
    train_inputs, train_targets, train_ids = process_data(
        Config.TRAIN_CSV, "train", load_cached_data=True
    )
    val_inputs, val_targets, val_ids = process_data(
        Config.VAL_CSV, "val", load_cached_data=True
    )

    # Create Datasets
    train_dataset = RNADataset(train_inputs, train_targets, train_ids, mode="train")
    val_dataset = RNADataset(val_inputs, val_targets, val_ids, mode="val")

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = OptimizedHybridNet(Config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Mask for scoring: only first 68 positions are valid
    mask = torch.zeros((1, Config.SEQ_LEN), device=device)
    mask[:, : Config.PRED_LEN] = 1.0

    # 4. Training Loop
    best_loss = float("inf")
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, mask)
        val_loss = validate(model, val_loader, device, mask)

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Val MCRMSE: {best_loss:.6f}")

    # 5. Final Evaluation
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Calculate Final Metric
    final_metric = validate(model, val_loader, device, mask)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Load metadata for validation set
    val_df = pd.read_csv(Config.VAL_CSV)
    # Ensure alignment: the DataLoader preserves order because shuffle=False
    # We can assume val_df rows correspond to val_loader outputs in order

    # Calculate per-sample error
    scored_indices = [0, 1, 3]  # reactivity, deg_Mg_pH10, deg_Mg_50C
    per_sample_errors = []

    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)

            # Calculate error per sample
            # preds, targets: [Batch, Seq, 5]
            # mask: [1, Seq] -> [Batch, Seq]
            batch_mask = mask.expand(inputs.size(0), -1)

            # Iterate over batch
            for i in range(inputs.size(0)):
                sample_loss = 0.0
                count = 0
                for col_idx in scored_indices:
                    p = preds[i, :, col_idx]
                    t = targets[i, :, col_idx]
                    m = batch_mask[i]

                    mse = ((p - t) ** 2) * m
                    rmse = torch.sqrt(mse.sum() / (m.sum() + 1e-8))
                    sample_loss += rmse.item()
                    count += 1

                per_sample_errors.append(sample_loss / count)

    per_sample_errors = np.array(per_sample_errors)

    # Correlate with metadata
    if "signal_to_noise" in val_df.columns:
        sn = val_df["signal_to_noise"].values
        # Ensure lengths match (drop any extra rows if loader dropped last batch, though standard loader shouldn't)
        n = len(per_sample_errors)
        sn = sn[:n]

        corr_sn, _ = scipy.stats.pearsonr(per_sample_errors, sn)
        print(f"Correlation (Error vs Signal_to_Noise): {corr_sn:.4f}")

    if "SN_filter" in val_df.columns:
        sn_filter = val_df["SN_filter"].values[: len(per_sample_errors)]
        corr_filter, _ = scipy.stats.pearsonr(per_sample_errors, sn_filter)
        print(f"Correlation (Error vs SN_filter): {corr_filter:.4f}")

    # 7. Submission
    THRESHOLD = 0.6477736930052439
    if final_metric < THRESHOLD:
        generate_submission(best_model_path, device)
    else:
        print(
            f"Validation metric {final_metric} is not lower than threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
