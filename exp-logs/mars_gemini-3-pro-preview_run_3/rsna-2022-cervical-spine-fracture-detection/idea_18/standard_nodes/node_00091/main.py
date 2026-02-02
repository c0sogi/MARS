import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided library
from library.config import Config
from library.utils import seed_everything, setup_logger
from library.dataset import CervicalSpineDataset
from library.model import DynamicDepthConvNeXt
from library.loss import ImplicitWeightedLoss
from library.engine import fit
from library.inference import run_inference


def main():
    # --- 1. Setup ---
    seed_everything(Config.SEED)
    logger = setup_logger(os.path.join(Config.WORKING_DIR, "train.log"))
    device = Config.DEVICE

    # Configure for fast baseline execution within time limits
    # Limit data size and epochs to ensure completion within 29 minutes
    Config.set_debug_mode(debug=True, data_size=250, epochs=2)

    print(f"Running on device: {device}")

    # --- 2. Data Loading ---
    print("Initializing Datasets...")
    # Load cached data is True to utilize pre-sorted paths if available
    train_dataset = CervicalSpineDataset(
        mode="train", load_cached_data=True, seq_length=Config.SEQ_LENGTH
    )
    val_dataset = CervicalSpineDataset(
        mode="val", load_cached_data=True, seq_length=Config.SEQ_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- 3. Model Setup ---
    print("Initializing Model...")
    model = DynamicDepthConvNeXt(
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
    )
    model.to(device)

    # --- 4. Training ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler setup
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(total_steps * Config.T_MAX_COEF), eta_min=Config.MIN_LR
    )

    loss_fn = ImplicitWeightedLoss()

    print("Starting Training...")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        epochs=Config.EPOCHS,
        patience=2,  # Strict patience for baseline
        save_path=os.path.join(Config.WORKING_DIR, "best_model.pth"),
    )

    # --- 5. Validation & Metric Calculation ---
    print("Performing Final Validation...")
    # Load best model weights
    if os.path.exists(os.path.join(Config.WORKING_DIR, "best_model.pth")):
        model.load_state_dict(
            torch.load(os.path.join(Config.WORKING_DIR, "best_model.pth"))
        )

    model.eval()

    val_losses = []
    val_depths = []

    total_loss_accum = 0.0
    count = 0

    # Loss function for per-sample calculation (no reduction)
    bce_none = torch.nn.BCEWithLogitsLoss(reduction="none")

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)

            # --- Metric Calculation (Weighted Log Loss) ---
            # The metric is defined by the ImplicitWeightedLoss structure:
            # Loss = BCE_Patient + Mean(BCE_C1...C7)

            # Per sample losses
            loss_c = bce_none(logits[:, :7], targets[:, :7])  # Shape: (B, 7)
            loss_p = bce_none(logits[:, 7], targets[:, 7])  # Shape: (B,)

            # Average over C1-C7 for each sample
            loss_c_mean = loss_c.mean(dim=1)  # Shape: (B,)

            # Total weighted loss per sample
            batch_loss = loss_p + loss_c_mean  # Shape: (B,)

            # Accumulate for final metric
            total_loss_accum += batch_loss.sum().item()
            count += inputs.size(0)

            # --- Failure Analysis Data Collection ---
            val_losses.extend(batch_loss.cpu().numpy().tolist())

            # Retrieve depths (number of slices) for failure analysis
            # Since val_loader is not shuffled, we can map indices to the dataset
            start_idx = i * Config.BATCH_SIZE
            end_idx = start_idx + inputs.size(0)

            for idx in range(start_idx, end_idx):
                if idx < len(val_dataset):
                    row = val_dataset.df.iloc[idx]
                    uid = row["StudyInstanceUID"]
                    # Get depth from cache (list of filenames)
                    files = val_dataset.sorted_paths_map.get(uid, [])
                    val_depths.append(len(files))
                else:
                    val_depths.append(0)

    final_metric = total_loss_accum / count if count > 0 else 0.0
    # Print exact metric format required
    print(f"Final Validation Metric: {final_metric}")

    # --- 6. Failure Analysis ---
    print("Performing Failure Analysis...")
    if len(val_losses) > 1 and len(val_depths) > 1:
        # Calculate correlation between Error and Depth
        corr, _ = pearsonr(val_losses, val_depths)
        print(
            f"Correlation between Error (Loss) and Volume Depth (Slices): {corr:.10f}"
        )

        # Identify worst case
        high_loss_idx = np.argmax(val_losses)
        print(
            f"Max Loss Sample Index: {high_loss_idx}, Loss: {val_losses[high_loss_idx]:.4f}, Depth: {val_depths[high_loss_idx]}"
        )
    else:
        print("Insufficient data for failure analysis.")

    # --- 7. Submission Logic ---
    threshold = 0.06429807151236185
    print(
        f"Checking submission criteria: Metric {final_metric} < Threshold {threshold}?"
    )

    if final_metric < threshold:
        print("Criteria met. Generating submission...")
        run_inference(
            model_path=os.path.join(Config.WORKING_DIR, "best_model.pth"),
            output_path=Config.SUBMISSION_PATH,
            batch_size=Config.BATCH_SIZE,
            device=device,
        )
    else:
        print("Criteria NOT met. Skipping submission generation.")


if __name__ == "__main__":
    main()
