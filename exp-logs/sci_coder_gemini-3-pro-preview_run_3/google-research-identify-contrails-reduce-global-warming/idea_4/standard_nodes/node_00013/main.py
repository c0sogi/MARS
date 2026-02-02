import os
import torch
import pandas as pd
import numpy as np
import random
from torch.utils.data import DataLoader, Subset

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, get_device
from library.dataset import ContrailDataset, get_transforms
from library.model import ContrailUNet
from library.loss import BCEDiceLoss
from library.engine import fit, validate, predict_and_submit
from library.checkpointing import CheckpointManager, average_weights


def perform_failure_analysis(model, loader, device, metadata_path):
    """
    Calculates per-sample error (1 - Dice) and correlates it with metadata features.
    """
    print("Performing Failure Analysis...")
    model.eval()

    scores = []
    ids = []

    # 1. Calculate Per-Sample Dice
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            record_ids = batch["record_id"]

            # Inference
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Calculate Dice per image in the batch
            # Shape: (B, 1, H, W) -> Sum over (1, 2, 3)
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))

            # Add epsilon to avoid division by zero
            dice = (2.0 * intersection + 1e-6) / (union + 1e-6)

            scores.extend(dice.cpu().numpy())
            ids.extend(record_ids)

    # 2. Merge with Metadata
    if not os.path.exists(metadata_path):
        print(
            f"Metadata file not found at {metadata_path}. Skipping correlation analysis."
        )
        return

    df_meta = pd.read_csv(metadata_path)

    # Ensure ID types match for merging
    df_meta["record_id"] = df_meta["record_id"].astype(str)

    analysis_df = pd.DataFrame(
        {
            "record_id": [str(x) for x in ids],
            "dice": scores,
            "error": [1.0 - s for s in scores],
        }
    )

    merged = pd.merge(analysis_df, df_meta, on="record_id", how="inner")

    if merged.empty:
        print("Merged analysis DataFrame is empty. Check record_id matching.")
        return

    # 3. Calculate Correlations
    print("\nCorrelation of Error (1 - Dice) with Metadata Features:")
    features = ["timestamp", "row_min", "col_min"]

    for feat in features:
        if feat in merged.columns:
            # Drop NaNs just in case
            valid_data = merged[["error", feat]].dropna()
            if not valid_data.empty:
                corr = valid_data["error"].corr(valid_data[feat])
                print(f"{feat}: {corr:.4f}")
            else:
                print(f"{feat}: Insufficient data")
        else:
            print(f"{feat}: Feature not found in metadata")
    print("-" * 30)


def main():
    # --- 1. Configuration & Setup ---
    # Using Full Dataset and Standard U-Net (Cite solution_lesson_node_00011)
    set_seed(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # --- 2. Data Loading ---
    print("Initializing DataLoaders...")

    # Training Data (Full Dataset)
    train_dataset = ContrailDataset(split="train", transform=get_transforms("train"))
    print(f"Training on full dataset: {len(train_dataset)} samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Validation Data (Full)
    val_dataset = ContrailDataset(
        split="validation", transform=get_transforms("validation")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- 3. Model Initialization ---
    print("Initializing Model (Standard U-Net with EfficientNet-B0)...")
    model = ContrailUNet().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)

    checkpoint_manager = CheckpointManager(mode="max", top_k=Config.N_BEST_MODELS)

    # --- 4. Training ---
    print("Starting Training Loop...")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        epochs=Config.EPOCHS,
        checkpoint_manager=checkpoint_manager,
        patience=5,
    )

    # --- 5. Weight Averaging ---
    print("Performing Convergence-Aware Weight Averaging...")
    avg_state_dict = average_weights(threshold_epoch=Config.CONVERGENCE_EPOCH_THRESHOLD)

    if avg_state_dict is not None:
        model.load_state_dict(avg_state_dict)
        print("Loaded averaged weights.")
    else:
        print(
            "Weight averaging failed (no checkpoints met criteria). Loading best single model."
        )
        best_state_dict = checkpoint_manager.load_best()
        if best_state_dict:
            model.load_state_dict(best_state_dict)

    # --- 6. Final Validation ---
    print("Running Final Validation on Hold-out Set...")
    _, final_metric = validate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # --- 7. Failure Analysis ---
    perform_failure_analysis(model, val_loader, device, Config.VALIDATION_METADATA_PATH)

    # --- 8. Submission ---
    SUBMISSION_THRESHOLD = 0.5723259114122367

    if final_metric > SUBMISSION_THRESHOLD:
        print(
            f"Metric ({final_metric:.6f}) exceeds threshold ({SUBMISSION_THRESHOLD:.6f}). Generating submission..."
        )

        test_dataset = ContrailDataset(split="test", transform=get_transforms("test"))
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        predict_and_submit(
            model, test_loader, device, output_path=Config.SUBMISSION_PATH
        )
    else:
        print(
            f"Metric ({final_metric:.6f}) did not exceed threshold ({SUBMISSION_THRESHOLD:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
