import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from provided library files
from library.config import Config
from library.utils import set_seed, average_checkpoints, rle_encode, dice_coeff
from library.dataset import ContrailDataset, get_transforms
from library.model import ConvNeXtUNet
from library.loss import HybridLoss
from library.trainer import Trainer


def predict_with_tta(model, images):
    """
    Test Time Augmentation: Average predictions of Original, HFlip, and VFlip.
    """
    # 1. Original
    probs = torch.sigmoid(model(images))

    # 2. Horizontal Flip
    images_h = torch.flip(images, dims=[3])
    probs_h = torch.sigmoid(model(images_h))
    probs += torch.flip(probs_h, dims=[3])

    # 3. Vertical Flip
    images_v = torch.flip(images, dims=[2])
    probs_v = torch.sigmoid(model(images_v))
    probs += torch.flip(probs_v, dims=[2])

    # Average
    probs /= 3.0
    return probs


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Initializing Datasets...")
    # Train dataset with augmentations
    train_dataset = ContrailDataset(
        split="train", transform=get_transforms("train"), debug=Config.DEBUG
    )

    # Validation dataset (no augmentation, just tensor conversion)
    val_dataset = ContrailDataset(
        split="validation", transform=get_transforms("validation"), debug=Config.DEBUG
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

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 3. Model Initialization
    print(f"Initializing Model: {Config.MODEL_NAME} with {Config.BACKBONE} backbone...")
    model = ConvNeXtUNet()
    model = model.to(device)

    # 4. Training Setup
    criterion = HybridLoss(bce_weight=0.5, dice_weight=0.5)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
    )

    # 5. Training Loop
    trainer.fit()

    # 6. Model Averaging (Convergence-Aware)
    print("Performing Convergence-Aware Model Averaging...")
    # trainer.top_k_checkpoints contains tuples of (score, path)
    # We filter for checkpoints saved after START_SAVING_EPOCH

    # Extract paths from the top_k list
    best_ckpt_paths = [ckpt[1] for ckpt in trainer.top_k_checkpoints]

    # Filter paths based on epoch number in filename (defensive coding)
    # Filename format: checkpoint_epoch_{epoch}_dice_{score}.pth
    final_ckpt_paths = []
    for path in best_ckpt_paths:
        try:
            filename = os.path.basename(path)
            epoch_str = filename.split("_")[2]
            epoch = int(epoch_str)
            if epoch >= Config.START_SAVING_EPOCH:
                final_ckpt_paths.append(path)
        except Exception as e:
            print(f"Skipping checkpoint {path} due to parsing error: {e}")

    if not final_ckpt_paths:
        print("No checkpoints found meeting criteria. Using best available.")
        final_ckpt_paths = [trainer.top_k_checkpoints[0][1]]

    print(f"Averaging {len(final_ckpt_paths)} checkpoints: {final_ckpt_paths}")

    averaged_weights_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    avg_state_dict = average_checkpoints(
        final_ckpt_paths, output_path=averaged_weights_path
    )

    # Load averaged weights
    model.load_state_dict(avg_state_dict)

    # 7. Final Validation & Failure Analysis
    print("Running Final Validation on Averaged Model...")
    model.eval()

    # Metrics for Global Dice
    total_intersection = 0.0
    total_union = 0.0

    # Metrics for Failure Analysis
    per_sample_dice = []

    # We need to align predictions with metadata.
    # The val_loader order matches val_dataset which matches val_dataset.df

    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)

            # Use TTA for validation
            preds = predict_with_tta(model, images)

            # Binary masks for Dice calculation
            pred_binary = (preds > Config.THRESHOLD).float()

            # Global Dice Accumulation
            # Flatten per batch
            p_flat = pred_binary.view(pred_binary.size(0), -1)
            t_flat = masks.view(masks.size(0), -1)

            intersection = (p_flat * t_flat).sum(dim=1)
            union = p_flat.sum(dim=1) + t_flat.sum(dim=1)

            total_intersection += intersection.sum().item()
            total_union += union.sum().item()

            # Per-sample Dice for Failure Analysis
            # Dice = 2*I / (U + smooth)
            smooth = 1e-6
            batch_dice = (2.0 * intersection + smooth) / (union + smooth)
            per_sample_dice.extend(batch_dice.cpu().numpy().tolist())

    # Compute Global Dice
    final_metric = (2.0 * total_intersection) / (total_union + 1e-6)

    print(f"Final Validation Metric: {final_metric:.16f}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    if len(per_sample_dice) == len(val_dataset.df):
        # Add dice to dataframe
        analysis_df = val_dataset.df.copy()
        analysis_df["dice"] = per_sample_dice
        analysis_df["error"] = 1.0 - analysis_df["dice"]

        # Calculate correlations
        # We look for correlation between Error (1-Dice) and metadata
        correlations = {}
        meta_cols = ["timestamp", "row_min", "col_min"]

        for col in meta_cols:
            if col in analysis_df.columns:
                corr = analysis_df["error"].corr(analysis_df[col])
                correlations[col] = corr

        print("Correlation between Model Error (1-Dice) and Metadata:")
        for col, corr in correlations.items():
            print(f"  {col}: {corr:.4f}")

        # Interpretation
        max_corr_col = max(correlations, key=lambda k: abs(correlations[k]))
        print(
            f"Strongest error correlation is with {max_corr_col} ({correlations[max_corr_col]:.4f})"
        )
    else:
        print(
            "Warning: Mismatch between predictions and metadata length. Skipping detailed analysis."
        )

    # 8. Submission
    THRESHOLD_SCORE = 0.6131601379732645

    if final_metric > THRESHOLD_SCORE:
        print(
            f"\nValidation metric {final_metric:.6f} > {THRESHOLD_SCORE}. Generating submission..."
        )

        test_dataset = ContrailDataset(
            split="test",
            transform=get_transforms("test"),
            debug=False,  # Always predict on full test set
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        submission_rows = []

        with torch.no_grad():
            for images, record_ids in test_loader:
                images = images.to(device)

                # Inference with TTA
                preds = predict_with_tta(model, images)

                # Convert to binary mask
                pred_binary = (preds > Config.THRESHOLD).cpu().numpy()

                # Encode
                for i in range(len(record_ids)):
                    # Extract single mask: (1, H, W) -> (H, W)
                    mask = pred_binary[i, 0, :, :]
                    rle = rle_encode(mask)
                    submission_rows.append(
                        {"record_id": record_ids[i], "encoded_pixels": rle}
                    )

        # Create DataFrame and save
        sub_df = pd.DataFrame(submission_rows)
        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        sub_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nValidation metric {final_metric:.6f} <= {THRESHOLD_SCORE}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
