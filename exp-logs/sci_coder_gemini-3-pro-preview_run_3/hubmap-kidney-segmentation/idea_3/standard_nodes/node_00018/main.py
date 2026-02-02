import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Import from provided library files
from library.utils import set_seed, get_logger, compute_dice_score
from library.dataset import prepare_datasets
from library.model import build_model
from library.losses import DeepSupervisionLoss
from library.training import Trainer
from library.inference import run_inference


def main():
    # 1. Setup
    logger = get_logger("RunFile")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")

    # Hyperparameters
    TILE_SIZE = 1024
    BATCH_SIZE = 2  # Small batch size for large tiles/model
    ACCUMULATION_STEPS = 16  # Simulate larger batch size (32)
    EPOCHS = 1  # Fast baseline
    LR = 1e-4
    MAX_TRAIN_TILES = 4000
    MAX_VAL_TILES = 1000

    # 2. Prepare Data
    logger.info("Preparing datasets...")
    # Load cached data if available, otherwise generate
    train_dataset, val_dataset = prepare_datasets(
        tile_size=TILE_SIZE,
        overlap=0.25,  # Reduced overlap for training efficiency
        do_normalization=True,
        load_cached_data=True,
        debug=False,
    )

    # Subsample datasets to ensure runtime < 2 hours
    if len(train_dataset.tile_df) > MAX_TRAIN_TILES:
        logger.info(f"Subsampling training data to {MAX_TRAIN_TILES} tiles.")
        train_dataset.tile_df = train_dataset.tile_df.sample(
            n=MAX_TRAIN_TILES, random_state=42
        ).reset_index(drop=True)

    if len(val_dataset.tile_df) > MAX_VAL_TILES:
        logger.info(f"Subsampling validation data to {MAX_VAL_TILES} tiles.")
        val_dataset.tile_df = val_dataset.tile_df.sample(
            n=MAX_VAL_TILES, random_state=42
        ).reset_index(drop=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Build Model
    logger.info("Building model...")
    model = build_model(encoder_name="convnext_base", in_channels=3, classes=1).to(
        device
    )

    # 4. Training Configuration
    criterion = DeepSupervisionLoss(weights=[1.0, 0.5, 0.25])
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    config = {"accumulation_steps": ACCUMULATION_STEPS, "patience": 3}

    # 5. Train
    trainer = Trainer(model, optimizer, scheduler, criterion, device, scaler, config)
    save_path = "./working/best_model.pth"

    logger.info("Starting training...")
    trainer.fit(train_loader, val_loader, EPOCHS, save_path)

    # 6. Final Validation & Failure Analysis
    logger.info("Loading best model for analysis...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    val_dice_scores = []

    # We need to map predictions back to metadata for failure analysis.
    # val_loader has shuffle=False, so order matches val_dataset.tile_df
    val_metadata_df = val_dataset.tile_df.copy()

    # Load global metadata to merge features
    global_meta_df = pd.read_csv("./metadata/val.csv")

    logger.info("Running validation inference...")
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.cpu().numpy()

            # Inference
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds_bin = (preds > 0.5).float().cpu().numpy()

            # Compute Dice per sample
            for i in range(len(masks)):
                # masks is (B, H, W), preds_bin is (B, 1, H, W)
                score = compute_dice_score(masks[i], preds_bin[i, 0])
                val_dice_scores.append(score)

    # Calculate Final Metric
    final_metric = np.mean(val_dice_scores)
    print(f"Final Validation Metric: {final_metric:.16f}")

    # Failure Analysis
    logger.info("Performing Failure Analysis...")
    val_metadata_df["dice"] = val_dice_scores
    val_metadata_df["error"] = 1.0 - val_metadata_df["dice"]

    # Merge with global metadata to get patient/image features
    # val_metadata_df has 'id' column
    analysis_df = pd.merge(val_metadata_df, global_meta_df, on="id", how="left")

    features_to_analyze = [
        "age",
        "weight_kilograms",
        "bmi_kg/m^2",
        "percent_cortex",
        "percent_medulla",
    ]

    print("Failure Analysis - Correlation with Error Magnitude:")
    for feature in features_to_analyze:
        if feature in analysis_df.columns:
            # Drop NaNs for correlation calculation
            valid_data = analysis_df[[feature, "error"]].dropna()
            if len(valid_data) > 1:
                corr = np.corrcoef(valid_data[feature], valid_data["error"])[0, 1]
                print(f"{feature}: {corr:.4f}")
            else:
                print(f"{feature}: Not enough data")
        else:
            print(f"{feature}: Feature not found")

    # 7. Submission
    THRESHOLD = 0.9132
    if final_metric > THRESHOLD:
        logger.info(
            f"Validation metric {final_metric:.4f} > {THRESHOLD}. Generating submission..."
        )
        run_inference(
            model_path=save_path,
            output_path="./submission/submission.csv",
            tile_size=TILE_SIZE,
            overlap=0.5,
            batch_size=BATCH_SIZE,
            do_normalization=True,
        )
    else:
        logger.info(
            f"Validation metric {final_metric:.4f} <= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
