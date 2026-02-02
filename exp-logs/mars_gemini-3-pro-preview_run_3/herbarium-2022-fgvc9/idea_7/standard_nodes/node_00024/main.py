import os
import sys
import torch
import pandas as pd
import numpy as np
import logging

# Append current directory to system path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library import utils, data, model, train, inference


def main():
    # 1. Setup and Logging
    utils.seed_everything(Config.SEED)
    logger = utils.get_logger("runfile")
    logger.info("Starting runfile execution...")

    # 2. Configure for Fast Baseline
    # Override Config values to meet time constraints (max 2 hours)
    # We use a subset of data and reduced epochs.

    Config.STAGE1_EPOCHS = 2
    Config.STAGE2_EPOCHS = 1

    # Subset size: 200,000 samples (approx 37% of full train data)
    # This provides a good balance between training speed and model performance.
    subset_size = 200000

    logger.info(f"Creating training subset of {subset_size} samples...")
    full_train_df = pd.read_csv(Config.TRAIN_CSV)

    if len(full_train_df) > subset_size:
        # Random sample to create subset
        train_subset = full_train_df.sample(
            n=subset_size, random_state=Config.SEED
        ).reset_index(drop=True)
    else:
        train_subset = full_train_df

    # Save subset to working directory and update Config to use it
    subset_path = os.path.join(Config.WORK_DIR, "train_subset.csv")
    train_subset.to_csv(subset_path, index=False)
    Config.TRAIN_CSV = subset_path

    logger.info(f"Config updated: TRAIN_CSV={Config.TRAIN_CSV}")
    logger.info(f"Config updated: STAGE1_EPOCHS={Config.STAGE1_EPOCHS}")
    logger.info(f"Config updated: STAGE2_EPOCHS={Config.STAGE2_EPOCHS}")

    # 3. Stage 1 Training (224x224)
    logger.info("=== Stage 1: Feature Learning (224x224) ===")
    train_loader_s1, val_loader_s1, _ = data.get_dataloaders(
        img_size=Config.STAGE1_IMG_SIZE,
        batch_size=Config.STAGE1_BATCH_SIZE,
        debug=False,
    )

    # Initialize model (pretrained backbone)
    net = model.HierarchicalEfficientNet(pretrained=True)

    # Train Stage 1
    net = train.fit(
        model=net,
        train_loader=train_loader_s1,
        val_loader=val_loader_s1,
        epochs=Config.STAGE1_EPOCHS,
        checkpoint_dir=Config.CHECKPOINT_DIR_STAGE1,
        stage_name="Stage 1",
    )

    # 4. Stage 2 Training (384x384)
    logger.info("=== Stage 2: Fine-Grained Refinement (384x384) ===")
    # Get dataloaders with higher resolution
    train_loader_s2, val_loader_s2, test_loader = data.get_dataloaders(
        img_size=Config.STAGE2_IMG_SIZE,
        batch_size=Config.STAGE2_BATCH_SIZE,
        debug=False,
    )

    # Train Stage 2 (continues from Stage 1 best model)
    net = train.fit(
        model=net,
        train_loader=train_loader_s2,
        val_loader=val_loader_s2,
        epochs=Config.STAGE2_EPOCHS,
        checkpoint_dir=Config.CHECKPOINT_DIR_STAGE2,
        stage_name="Stage 2",
    )

    # 5. Final Validation and Metric Calculation
    logger.info("=== Final Validation ===")
    device = torch.device(Config.DEVICE)
    criterion = train.HierarchicalLoss()

    # Validate on the full validation set
    val_loss, val_f1 = train.validate(net, val_loader_s2, criterion, device)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {val_f1}")

    # 6. Failure Analysis
    logger.info("=== Failure Analysis ===")
    net.eval()
    all_preds = []
    all_targets = []

    # Collect predictions
    with torch.no_grad():
        for images, targets in val_loader_s2:
            images = images.to(device)
            outputs = net(images)
            preds = torch.argmax(outputs["species"], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets["species"].cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate errors (1 for error, 0 for correct)
    errors = (all_preds != all_targets).astype(int)

    # Get metadata for correlation analysis
    # val_loader_s2.dataset.df corresponds to the validation dataframe in order
    val_df = val_loader_s2.dataset.df

    # Feature 1: File Size
    # Compute file sizes for validation images
    file_sizes = []
    for rel_path in val_df["file_path"]:
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        try:
            file_sizes.append(os.path.getsize(full_path))
        except:
            file_sizes.append(0)

    # Feature 2: Class Frequency
    # Use frequency from the training subset used for training
    train_counts = train_subset["category_id"].value_counts().to_dict()
    class_freqs = [train_counts.get(cat, 0) for cat in val_df["category_id"]]

    # Calculate Correlations
    if len(errors) == len(file_sizes) and len(errors) == len(class_freqs):
        # Handle potential constant input (std=0) which causes NaN in corrcoef
        if np.std(errors) == 0:
            corr_size = 0.0
            corr_freq = 0.0
        else:
            corr_size = np.corrcoef(errors, file_sizes)[0, 1]
            corr_freq = np.corrcoef(errors, class_freqs)[0, 1]

        print(f"Correlation between Error and File Size: {corr_size}")
        print(f"Correlation between Error and Class Frequency: {corr_freq}")
    else:
        logger.warning("Mismatch in data lengths for failure analysis.")

    # 7. Submission Generation
    threshold = 0.5930838412243743
    if val_f1 > threshold:
        logger.info(
            f"Validation F1 ({val_f1}) > Threshold ({threshold}). Generating submission..."
        )

        # Run inference using the best model from Stage 2
        inference.run_inference(
            checkpoint_path=os.path.join(
                Config.CHECKPOINT_DIR_STAGE2, "best_model.pth"
            ),
            batch_size=Config.STAGE2_BATCH_SIZE,
            img_size=Config.STAGE2_IMG_SIZE,
            device=Config.DEVICE,
            debug=False,
        )
    else:
        logger.info(
            f"Validation F1 ({val_f1}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
