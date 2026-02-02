import os
import time
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, rle_encode
from library.losses import CompositeLoss
from library.dataset import ContrailDataset, get_transforms
from library.model import UNetPlusPlus
from library.train import train_one_epoch, validate


def failure_analysis(model, dataset, device):
    """
    Performs failure analysis on the validation set by correlating
    prediction error (1 - Dice) with metadata features.
    """
    print("\nPerforming Failure Analysis...")
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    errors = []
    metadata_rows = []

    # Access underlying dataframe for metadata
    df = dataset.df
    current_idx = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            # Predict
            logits = model(images)
            preds = torch.sigmoid(logits)
            preds_bin = (preds > Config.THRESHOLD).float()

            # Calculate per-sample Dice
            # Shape: (B, 1, H, W) -> Sum over spatial dims (1, 2, 3)
            intersection = (preds_bin * targets).sum(dim=(1, 2, 3))
            union = preds_bin.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

            # Dice per image
            dice_scores = (2.0 * intersection + 1e-6) / (union + 1e-6)
            batch_errors = 1.0 - dice_scores.cpu().numpy()
            errors.extend(batch_errors)

            # Collect corresponding metadata
            batch_size = images.size(0)
            batch_indices = range(current_idx, current_idx + batch_size)
            for idx in batch_indices:
                metadata_rows.append(df.iloc[idx])

            current_idx += batch_size

    # Create analysis dataframe
    analysis_df = pd.DataFrame(metadata_rows)
    analysis_df["error"] = errors

    # Calculate correlations
    features = ["timestamp", "row_min", "col_min", "row_size", "col_size"]
    print("Correlation between Error (1-Dice) and Metadata:")
    for feat in features:
        if feat in analysis_df.columns:
            # Ensure numeric
            if pd.api.types.is_numeric_dtype(analysis_df[feat]):
                corr = analysis_df[feat].corr(analysis_df["error"])
                print(f"{feat}: {corr:.4f}")


def generate_submission(model, device):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    print("\nGenerating submission...")

    test_dataset = ContrailDataset(
        metadata_csv_path=Config.TEST_METADATA_PATH,
        stage="test",
        transform=get_transforms("test"),
    )

    loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()
    submission_data = []
    record_ids = test_dataset.df["record_id"].astype(str).tolist()
    current_idx = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # Inference
            logits = model(images)
            preds = torch.sigmoid(logits)

            # Thresholding
            preds_bin = (preds > Config.THRESHOLD).cpu().numpy()

            # Encode
            batch_size = preds_bin.shape[0]
            for i in range(batch_size):
                # Extract (H, W) mask
                mask = preds_bin[i, 0]
                encoded = rle_encode(mask)

                rec_id = record_ids[current_idx]
                submission_data.append({"record_id": rec_id, "encoded_pixels": encoded})
                current_idx += 1

    # Save
    sub_df = pd.DataFrame(submission_data)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # 1. Setup & Config Override
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Override for fast baseline execution
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 32
    TRAIN_SUBSET_SIZE = 2000  # Subsample for speed

    print(f"Starting Fast Baseline Run on {device}")

    # 2. Data Loading
    full_train_dataset = ContrailDataset(
        metadata_csv_path=Config.TRAIN_METADATA_PATH,
        stage="train",
        transform=get_transforms("train"),
    )

    # Subsample training data
    if len(full_train_dataset) > TRAIN_SUBSET_SIZE:
        indices = np.random.choice(
            len(full_train_dataset), TRAIN_SUBSET_SIZE, replace=False
        )
        train_dataset = Subset(full_train_dataset, indices)
    else:
        train_dataset = full_train_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_dataset = ContrailDataset(
        metadata_csv_path=Config.VAL_METADATA_PATH,
        stage="validation",
        transform=get_transforms("validation"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = UNetPlusPlus(
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=Config.ENCODER_WEIGHTS,
        in_channels=Config.IN_CHANNELS,
        classes=Config.CLASSES,
        output_stride=Config.ENCODER_OUTPUT_STRIDE,
    )
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = CompositeLoss(
        weight_focal=Config.WEIGHT_FOCAL,
        weight_dice=Config.WEIGHT_DICE,
        focal_alpha=Config.FOCAL_ALPHA,
        focal_gamma=Config.FOCAL_GAMMA,
    )

    scaler = GradScaler()

    # 4. Training Loop
    best_dice = 0.0

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, scaler, device, epoch
        )

        val_loss, val_dice = validate(val_loader, model, criterion, device)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # 5. Final Evaluation
    # Load best model
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH))

    _, final_metric = validate(val_loader, model, criterion, device)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    failure_analysis(model, val_dataset, device)

    # 7. Submission
    THRESHOLD = 0.5973177358563411
    if final_metric > THRESHOLD:
        generate_submission(model, device)
    else:
        print(
            f"Metric {final_metric} did not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
