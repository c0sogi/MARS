import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy import stats

# Import from provided library files
from library.config import Config
from library.utils import set_seed, get_logger, compute_auc
from library.dataset import TumorDataset, get_transforms
from library.model import ConvNeXtTinyCustom
from library.engine import train_one_epoch, validate, predict_tta

# Initialize Logger
logger = get_logger("baseline")


def evaluate_and_analyze(model, loader, device):
    """
    Evaluates the model on the validation set, computes the final AUC,
    and performs failure analysis by correlating error magnitude with
    image brightness and contrast.
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_brightness = []
    all_contrast = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            # Labels are needed for AUC and Error calc

            logits = model(images)
            probs = torch.sigmoid(logits)

            # Compute metadata stats for failure analysis
            # Brightness: Mean of pixel values
            # Contrast: Std of pixel values
            # Note: Images are normalized, but correlation remains valid
            b = torch.mean(images, dim=(1, 2, 3)).cpu().numpy()
            c = torch.std(images, dim=(1, 2, 3)).cpu().numpy()

            all_labels.append(labels.numpy())
            all_preds.append(probs.cpu().numpy().flatten())
            all_brightness.append(b)
            all_contrast.append(c)

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_brightness = np.concatenate(all_brightness)
    all_contrast = np.concatenate(all_contrast)

    # Compute AUC
    auc = compute_auc(all_labels, all_preds)

    # Failure Analysis: Correlation between Error and Meta-features
    errors = np.abs(all_labels - all_preds)

    # Handle edge case where variance is 0 (unlikely)
    if np.std(all_brightness) > 1e-9:
        corr_b, _ = stats.pearsonr(errors, all_brightness)
    else:
        corr_b = 0.0

    if np.std(all_contrast) > 1e-9:
        corr_c, _ = stats.pearsonr(errors, all_contrast)
    else:
        corr_c = 0.0

    return auc, corr_b, corr_c


def run():
    # -------------------------------------------------------------------------
    # 1. Configuration for Fast Baseline
    # -------------------------------------------------------------------------
    # Override Config defaults to meet runtime constraints
    Config.EPOCHS = 5
    Config.PATIENCE = 3

    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Loading & Subsampling
    # -------------------------------------------------------------------------
    logger.info("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Subsample training data to 15,000 samples for speed
    SAMPLE_SIZE = 15000
    if len(train_df) > SAMPLE_SIZE:
        logger.info(
            f"Subsampling training data from {len(train_df)} to {SAMPLE_SIZE} samples."
        )
        train_df = train_df.sample(n=SAMPLE_SIZE, random_state=Config.SEED).reset_index(
            drop=True
        )

    # Initialize Datasets
    train_dataset = TumorDataset(train_df, transforms=get_transforms("train"))
    val_dataset = TumorDataset(val_df, transforms=get_transforms("val"))
    test_dataset = TumorDataset(test_df, transforms=get_transforms("test"))

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 3. Model Setup
    # -------------------------------------------------------------------------
    logger.info(f"Initializing model: {Config.MODEL_NAME}")
    model = ConvNeXtTinyCustom(pretrained=Config.PRETRAINED)
    model.to(Config.DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    logger.info("Starting training...")
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE
        )

        # Validate (using engine's validate for speed)
        val_loss, val_auc = validate(model, val_loader, criterion, Config.DEVICE)

        # Scheduler Step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.6f}"
        )

        # Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            logger.info(f"New best model saved with AUC: {best_auc:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                logger.info("Early stopping triggered.")
                break

    # -------------------------------------------------------------------------
    # 5. Final Evaluation & Failure Analysis
    # -------------------------------------------------------------------------
    logger.info("Loading best model for final analysis...")
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )

    auc, corr_b, corr_c = evaluate_and_analyze(model, val_loader, Config.DEVICE)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {auc}")

    print("-" * 30)
    print("Failure Analysis (Correlation with Error Magnitude)")
    print(f"Brightness Correlation: {corr_b:.4f}")
    print(f"Contrast Correlation:   {corr_c:.4f}")
    print("-" * 30)

    # -------------------------------------------------------------------------
    # 6. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.9849192531860572

    if auc > THRESHOLD:
        logger.info(
            f"Validation AUC ({auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Use TTA for robust inference
        predictions = predict_tta(model, test_loader, Config.DEVICE)

        submission_df = pd.DataFrame(
            {"id": test_df["id"], "label": predictions.flatten()}
        )

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        logger.info(
            f"Validation AUC ({auc}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    run()
