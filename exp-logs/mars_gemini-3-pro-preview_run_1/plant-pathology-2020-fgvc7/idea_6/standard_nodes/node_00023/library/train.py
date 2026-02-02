import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import AppleDataset, get_transforms
from library.model import AppleResNet
from library.loss import WeightedSoftCrossEntropy
from library.engine import train_one_epoch, validate_one_epoch

logger = get_logger("train")


def generate_submission(debug: bool = False):
    """
    Generates submission file using trained models and Test-Time Augmentation (TTA).
    """
    logger.info("Starting Inference and Submission Generation...")

    # Load Test Metadata
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    if debug:
        test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        logger.info(f"Debug mode: Inference on {len(test_df)} samples.")

    # Dataset & Loader
    test_dataset = AppleDataset(test_df, transform=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Models from all folds
    models = []
    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"resnet34_fold_{fold}.pth")
        if os.path.exists(model_path):
            model = AppleResNet()
            model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
            model.to(Config.DEVICE)
            model.eval()
            models.append(model)
            logger.info(f"Loaded model for Fold {fold}")
        else:
            logger.warning(f"Model for Fold {fold} not found at {model_path}")

    if not models:
        logger.error("No models found. Skipping submission generation.")
        return

    all_preds = []
    image_ids = []

    # Inference Loop
    with torch.no_grad():
        for data in test_loader:
            images = data["image"].to(Config.DEVICE)
            ids = data["image_id"]
            image_ids.extend(ids)

            # Test-Time Augmentation (TTA)
            # 1. Original
            # 2. Horizontal Flip
            # 3. Vertical Flip
            images_h = torch.flip(images, dims=[3])
            images_v = torch.flip(images, dims=[2])

            batch_preds = []
            for model in models:
                # Get probabilities
                p_orig = torch.softmax(model(images), dim=1)
                p_h = torch.softmax(model(images_h), dim=1)
                p_v = torch.softmax(model(images_v), dim=1)

                # Average TTA for this model
                p_avg = (p_orig + p_h + p_v) / 3.0
                batch_preds.append(p_avg.cpu().numpy())

            # Ensemble: Average predictions across all models
            ensemble_preds = np.mean(batch_preds, axis=0)
            all_preds.append(ensemble_preds)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)

    # Create Submission DataFrame
    # Columns must match Config.CLASS_LABELS order
    submission = pd.DataFrame(all_preds, columns=Config.CLASS_LABELS)
    submission.insert(0, "image_id", image_ids)

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved successfully to {Config.SUBMISSION_PATH}")


def run_training(debug: bool = False):
    """
    Orchestrates the Stratified 5-Fold Cross-Validation training pipeline.

    Args:
        debug (bool): If True, runs on a small subset of data for debugging.
    """
    seed_everything(Config.SEED)
    Config.setup_directories()

    logger.info("Loading metadata for Cross-Validation...")
    # Load only training metadata to avoid leakage into the hold-out validation set
    full_df = pd.read_csv(Config.TRAIN_METADATA_PATH)

    if debug:
        full_df = full_df.sample(
            n=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)
        logger.info(f"Debug mode: Training on {len(full_df)} samples.")

    # Ensure stratify label exists
    if "stratify_label" not in full_df.columns:
        full_df["stratify_label"] = full_df[Config.CLASS_LABELS].idxmax(axis=1)

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(full_df, full_df["stratify_label"])
    ):
        logger.info(f"\n{'='*20} Fold {fold} {'='*20}")

        # Split Data
        df_train = full_df.iloc[train_idx].reset_index(drop=True)
        df_val = full_df.iloc[val_idx].reset_index(drop=True)

        # Datasets & Loaders
        train_dataset = AppleDataset(df_train, transform=get_transforms("train"))
        val_dataset = AppleDataset(df_val, transform=get_transforms("valid"))

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

        # Model
        model = AppleResNet()
        model.to(Config.DEVICE)

        # Optimizer & Scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        # Synchronize T_0 with Config.EPOCHS to ensure full decay cycle (Cite Lesson 00015)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=Config.EPOCHS, T_mult=Config.T_MULT, eta_min=Config.MIN_LR
        )

        # Loss Function
        # Uses default class weights calculation from config/metadata
        criterion = WeightedSoftCrossEntropy()

        # Training Loop
        best_auc = 0.0
        patience = 15
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, Config.DEVICE, epoch
            )
            val_loss, val_auc = validate_one_epoch(
                model, val_loader, criterion, Config.DEVICE
            )

            scheduler.step()

            logger.info(
                f"Fold {fold} | Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc}"
            )

            # Save Best Model
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                save_path = os.path.join(
                    Config.WORKING_DIR, f"resnet34_fold_{fold}.pth"
                )
                torch.save(model.state_dict(), save_path)
                logger.info(
                    f"New best model saved for Fold {fold} with AUC: {best_auc}"
                )
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        fold_scores.append(best_auc)

        # Cleanup to save memory
        del (
            model,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            train_dataset,
            val_dataset,
        )
        torch.cuda.empty_cache()

    logger.info(f"\nCross-Validation Complete.")
    logger.info(f"Fold Scores: {fold_scores}")
    logger.info(f"Mean AUC: {np.mean(fold_scores)}")

    # Generate Submission
    generate_submission(debug)
