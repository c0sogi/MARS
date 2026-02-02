import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import logging

from library.config import Config
from library.utils import seed_everything, get_logger, save_checkpoint, load_checkpoint
from library.dataset import get_dataloaders, get_test_dataloader
from library.model import HierarchicalEfficientNet
from library.loss import HierarchicalLoss


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)

        # Move all target tensors to device
        targets = {k: v.to(device) for k, v in targets.items()}

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP_VAL)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes Macro F1 for Species and average Hierarchical Loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Collect species predictions for F1 score
            # Output shape: (B, Num_Species)
            species_logits = outputs["species"]
            preds = torch.argmax(species_logits, dim=1).cpu().numpy()
            true_labels = targets["species"].cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(true_labels)

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Calculate Macro F1 Score
    val_f1 = f1_score(all_targets, all_preds, average="macro")

    return val_f1, avg_loss


def run_training_stage(
    stage_num, model, train_loader, val_loader, epochs, lr, device, logger
):
    """
    Runs a specific training stage (e.g., Stage 1 or Stage 2).
    """
    logger.info(f"Starting Training Stage {stage_num}")
    logger.info(f"Hyperparameters: Epochs={epochs}, Max LR={lr}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY)

    # Scheduler (OneCycleLR)
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=10000.0,
    )

    criterion = HierarchicalLoss()

    best_f1 = 0.0
    patience = 5  # Early stopping patience
    patience_counter = 0

    checkpoint_dir = os.path.join(Config.OUTPUT_DIR, f"stage_{stage_num}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_f1, val_loss = validate(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val F1: {val_f1}"
        )

        # Checkpoint logic
        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1
            patience_counter = 0
            logger.info(f"New best model found! F1: {val_f1}")
        else:
            patience_counter += 1

        # Save checkpoint
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_f1": best_f1,
                "stage": stage_num,
            },
            is_best,
            checkpoint_dir,
        )

        # Early Stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch} epochs.")
            break

    logger.info(f"Stage {stage_num} completed. Best Val F1: {best_f1}")

    # Load best model from this stage before returning
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path}")
        load_checkpoint(best_model_path, model, device=device)

    return best_f1


def generate_submission(model, device, logger):
    """
    Generates submission file using the trained model.
    Applies Horizontal Flip TTA.
    """
    logger.info("Generating submission...")

    # Use Stage 2 image size for inference
    image_size = Config.STAGE2_IMAGE_SIZE
    batch_size = Config.STAGE2_BATCH_SIZE

    test_loader = get_test_dataloader(image_size, batch_size)

    model.eval()
    predictions = []
    image_ids = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # 1. Forward pass original
            outputs_orig = model(images)
            logits_orig = outputs_orig["species"]

            # 2. Forward pass flipped (TTA)
            images_flipped = torch.flip(
                images, dims=[3]
            )  # Flip width dimension (N, C, H, W)
            outputs_flip = model(images_flipped)
            logits_flip = outputs_flip["species"]

            # 3. Average logits
            avg_logits = (logits_orig + logits_flip) / 2.0

            # 4. Get predictions
            preds = torch.argmax(avg_logits, dim=1).cpu().numpy()

            predictions.extend(preds)
            image_ids.extend(ids)

    # Create submission DataFrame
    df_sub = pd.DataFrame({"Id": image_ids, "Predicted": predictions})

    # Save
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    submission_path = os.path.join(Config.OUTPUT_DIR, "submission.csv")
    df_sub.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")


def main():
    seed_everything(Config.SEED)

    # Setup Logger
    log_file = os.path.join(Config.OUTPUT_DIR, "train.log")
    logger = get_logger("PlantTraining", log_file)

    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")

    # Initialize Model
    logger.info(f"Initializing model: {Config.MODEL_NAME}")
    model = HierarchicalEfficientNet(
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes_species=Config.NUM_CLASSES_SPECIES,
        num_classes_genus=Config.NUM_CLASSES_GENUS,
        num_classes_family=Config.NUM_CLASSES_FAMILY,
    )
    model = model.to(device)

    # ====================================================
    # Stage 1: Feature Learning (Low Res)
    # ====================================================
    train_loader_s1, val_loader_s1 = get_dataloaders(stage=1, debug=Config.DEBUG)

    run_training_stage(
        stage_num=1,
        model=model,
        train_loader=train_loader_s1,
        val_loader=val_loader_s1,
        epochs=Config.STAGE1_EPOCHS,
        lr=Config.STAGE1_LR,
        device=device,
        logger=logger,
    )

    # ====================================================
    # Stage 2: Fine-Grained Refinement (High Res)
    # ====================================================
    # Note: Model already has best weights from Stage 1 loaded by run_training_stage

    train_loader_s2, val_loader_s2 = get_dataloaders(stage=2, debug=Config.DEBUG)

    run_training_stage(
        stage_num=2,
        model=model,
        train_loader=train_loader_s2,
        val_loader=val_loader_s2,
        epochs=Config.STAGE2_EPOCHS,
        lr=Config.STAGE2_LR,
        device=device,
        logger=logger,
    )

    # ====================================================
    # Submission
    # ====================================================
    generate_submission(model, device, logger)

    logger.info("Training and submission generation completed successfully.")
