import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

from library.config import Config
from library.utils import seed_everything, ModelEMA
from library.dataset import AnimalDataset, get_transforms
from library.model import MultiTaskConvNeXt
from library.loss import CompositeLoss, get_dampened_class_weights


def train_one_epoch(model, loader, optimizer, criterion, device, ema_model=None):
    """
    Trains the model for one epoch using the composite loss.
    Updates the EMA model if provided.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        images = batch["image"].to(device)
        # CompositeLoss expects a targets dict with 'species_label' and 'detection_label'
        targets = {
            "species_label": batch["species_label"].to(device),
            "detection_label": batch["detection_label"].to(device),
        }

        batch_size = images.size(0)

        optimizer.zero_grad()

        outputs = model(images)

        # Calculate composite loss (Species Focal Loss + Lambda * Detection BCE)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if ema_model:
            ema_model.update(model)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Uses the Species Head logits for the final prediction.
    Returns the Macro F1 Score.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["species_label"].to(device)

            outputs = model(images)
            # Use species_logits for classification
            preds = torch.argmax(outputs["species_logits"], dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    score = f1_score(all_targets, all_preds, average="macro")
    return score


def train_model(epochs=Config.EPOCHS, sample_size=None):
    """
    Main training routine.
    Initializes datasets, model, optimizer, and executes the training loop with early stopping.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Initializing Datasets...")
    # Load datasets with optional sample_size for debugging
    train_dataset = AnimalDataset(
        mode="train", transform=get_transforms("train"), sample_size=sample_size
    )
    val_dataset = AnimalDataset(
        mode="val", transform=get_transforms("val"), sample_size=sample_size
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

    print(f"Initializing Model: {Config.MODEL_NAME}")
    model = MultiTaskConvNeXt(pretrained=True).to(device)

    # Initialize EMA (Exponential Moving Average) of the model weights
    ema_model = ModelEMA(model, decay=Config.EMA_DECAY, device=device)

    # Calculate class weights for the Species Focal Loss
    print("Calculating class weights...")
    class_weights = get_dampened_class_weights(train_dataset.df, device)

    # Initialize Composite Loss
    criterion = CompositeLoss(
        class_weights=class_weights, lambda_detection=Config.LAMBDA_DETECTION
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_f1 = 0.0
    patience_counter = 0

    print("Starting Training...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, ema_model
        )

        # Validate using EMA model for better stability
        val_f1 = evaluate(ema_model.module, val_loader, device)

        scheduler.step()

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Macro F1: {val_f1}"
        )

        # Save Best Model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(ema_model.module.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New Best Model Saved! F1: {best_f1}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Save final EMA model state
    torch.save(ema_model.module.state_dict(), Config.EMA_MODEL_PATH)
    print("Training completed.")


def generate_submission():
    """
    Generates predictions for the test set using the best saved model.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Loading Test Data...")
    test_dataset = AnimalDataset(mode="test", transform=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print("Loading Model...")
    model = MultiTaskConvNeXt(pretrained=False).to(device)

    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading weights from {Config.BEST_MODEL_PATH}")
        state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("Warning: Best model not found. Using random initialization.")

    model.eval()

    predictions = []
    ids = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            batch_ids = batch["id"]

            outputs = model(images)
            # Use species head for final prediction
            preds = torch.argmax(outputs["species_logits"], dim=1)

            predictions.extend(preds.cpu().numpy())
            ids.extend(batch_ids)

    df_sub = pd.DataFrame({"Id": ids, "Predicted": predictions})
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
