import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_logger, seed_everything
from library.model import DogCatClassifier, ModelEMA
from library.dataset import create_dataloaders

logger = get_logger()


def train_one_epoch(model, loader, optimizer, criterion, device, ema_model=None):
    """
    Trains the model for one epoch using Mixup augmentation and EMA updates.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Mixup Augmentation
        alpha = Config.MIXUP_ALPHA
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        # Random permutation for mixing
        index = torch.randperm(batch_size).to(device)
        mixed_images = lam * images + (1 - lam) * images[index]

        # Mix labels for Soft-Target Cross-Entropy
        # Labels are (B), need (B, 1) for broadcasting if necessary,
        # but BCEWithLogitsLoss typically expects same shape as input (B, 1)
        y_a = labels.view(-1, 1)
        y_b = labels[index].view(-1, 1)
        mixed_labels = lam * y_a + (1 - lam) * y_b

        optimizer.zero_grad()

        # Forward pass
        outputs = model(mixed_images)
        loss = criterion(outputs, mixed_labels)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        if Config.MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        # Update Model EMA
        if ema_model:
            ema_model.update(model)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    return running_loss / dataset_size


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    correct_counts = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

            # Calculate accuracy
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct_counts += (preds == labels).sum().item()

    epoch_loss = running_loss / dataset_size
    epoch_acc = correct_counts / dataset_size
    return epoch_loss, epoch_acc


def predict_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation (Horizontal Flip).
    """
    model.eval()
    results = []

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)

            # 1. Forward pass on original images
            logits = model(images)
            probs = torch.sigmoid(logits)

            # 2. Forward pass on horizontally flipped images
            # tensor is (B, C, H, W), flip on last dimension (W)
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip)
            probs_flip = torch.sigmoid(logits_flip)

            # 3. Average probabilities
            avg_probs = (probs + probs_flip) / 2.0

            # Collect results
            avg_probs_np = avg_probs.cpu().numpy().flatten()
            ids_np = ids.numpy().flatten()

            for id_val, prob_val in zip(ids_np, avg_probs_np):
                results.append({"id": int(id_val), "label": prob_val})

    return pd.DataFrame(results)


def run_training(debug=False):
    """
    Main function to orchestrate training, validation, and submission.
    """
    # Setup environment
    seed_everything(Config.SEED)
    Config.setup()
    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")

    # Prepare DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(debug=debug)

    # Initialize Model
    model = DogCatClassifier(pretrained=True).to(device)

    # Initialize EMA
    ema = None
    if Config.USE_EMA:
        ema = ModelEMA(model, decay=Config.EMA_DECAY)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # Loss Function (handles soft targets from Mixup)
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_loss = float("inf")
    patience = 3
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "model_best.pth")

    logger.info("Starting training...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, ema_model=ema
        )

        # Validate
        # Use EMA model for validation if available to track its performance
        val_model = ema.get_model() if ema else model
        val_loss, val_acc = validate(val_model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step()

        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {train_loss:.8f} - Val Loss: {val_loss:.8f} - Val Acc: {val_acc:.8f}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save the best model weights (EMA weights if enabled)
            torch.save(val_model.state_dict(), best_model_path)
            logger.info(f"New best model saved with loss {best_loss:.8f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    # Inference
    logger.info("Starting inference with TTA...")

    # Load best model for inference
    inference_model = DogCatClassifier(pretrained=False)
    inference_model.load_state_dict(torch.load(best_model_path, map_location=device))
    inference_model.to(device)

    # Generate predictions
    submission_df = predict_tta(inference_model, test_loader, device)

    # Format and Save Submission
    submission_df = submission_df.sort_values("id")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
