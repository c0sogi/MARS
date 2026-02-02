import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from library.config import CFG
from library.data import get_loaders
from library.modeling import get_model

# Define Loss Criterion
CRITERION = nn.BCEWithLogitsLoss()


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    """
    Trains the model for one epoch using Automatic Mixed Precision.
    """
    model.train()

    dataset_size = 0
    running_loss = 0.0

    scaler = GradScaler(enabled=CFG.use_amp)

    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float).unsqueeze(1)

        batch_size = images.size(0)

        with autocast(enabled=CFG.use_amp):
            outputs = model(images)
            loss = CRITERION(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size

    # Step scheduler at the end of the epoch
    if scheduler is not None:
        scheduler.step()

    return epoch_loss


def valid_one_epoch(model, dataloader, device):
    """
    Validates the model for one epoch.
    Returns the average loss and the raw predictions (probabilities).
    """
    model.eval()

    dataset_size = 0
    running_loss = 0.0
    preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float).unsqueeze(1)

            batch_size = images.size(0)

            outputs = model(images)
            loss = CRITERION(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities
            preds.append(torch.sigmoid(outputs).detach().cpu().numpy())

    epoch_loss = running_loss / dataset_size
    predictions = np.concatenate(preds)

    return epoch_loss, predictions


def train_fold(df, fold, model_name, device=CFG.device):
    """
    Orchestrates the training process for a single fold and model architecture.
    Includes Early Stopping and Model Checkpointing.
    """
    print(f"--- Training Fold {fold} for Model: {model_name} ---")

    train_loader, valid_loader = get_loaders(fold, df)

    model = get_model(model_name, pretrained=True)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )

    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr
    )

    best_score = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    # Create directory for saving models
    # Structure: ./working/idea_3/{model_name}/
    model_output_dir = os.path.join(CFG.output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    for epoch in range(CFG.epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, epoch
        )
        valid_loss, _ = valid_one_epoch(model, valid_loader, device)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{CFG.epochs} [{model_name} Fold {fold}]")
        print(f"Train Loss: {train_loss}")
        print(f"Valid Loss: {valid_loss}")  # Full precision
        print(f"Time: {elapsed:.2f}s")

        # Early Stopping and Checkpointing
        if valid_loss < best_score:
            best_score = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            save_path = os.path.join(model_output_dir, f"model_fold_{fold}.pth")
            torch.save(model.state_dict(), save_path)

            print(f"Validation loss improved. Saved model to {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{CFG.patience}")

        if patience_counter >= CFG.patience:
            print("Early stopping triggered.")
            break

    # Load best weights before returning
    model.load_state_dict(best_model_wts)

    return model, best_score


def inference(model, dataloader, device=CFG.device):
    """
    Runs inference on the provided dataloader.
    Applies Test Time Augmentation (Horizontal Flip) if enabled in CFG.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device, dtype=torch.float)

            # Standard forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs).detach().cpu().numpy()

            # Test Time Augmentation (TTA)
            if CFG.tta:
                # Flip images horizontally (dim 3 is width in NCHW)
                images_flipped = torch.flip(images, dims=[3])
                outputs_flipped = model(images_flipped)
                probs_flipped = torch.sigmoid(outputs_flipped).detach().cpu().numpy()

                # Average predictions
                probs = (probs + probs_flipped) / 2.0

            preds.append(probs)

    return np.concatenate(preds)
