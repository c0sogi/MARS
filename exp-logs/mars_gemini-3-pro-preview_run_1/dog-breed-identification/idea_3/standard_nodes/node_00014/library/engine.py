import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.dataset import get_label_mapping


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): The training data loader.
        optimizer (Optimizer): The optimizer to use.
        device (str): The device to train on.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.CrossEntropyLoss()

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set using TTA (Original + Flip).

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): The validation data loader.
        device (str): The device to evaluate on.

    Returns:
        float: The average validation loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    # Use NLLLoss because we will compute log probabilities manually after TTA
    criterion = nn.NLLLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # TTA: Original
            outputs1 = model(images)
            probs1 = torch.softmax(outputs1, dim=1)

            # TTA: Horizontal Flip
            images_flip = torch.flip(images, dims=[3])
            outputs2 = model(images_flip)
            probs2 = torch.softmax(outputs2, dim=1)

            # Average probabilities
            avg_probs = (probs1 + probs2) / 2.0

            # Compute loss (add epsilon for numerical stability)
            log_probs = torch.log(avg_probs + 1e-9)
            loss = criterion(log_probs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def run_training(
    model,
    train_loader,
    val_loader,
    device,
    phase1_epochs=Config.PHASE1_EPOCHS,
    phase2_epochs=Config.PHASE2_EPOCHS,
    patience=Config.PATIENCE,
):
    """
    Executes the two-phase training pipeline:
    1. Head Adaptation (Frozen Backbone)
    2. Fine-Tuning (Full Model with Discriminative LRs and Cosine Annealing)

    Tracks the best model across both phases and implements early stopping.
    """
    best_loss = float("inf")

    # --------------------------------------------------------------------------
    # Phase 1: Head Adaptation (Frozen Backbone)
    # --------------------------------------------------------------------------
    print("Starting Phase 1: Head Adaptation")

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Optimizer for head only (filter ensures we only optimize requires_grad=True params)
    optimizer_p1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.PHASE1_LR,
        weight_decay=Config.WEIGHT_DECAY,
    )

    for epoch in range(phase1_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer_p1, device)
        val_loss = evaluate(model, val_loader, device)

        print(
            f"Phase 1 Epoch {epoch+1}/{phase1_epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # --------------------------------------------------------------------------
    # Phase 2: Fine-Tuning (Full Model)
    # --------------------------------------------------------------------------
    print("Starting Phase 2: Fine-Tuning")

    # Unfreeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = True

    # Discriminative Learning Rates: Low for backbone, High for head
    optimizer_p2 = optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": Config.PHASE2_LR_BACKBONE},
            {"params": model.fc.parameters(), "lr": Config.PHASE2_LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2, T_max=phase2_epochs, eta_min=Config.ETA_MIN
    )

    patience_counter = 0

    for epoch in range(phase2_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer_p2, device)
        val_loss = evaluate(model, val_loader, device)

        scheduler.step()

        print(
            f"Phase 2 Epoch {epoch+1}/{phase2_epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Validation Loss: {best_loss}")


def generate_submission(
    model, test_loader, device, output_path=Config.FINAL_SUBMISSION_PATH
):
    """
    Generates predictions for the test set using TTA and saves them to a CSV file.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on.
        output_path (str): Path to save the submission CSV.
    """
    model.eval()

    ids = []
    probs = []

    with torch.no_grad():
        for images, img_ids in test_loader:
            images = images.to(device)

            # TTA: Original
            outputs1 = model(images)
            probs1 = torch.softmax(outputs1, dim=1)

            # TTA: Horizontal Flip
            images_flip = torch.flip(images, dims=[3])
            outputs2 = model(images_flip)
            probs2 = torch.softmax(outputs2, dim=1)

            # Average probabilities
            avg_probs = (probs1 + probs2) / 2.0

            ids.extend(img_ids)
            probs.append(avg_probs.cpu().numpy())

    # Concatenate all batch probabilities
    probs = np.concatenate(probs, axis=0)

    # Get class names to ensure correct column order
    _, classes = get_label_mapping(load_cached_data=True)

    # Create DataFrame
    df = pd.DataFrame(probs, columns=classes)
    df.insert(0, "id", ids)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
