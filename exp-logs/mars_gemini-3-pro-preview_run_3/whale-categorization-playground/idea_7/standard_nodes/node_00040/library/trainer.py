import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed, calculate_map5
from library.loss import ArcFaceLoss
from library.model import WhaleModel


def train_one_epoch(model, criterion, optimizer, dataloader, device, epoch):
    """
    Trains the model for one epoch using ArcFace loss.

    Args:
        model (nn.Module): The neural network model.
        criterion (nn.Module): The ArcFace loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        dataloader (DataLoader): Training data loader.
        device (str): Device to run training on.
        epoch (int): Current epoch number.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    criterion.train()

    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass: Get embeddings
        embeddings = model(images)

        # Calculate loss
        loss = criterion(embeddings, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, criterion, dataloader, device):
    """
    Validates the model.

    Uses the learned ArcFace prototypes (class centers) as the gallery to
    compute MAP@5 efficiently during training.

    Args:
        model (nn.Module): The neural network model.
        criterion (nn.Module): The ArcFace loss function (contains prototypes).
        dataloader (DataLoader): Validation data loader.
        device (str): Device to run validation on.

    Returns:
        tuple: (Average Validation Loss, MAP@5 Score)
    """
    model.eval()
    criterion.eval()

    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    # Get normalized prototypes from ArcFaceLoss for inference
    # Shape: (num_classes, embedding_dim)
    prototypes = criterion.weight.detach().clone()
    prototypes = F.normalize(prototypes, p=2, dim=1)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Get embeddings
            embeddings = model(images)

            # Calculate validation loss
            loss = criterion(embeddings, labels)
            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

            # Prediction for MAP@5
            # 1. Normalize embeddings
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)

            # 2. Cosine similarity: (Batch, Dim) @ (Num_Classes, Dim).T -> (Batch, Num_Classes)
            logits = torch.mm(norm_embeddings, prototypes.t())

            # 3. Get top 5 predictions
            _, top_indices = torch.topk(logits, k=5, dim=1)

            all_preds.extend(top_indices.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    epoch_loss = running_loss / dataset_size

    # Calculate MAP@5
    map5_score = calculate_map5(all_preds, all_targets)

    return epoch_loss, map5_score


def train_model(model_name, train_loader, val_loader, num_classes, device=None):
    """
    Executes the full training loop for a specific backbone architecture.

    Args:
        model_name (str): Name of the backbone (e.g., 'efficientnet_b2').
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        num_classes (int): Total number of classes.
        device (str, optional): Device to train on. Defaults to Config.DEVICE.

    Returns:
        nn.Module: The trained model with the best validation weights loaded.
    """
    if device is None:
        device = Config.DEVICE

    print(f"Initializing model: {model_name}")

    # Initialize Model
    model = WhaleModel(model_name=model_name, pretrained=Config.PRETRAINED)
    model = model.to(device)

    # Initialize Loss
    # Input features to ArcFace is Config.EMBEDDING_DIM (output of model head)
    criterion = ArcFaceLoss(
        in_features=Config.EMBEDDING_DIM,
        out_features=num_classes,
        s=Config.ARCFACE_S,
        m=Config.ARCFACE_M,
    )
    criterion = criterion.to(device)

    # Initialize Optimizer
    # We optimize both model parameters and ArcFace loss centers (criterion.parameters())
    optimizer = AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    # Tracking
    best_map5 = -1.0
    patience_counter = 0

    save_dir = Config.WORKING_DIR
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"{model_name}_best.pth")

    start_time = time.time()

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, criterion, optimizer, train_loader, device, epoch
        )

        # Validate
        val_loss, val_map5 = validate(model, criterion, val_loader, device)

        # Step Scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Time: {epoch_time:.2f}s | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | Val MAP@5: {val_map5}"
        )

        # Checkpoint & Early Stopping
        if val_map5 > best_map5:
            best_map5 = val_map5
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New Best MAP@5! Model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.2f}s. Best MAP@5: {best_map5}")

    # Load best weights
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model weights.")

    return model
