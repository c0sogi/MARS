import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import seed_everything
from library.dataset import CatDogDataset
from library.models import build_model


def train_one_model(model_name: str, patience: int = 3):
    """
    Trains a single model architecture specified by model_name.
    Implements the training loop, validation, checkpointing, and early stopping.

    Args:
        model_name (str): The name of the model backbone (must match timm names).
        patience (int): Number of epochs to wait for improvement before early stopping.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Starting training for model: {model_name} on {device}")

    # 1. Prepare DataLoaders
    train_dataset = CatDogDataset(Config.TRAIN_CSV, phase="train", debug=Config.DEBUG)
    val_dataset = CatDogDataset(Config.VAL_CSV, phase="val", debug=Config.DEBUG)

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

    # 2. Build Model
    model = build_model(model_name, pretrained=True)
    model = model.to(device)

    # 3. Define Optimizer, Scheduler, and Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # Binary Cross Entropy with Logits (combines Sigmoid + BCELoss)
    # This is numerically stable and standard for binary classification
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, f"{model_name}_best.pth")

    for epoch in range(1, Config.EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)  # Shape: [batch, 1]

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            train_samples += images.size(0)

        # Update learning rate
        scheduler.step()

        avg_train_loss = train_loss_sum / train_samples

        # --- Validation Phase ---
        model.eval()
        val_loss_sum = 0.0
        val_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * images.size(0)
                val_samples += images.size(0)

        avg_val_loss = val_loss_sum / val_samples

        # Print metrics (Full precision as requested)
        print(
            f"Epoch {epoch}/{Config.EPOCHS} - Train Loss: {avg_train_loss} - Val Loss: {avg_val_loss}"
        )

        # --- Checkpointing & Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation loss improved. Model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Finished training {model_name}. Best Val Loss: {best_val_loss}")

    # Clean up GPU memory
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()


def train_all_models():
    """
    Iterates through all models defined in Config.MODELS and trains them sequentially.
    """
    for model_name in Config.MODELS:
        train_one_model(model_name)
