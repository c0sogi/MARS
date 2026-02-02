import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import get_score
from library.models import AdaptiveBackbone
from library.ensemble import extract_features_and_cache


def run_fine_tuning(train_loader, val_loader, epochs):
    """
    Fine-tunes the AdaptiveBackbone model using a temporary regression head.
    Cite debug_lesson_4: Explicitly defining this function to fix ImportError.
    """
    print(f"Starting fine-tuning for {epochs} epochs...")
    device = Config.DEVICE

    # Initialize model
    model = AdaptiveBackbone(pretrained=True).to(device)

    # Add a temporary head for fine-tuning
    # The backbone returns concatenated embeddings
    head = nn.Linear(model.embedding_dim, 1).to(device)

    # Loss: MSE for regression (1-100)
    criterion = nn.MSELoss()

    # Optimizer: Fine-tune backbone and train head
    # Use lower LR for backbone, higher for head
    optimizer = optim.AdamW(
        [
            {"params": model.parameters(), "lr": Config.LEARNING_RATE},
            {"params": head.parameters(), "lr": 1e-3},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[Config.LEARNING_RATE, 1e-3],
        steps_per_epoch=len(train_loader),
        epochs=epochs,
    )

    for epoch in range(epochs):
        model.train()
        head.train()
        train_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()

            embeddings = model(images)
            preds = head(embeddings).squeeze()

            loss = criterion(preds, targets)
            loss.backward()

            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), Config.MAX_GRAD_NORM
            )

            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        head.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                targets = batch["target"].to(device)

                embeddings = model(images)
                preds = head(embeddings).squeeze()

                val_preds.append(preds.cpu().numpy())
                val_targets.append(targets.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_rmse = get_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f}"
        )

    return model


def extract_features(model, dataloader, mode, tta=False, load_cached_data=True):
    """
    Extracts features using the fine-tuned model.
    Wraps the caching and TTA logic provided in library.ensemble.

    Args:
        model: The fine-tuned AdaptiveBackbone model.
        dataloader: DataLoader for the dataset.
        mode: 'train', 'valid', or 'test'.
        tta: Whether to apply Test-Time Augmentation.
        load_cached_data: Whether to try loading from cache first.

    Returns:
        tuple: (features, targets, ids)
    """
    return extract_features_and_cache(
        model=model,
        dataloader=dataloader,
        device=Config.DEVICE,
        mode=mode,
        tta=tta,
        load_cached_data=load_cached_data,
    )
