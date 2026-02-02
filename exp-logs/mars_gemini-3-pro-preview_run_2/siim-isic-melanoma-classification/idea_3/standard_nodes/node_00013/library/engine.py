import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from library.config import Config
from library.utils import seed_everything, get_score
from library.data import get_loaders
from library.model import ContextGatedEfficientNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for (images, meta), targets in loader:
        images = images.to(device)
        meta = meta.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(images, meta)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Detach and apply sigmoid for metric calculation
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(probs)

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate AUC, handling potential edge cases (e.g., single class in batch)
    try:
        epoch_auc = get_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for (images, meta), targets in loader:
            images = images.to(device)
            meta = meta.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(images, meta)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_targets.append(targets.detach().cpu().numpy())
            all_preds.append(probs)

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        epoch_auc = get_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []
    # Extract image names from the dataset dataframe
    image_names = loader.dataset.df["image_name"].values

    with torch.no_grad():
        for (images, meta), _ in loader:
            images = images.to(device)
            meta = meta.to(device)

            logits = model(images, meta)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)

    all_preds = np.concatenate(all_preds).flatten()
    return image_names, all_preds


def run():
    """
    Main execution function for training and inference.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # Determine metadata dimension from a batch
    # Batch structure: ((images, meta), targets)
    dummy_batch = next(iter(train_loader))
    meta_dim = dummy_batch[0][1].shape[1]
    print(f"Metadata dimension detected: {meta_dim}")

    # 2. Model Initialization
    print("Initializing model...")
    model = ContextGatedEfficientNet(meta_dim=meta_dim, pretrained=Config.PRETRAINED)
    model = model.to(device)

    # 3. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Linear Warmup -> Cosine Annealing
    # Schedulers step per epoch
    scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=Config.WARMUP_EPOCHS)
    scheduler2 = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS - Config.WARMUP_EPOCHS
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[Config.WARMUP_EPOCHS],
    )

    # Loss Function with static Positive Class Weight
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 4. Training Loop
    best_auc = 0.0
    patience = 5
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "model_best.pth")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | LR: {current_lr} | "
            f"Train Loss: {train_loss} | Train AUC: {train_auc} | "
            f"Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs of no improvement."
                )
                break

    # 5. Prediction
    print("Generating predictions...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: No best model found, using current model state.")

    image_names, preds = predict(model, test_loader, device)

    # Save Submission
    submission_df = pd.DataFrame({"image_name": image_names, "target": preds})

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
