import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import timm

from library.config import Config
from library.utils import seed_everything, save_checkpoint, load_checkpoint
from library.dataset import get_fold_dataloaders, get_test_dataloader


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 based classifier for Dog vs Cat.
    Uses timm to load the backbone and adjusts the head for binary classification.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
    ):
        super(EfficientNetClassifier, self).__init__()
        # timm.create_model handles the loading of weights and replacement of the classifier head
        # when num_classes is specified.
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        # BCEWithLogitsLoss expects float targets of shape (N, 1)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    # Note: Scheduler step is handled in the fold loop (per epoch) based on Config

    return running_loss / dataset_size


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

    return running_loss / dataset_size


def train_fold(fold_idx):
    """
    Trains a single fold of the K-Fold Cross Validation.
    Saves the best model checkpoint based on validation loss.
    """
    seed_everything(Config.SEED + fold_idx)
    device = Config.DEVICE

    # DataLoaders
    train_loader, val_loader = get_fold_dataloaders(fold_idx)

    # Model
    model = EfficientNetClassifier()
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    # T_max corresponds to the number of epochs for CosineAnnealingLR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"fold_{fold_idx}.pth")

    print(f"\nStarting training for Fold {fold_idx}...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Step scheduler after each epoch
        scheduler.step()

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Fold {fold_idx} | Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch+1}")
                break

    return best_val_loss


def train_kfold():
    """
    Orchestrates the training of all K folds.
    """
    print(f"Training {Config.N_FOLDS} folds...")
    scores = []
    for fold in range(Config.N_FOLDS):
        score = train_fold(fold)
        scores.append(score)

    print("\nK-Fold Training Completed.")
    print(f"Average Validation Log Loss: {np.mean(scores):.8f}")


def generate_submission():
    """
    Generates predictions for the test set using an ensemble of all K-Fold models.
    Saves the submission file to Config.SUBMISSION_FILE.
    """
    print("\nGenerating submission...")
    device = Config.DEVICE
    test_loader = get_test_dataloader()

    # Dictionary to accumulate probabilities: id -> sum_of_probs
    id_to_prob_sum = {}
    # To count how many models contributed (should be N_FOLDS, but robust to missing checkpoints)
    id_to_count = {}

    for fold in range(Config.N_FOLDS):
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"fold_{fold}.pth")
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint for fold {fold} not found. Skipping.")
            continue

        print(f"Predicting with Fold {fold} model...")

        model = EfficientNetClassifier()
        load_checkpoint(checkpoint_path, model, device=device)
        model.to(device)
        model.eval()

        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(device)

                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                # ids comes as a tensor from the loader
                ids_np = ids.numpy().flatten()

                for img_id, prob in zip(ids_np, probs):
                    img_id = int(img_id)
                    if img_id not in id_to_prob_sum:
                        id_to_prob_sum[img_id] = 0.0
                        id_to_count[img_id] = 0
                    id_to_prob_sum[img_id] += prob
                    id_to_count[img_id] += 1

    # Average predictions and format for submission
    results = []
    sorted_ids = sorted(id_to_prob_sum.keys())

    for img_id in sorted_ids:
        avg_prob = id_to_prob_sum[img_id] / id_to_count[img_id]
        results.append({"id": img_id, "label": avg_prob})

    submission_df = pd.DataFrame(results)

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
