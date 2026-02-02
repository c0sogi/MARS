import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import seed_everything, calculate_multilabel_auc, average_checkpoints
from library.dataset import BirdDataset, get_transforms
from library.model import BirdClassifier


class Trainer:
    def __init__(self, model, device, optimizer, scheduler, criterion):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)

            # Mixup
            if np.random.random() < 0.5:
                lam = np.random.beta(1.0, 1.0)
                index = torch.randperm(batch_size).to(self.device)

                mixed_images = lam * images + (1 - lam) * images[index, :]
                mixed_labels = lam * labels + (1 - lam) * labels[index, :]

                outputs = self.model(mixed_images)
                loss = self.criterion(outputs, mixed_labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        if self.scheduler is not None:
            self.scheduler.step()

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate_one_epoch(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                batch_size = images.size(0)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                # Apply sigmoid for probabilities
                preds = torch.sigmoid(outputs)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        epoch_loss = running_loss / dataset_size

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        auc_score = calculate_multilabel_auc(all_labels, all_preds)

        return epoch_loss, auc_score


def run_fold(fold_idx, df, backbone_name):
    """
    Runs training for a specific fold and backbone.
    """
    seed_everything(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Fold {fold_idx} with backbone {backbone_name} on {device}")

    # Split Data
    train_df = df[df["fold"] != fold_idx].reset_index(drop=True)
    val_df = df[df["fold"] == fold_idx].reset_index(drop=True)

    # Debug mode: subset data
    if Config.DEBUG:
        train_df = train_df.head(Config.BATCH_SIZE * 2)
        val_df = val_df.head(Config.BATCH_SIZE * 2)

    # Datasets
    train_dataset = BirdDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = BirdDataset(val_df, transforms=get_transforms("valid"), mode="train")

    # DataLoaders
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

    # Model
    model = BirdClassifier(backbone_name, Config.NUM_CLASSES, pretrained=True)
    model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    # Trainer
    trainer = Trainer(model, device, optimizer, scheduler, criterion)

    # Top-K Checkpoint Tracking
    # List of tuples: (auc_score, epoch, file_path)
    top_k_checkpoints = []

    for epoch in range(Config.EPOCHS):
        train_loss = trainer.train_one_epoch(train_loader)
        val_loss, val_auc = trainer.validate_one_epoch(val_loader)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc}"
        )

        # Checkpoint Logic
        ckpt_name = f"{backbone_name}_fold_{fold_idx}_epoch_{epoch+1}.pth"
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, ckpt_name)

        # We always save the current epoch first, then decide whether to keep it
        torch.save(model.state_dict(), ckpt_path)

        # Add to list
        top_k_checkpoints.append((val_auc, epoch + 1, ckpt_path))

        # Sort by AUC descending
        top_k_checkpoints.sort(key=lambda x: x[0], reverse=True)

        # Keep only Top-K
        if len(top_k_checkpoints) > Config.TOP_K_CHECKPOINTS:
            # Remove the worst checkpoint from list and disk
            worst_ckpt = top_k_checkpoints.pop()
            worst_path = worst_ckpt[2]
            if os.path.exists(worst_path):
                os.remove(worst_path)

    # End of training: Average the Top-K checkpoints
    print(f"Averaging top {len(top_k_checkpoints)} checkpoints for Fold {fold_idx}...")

    best_paths = [ckpt[2] for ckpt in top_k_checkpoints]
    averaged_weights = average_checkpoints(best_paths)

    # Save averaged model
    final_name = f"{backbone_name}_fold_{fold_idx}_averaged.pth"
    final_path = os.path.join(Config.CHECKPOINT_DIR, final_name)
    torch.save(averaged_weights, final_path)

    print(f"Saved averaged model to {final_path}")

    # Cleanup: Optionally remove the individual top-k checkpoints to save space
    # (Keeping them might be useful for analysis, but instructions imply we use the averaged one)
    # Here we will keep them as per standard practice unless space is critical,
    # but the logic above already limits us to K files per fold.

    return final_path
