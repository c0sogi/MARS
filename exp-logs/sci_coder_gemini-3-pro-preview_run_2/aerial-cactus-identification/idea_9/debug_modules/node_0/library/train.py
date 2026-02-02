import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import set_seed, save_checkpoint, compute_roc_auc
from library.dataset import get_dataloaders
from library.model import DeeplySupervisedUNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # (N, 1)

        optimizer.zero_grad()

        # Forward pass: Get logits from both heads
        semantic_logits, detail_logits = model(images)

        # Compute joint loss
        loss_semantic = criterion(semantic_logits, labels)
        loss_detail = criterion(detail_logits, labels)

        loss = (Config.LOSS_WEIGHT_SEMANTIC * loss_semantic) + (
            Config.LOSS_WEIGHT_DETAIL * loss_detail
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            # Forward pass
            semantic_logits, detail_logits = model(images)

            # Compute loss
            loss_semantic = criterion(semantic_logits, labels)
            loss_detail = criterion(detail_logits, labels)
            loss = (Config.LOSS_WEIGHT_SEMANTIC * loss_semantic) + (
                Config.LOSS_WEIGHT_DETAIL * loss_detail
            )

            running_loss += loss.item() * images.size(0)

            # Aggregate predictions: Average probabilities from both heads
            prob_semantic = torch.sigmoid(semantic_logits)
            prob_detail = torch.sigmoid(detail_logits)
            avg_prob = (prob_semantic + prob_detail) / 2.0

            all_labels.append(labels.cpu().numpy())
            all_preds.append(avg_prob.cpu().numpy())

    total_loss = running_loss / len(loader.dataset)

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    auc_score = compute_roc_auc(all_labels, all_preds)

    return total_loss, auc_score


def run_training():
    """
    Main driver function to run the training process.
    Iterates over seeds, trains models, and saves the best checkpoints.
    """
    # Load data once (caching handled internally)
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Homogeneous Seed Averaging Loop
    for seed in Config.SEEDS:
        print(f"\n{'='*40}")
        print(f"Starting Training for Seed {seed}")
        print(f"{'='*40}")

        # 1. Set Seed for reproducibility
        set_seed(seed)

        # 2. Initialize Model
        model = DeeplySupervisedUNet().to(device)

        # 3. Setup Optimizer and Scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.MAX_EPOCHS, eta_min=Config.ETA_MIN
        )

        # 4. Loss Function
        criterion = nn.BCEWithLogitsLoss()

        # 5. Training Loop
        best_auc = 0.0
        patience_counter = 0
        best_model_state = None

        for epoch in range(Config.MAX_EPOCHS):
            # Train
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )

            # Validate
            val_loss, val_auc = evaluate(model, val_loader, criterion, device)

            # Step Scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            print(
                f"Epoch [{epoch+1}/{Config.MAX_EPOCHS}] "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val AUC: {val_auc:.10f}"
            )

            # Early Stopping & Checkpointing
            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = model.state_dict()
                patience_counter = 0
                # Save immediately to ensure we have the file on disk
                save_filename = f"model_seed_{seed}.pth"
                save_checkpoint(best_model_state, save_filename)
                print(f"--> New Best AUC! Model saved to {save_filename}")
            else:
                patience_counter += 1
                print(
                    f"--> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
                )

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Finished training for seed {seed}. Best Val AUC: {best_auc:.10f}")
