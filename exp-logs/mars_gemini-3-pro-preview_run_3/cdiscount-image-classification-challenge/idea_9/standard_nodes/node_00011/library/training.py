import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything
from library.model import HierarchicalMLP


def train_one_epoch(model, loader, criterion, optimizer, device, mixup_alpha):
    """
    Trains the model for one epoch using Feature-Space MixUp.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (features, targets) in enumerate(loader):
        features = features.to(device)
        y1, y2, y3 = targets
        y1, y2, y3 = y1.to(device), y2.to(device), y3.to(device)

        batch_size = features.size(0)

        # Apply MixUp
        if mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            index = torch.randperm(batch_size).to(device)

            mixed_features = lam * features + (1 - lam) * features[index]

            y1_a, y1_b = y1, y1[index]
            y2_a, y2_b = y2, y2[index]
            y3_a, y3_b = y3, y3[index]

            # Forward pass
            out_l1, out_l2, out_l3 = model(mixed_features)

            # Multi-task Mixed Loss
            loss_l1 = lam * criterion(out_l1, y1_a) + (1 - lam) * criterion(
                out_l1, y1_b
            )
            loss_l2 = lam * criterion(out_l2, y2_a) + (1 - lam) * criterion(
                out_l2, y2_b
            )
            loss_l3 = lam * criterion(out_l3, y3_a) + (1 - lam) * criterion(
                out_l3, y3_b
            )

        else:
            # Standard training without MixUp
            out_l1, out_l2, out_l3 = model(features)
            loss_l1 = criterion(out_l1, y1)
            loss_l2 = criterion(out_l2, y2)
            loss_l3 = criterion(out_l3, y3)

        # Total Loss
        loss = loss_l1 + loss_l2 + loss_l3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Level 3 (Target) Accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct_l3 = 0
    total = 0

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            y1, y2, y3 = targets
            y1, y2, y3 = y1.to(device), y2.to(device), y3.to(device)

            out_l1, out_l2, out_l3 = model(features)

            # Calculate Loss (No MixUp)
            loss_l1 = criterion(out_l1, y1)
            loss_l2 = criterion(out_l2, y2)
            loss_l3 = criterion(out_l3, y3)
            loss = loss_l1 + loss_l2 + loss_l3

            running_loss += loss.item() * features.size(0)

            # Calculate Accuracy for Target (Level 3)
            _, predicted = torch.max(out_l3, 1)
            correct_l3 += (predicted == y3).sum().item()
            total += features.size(0)

    avg_loss = running_loss / total
    accuracy_l3 = correct_l3 / total

    return avg_loss, accuracy_l3


def train_ensemble_member(member_id, train_loader, val_loader):
    """
    Trains a single member of the MLP ensemble.

    Args:
        member_id (int): Identifier for this ensemble member (0 to ENSEMBLE_SIZE-1).
        train_loader (DataLoader): Loader for training data.
        val_loader (DataLoader): Loader for validation data.

    Returns:
        float: Best validation accuracy achieved by this member.
    """
    # Unique seed per member for diversity
    current_seed = Config.SEED + member_id
    seed_everything(current_seed)

    device = torch.device(Config.DEVICE)
    print(f"\n=== Training Ensemble Member {member_id} (Seed: {current_seed}) ===")

    # Initialize Model
    model = HierarchicalMLP(
        input_dim=Config.EMBEDDING_DIM,
        num_classes_l1=Config.NUM_CLASSES_L1,
        num_classes_l2=Config.NUM_CLASSES_L2,
        num_classes_l3=Config.NUM_CLASSES_L3,
    ).to(device)

    # Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Reduce LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=1
    )

    # Early Stopping Tracking
    best_acc = -1.0
    patience_counter = 0
    save_path = os.path.join(Config.CACHE_DIR, f"mlp_ensemble_{member_id}.pth")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, Config.MIXUP_ALPHA
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Acc L3: {val_acc:.8f}"
        )

        # Scheduler Step
        scheduler.step(val_acc)

        # Checkpoint & Early Stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            # print(f"  -> New best model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Member {member_id} finished. Best Validation Accuracy: {best_acc:.8f}")
    return best_acc
