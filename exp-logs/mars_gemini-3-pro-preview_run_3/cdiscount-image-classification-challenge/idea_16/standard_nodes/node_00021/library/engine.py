import os
import torch
import torch.nn as nn
import numpy as np
import time
from library.config import Config


def mixup_data(x, alpha=1.0, device="cuda"):
    """
    Applies MixUp to input features.
    Returns mixed inputs, pairs of targets (indices), and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, index, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes loss for MixUp: linear combination of losses for the two targets.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch using Feature-Space MixUp.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (features, l1, l2, l3) in enumerate(loader):
        features = features.to(device)
        l1 = l1.to(device)
        l2 = l2.to(device)
        l3 = l3.to(device)

        batch_size = features.size(0)

        # Apply MixUp
        mixed_features, idx_perm, lam = mixup_data(
            features, alpha=Config.MIXUP_ALPHA, device=device
        )

        # Forward pass
        optimizer.zero_grad()
        out_l1, out_l2, out_l3 = model(mixed_features)

        # Compute Hierarchical Loss with MixUp
        loss_l1 = mixup_criterion(criterion, out_l1, l1, l1[idx_perm], lam)
        loss_l2 = mixup_criterion(criterion, out_l2, l2, l2[idx_perm], lam)
        loss_l3 = mixup_criterion(criterion, out_l3, l3, l3[idx_perm], lam)

        total_loss = loss_l1 + loss_l2 + loss_l3

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Statistics
        running_loss += total_loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Level 3 accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct_l3 = 0
    dataset_size = 0

    with torch.no_grad():
        for features, l1, l2, l3 in loader:
            features = features.to(device)
            l1 = l1.to(device)
            l2 = l2.to(device)
            l3 = l3.to(device)

            batch_size = features.size(0)

            # Forward pass (No MixUp)
            out_l1, out_l2, out_l3 = model(features)

            # Compute Loss
            loss_l1 = criterion(out_l1, l1)
            loss_l2 = criterion(out_l2, l2)
            loss_l3 = criterion(out_l3, l3)
            total_loss = loss_l1 + loss_l2 + loss_l3

            running_loss += total_loss.item() * batch_size

            # Compute Accuracy (Level 3 is the target)
            _, preds_l3 = torch.max(out_l3, 1)
            correct_l3 += torch.sum(preds_l3 == l3).item()
            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    epoch_acc_l3 = correct_l3 / dataset_size

    return epoch_loss, epoch_acc_l3


def train_model(model, train_loader, val_loader, model_name="best_model.pth"):
    """
    Main training loop with Early Stopping.
    """
    device = Config.DEVICE
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss Function (Label Smoothing supported in recent PyTorch versions)
    # We use reduction='mean' by default
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    # Early Stopping Config
    patience = 5
    patience_counter = 0
    best_val_acc = 0.0
    best_model_path = os.path.join(Config.MODEL_DIR, model_name)

    print(f"Starting training on {device}...")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    start_time = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        epoch_duration = time.time() - epoch_start

        # Print full precision metrics as requested
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Time: {epoch_duration:.2f}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Accuracy (L3): {val_acc}")

        # Early Stopping Logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.2f}s. Best Val Accuracy: {best_val_acc}")

    # Load best model weights before returning
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model
