import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import set_seed, mixup_data, calculate_roc_auc, calculate_pos_weights
from library.data import get_loaders
from library.models import get_model


def train_step(model, inputs, labels, criterion, optimizer, device):
    """
    Performs a single training step with Mixup augmentation.
    """
    model.train()
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Apply Mixup
    inputs, targets_a, targets_b, lam = mixup_data(
        inputs, labels, alpha=Config.MIXUP_ALPHA, device=device
    )

    # Forward pass
    outputs = model(inputs)

    # Compute loss
    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
        outputs, targets_b
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def validate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and macro-averaged ROC AUC.
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            # Apply sigmoid for probabilities
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # Aggregate results
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    avg_loss = val_loss / len(val_loader.dataset)
    auc_score = calculate_roc_auc(all_targets, all_preds)

    return avg_loss, auc_score


def train_fold(fold_idx, backbone_name):
    """
    Trains a specific backbone on a specific fold.
    Saves the best model based on validation ROC AUC.
    """
    # Ensure reproducibility for this fold/model combination
    set_seed(Config.SEED + fold_idx)

    device = torch.device(Config.DEVICE)

    # Initialize Model
    print(f"Initializing {backbone_name} for Fold {fold_idx}...")
    model = get_model(backbone_name, num_classes=Config.NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # Get DataLoaders
    train_loader, val_loader = get_loaders(fold=fold_idx)

    # Calculate Positive Weights for Imbalance Handling
    # Access the underlying dataframe of the training dataset
    train_df = train_loader.dataset.df
    pos_weights = calculate_pos_weights(train_df, device=device)

    # Loss Function and Optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Training Loop State
    global_step = 0
    best_auc = 0.0
    best_model_path = os.path.join(
        Config.WORKING_DIR, f"model_{backbone_name}_fold_{fold_idx}.pth"
    )

    # Infinite iterator over train_loader
    train_iter = iter(train_loader)

    print(f"Starting training for {Config.MAX_STEPS} steps...")

    while global_step < Config.MAX_STEPS:
        try:
            batch = next(train_iter)
        except StopIteration:
            # Restart iterator if epoch ends
            train_iter = iter(train_loader)
            batch = next(train_iter)

        inputs, labels, _ = batch

        # Perform training step
        train_loss = train_step(model, inputs, labels, criterion, optimizer, device)
        global_step += 1

        # Validation Check
        if global_step % Config.VAL_CHECK_INTERVAL == 0:
            val_loss, val_auc = validate(model, val_loader, criterion, device)

            print(
                f"Step {global_step}: Train Loss {train_loss}, Val Loss {val_loss}, Val AUC {val_auc}"
            )

            # Save Best Model
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)
                print(f"  New best model saved! AUC: {best_auc}")

    print(
        f"Training complete for Fold {fold_idx} - {backbone_name}. Best AUC: {best_auc}"
    )

    # Clean up to free memory
    del model, optimizer, criterion
    torch.cuda.empty_cache()

    return best_auc
