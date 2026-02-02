import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library import config, utils, dataset, model


def mixup_data(x, y, alpha=0.4):
    """
    Applies Mixup augmentation to the data.
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates Mixup loss.
    Handles standard output: (Batch, Num_Classes).
    """
    # pred shape: (Batch, Num_Classes)
    # y_a, y_b shape: (Batch, Num_Classes)

    loss_a = criterion(pred, y_a)
    loss_b = criterion(pred, y_b)
    return lam * loss_a + (1 - lam) * loss_b


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = inputs.size(0)

        # Apply Mixup
        inputs, targets_a, targets_b, lam = mixup_data(
            inputs, targets, config.MIXUP_ALPHA
        )

        optimizer.zero_grad()

        # Forward pass
        # Output shape: (Batch, Num_Classes)
        outputs = model(inputs)

        # Compute Loss
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.size(0)

            # Forward pass
            # In eval mode, model returns averaged logits: (Batch, Num_Classes)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            preds_list.append(probs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(preds_list, axis=0)
    all_targets = np.concatenate(targets_list, axis=0)

    auc_score = utils.calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, auc_score


def run_fold(fold_id, df_folds):
    """
    Runs the training and validation loop for a single fold.
    """
    # Set seed for reproducibility
    utils.set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Fold {fold_id} on {device}")

    # Prepare DataLoaders
    train_loader, val_loader = dataset.get_dataloaders(
        fold_id, df_folds, debug=config.DEBUG
    )

    # Initialize Model
    net = model.BirdResNet(pretrained=config.PRETRAINED)
    net.to(device)

    # Calculate Positive Weights for Imbalance Handling
    # Use only training data for this fold to calculate weights
    df_train = df_folds[df_folds["fold"] != fold_id]
    label_cols = [c for c in df_train.columns if c.startswith("species_")]
    pos_weights = utils.calculate_pos_weights(df_train, label_cols)
    pos_weights = pos_weights.to(device)

    # Loss Function
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Optimizer
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    # Training Loop Variables
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(config.WORKING_DIR, f"model_fold_{fold_id}.pth")

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            net, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_auc = validate(net, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Print Metrics (Full Precision)
        print(
            f"Fold {fold_id} Epoch {epoch+1}/{config.EPOCHS} "
            f"[Time: {elapsed:.2f}s] "
            f"Train Loss: {train_loss} "
            f"Val Loss: {val_loss} "
            f"Val AUC: {val_auc}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(net.state_dict(), best_model_path)
            print(f"New best model saved for Fold {fold_id} with AUC: {best_auc}")
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered for Fold {fold_id} at Epoch {epoch+1}")
            break

    # Load best model weights before returning (optional, but good practice if used immediately)
    net.load_state_dict(torch.load(best_model_path, map_location=device))

    return best_auc
