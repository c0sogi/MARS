import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library import config, utils, data_loader, model


def train_one_epoch(train_loader, model, criterion, optimizer, device):
    """
    Training loop for one epoch.
    """
    model.train()
    losses = utils.AverageMeter()

    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # (Batch, 1)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(val_loader, model, criterion, device):
    """
    Validation loop. Returns average loss and ROC AUC score.
    """
    model.eval()
    losses = utils.AverageMeter()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            probs = torch.sigmoid(outputs)

            losses.update(loss.item(), images.size(0))
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    # Handle edge case where only one class is present in the batch/loader
    if len(np.unique(all_targets)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(all_targets, all_probs)

    return losses.avg, auc


def run_fold(fold_idx):
    """
    Runs training for a specific fold.
    """
    print(f"\nStarting Fold {fold_idx}...")

    # 1. Data Loading
    train_loader, val_loader = data_loader.get_dataloaders(
        fold_idx, load_cached_data=True
    )

    # 2. Model Initialization
    net = model.ACWIVNet(
        backbone_name=config.BACKBONE,
        pretrained=config.PRETRAINED,
        input_channels=config.INPUT_CHANNELS,
    )
    net = net.to(config.DEVICE)

    # 3. Optimizer & Loss
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_auc = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            train_loader, net, criterion, optimizer, config.DEVICE
        )

        # Validate
        val_loss, val_auc = validate(val_loader, net, criterion, config.DEVICE)

        elapsed = time.time() - start_time

        # Log Metrics
        metrics = {
            "Fold": fold_idx,
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "Val AUC": val_auc,
            "Time": f"{elapsed:.2f}s",
        }
        utils.print_metrics(metrics)

        # Checkpointing
        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            patience_counter = 0
            utils.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "best_auc": best_auc,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
                fold_idx=fold_idx,
            )
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1} for fold {fold_idx}")
            break

    print(f"Fold {fold_idx} finished. Best AUC: {best_auc}")
    return best_auc


def train_all_folds():
    """
    Main entry point to run 5-fold cross-validation.
    """
    utils.seed_everything(config.SEED)

    fold_scores = []

    for fold in range(config.NUM_FOLDS):
        score = run_fold(fold)
        fold_scores.append(score)

    print("\n=========================")
    print(f"Cross-Validation Complete")
    print(f"Fold AUCs: {fold_scores}")
    print(f"Mean AUC: {np.mean(fold_scores)}")
    print("=========================")
