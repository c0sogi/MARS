import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import from provided library files
import library.config as config
import library.utils as utils
import library.dataset as dataset
import library.model as model_lib


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch using Deep Supervision.
    Loss = 0.5 * Loss_HeadA + 0.5 * Loss_HeadB
    """
    model.train()
    loss_meter = utils.AverageMeter()

    for i, (images, labels, _) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device).float().view(-1, 1)

        # Forward pass
        # Model returns (logits_mid, logits_final)
        logits_mid, logits_final = model(images)

        # Compute Deep Supervision Loss
        loss_mid = criterion(logits_mid, labels)
        loss_final = criterion(logits_final, labels)

        # Weighted sum based on config (0.5, 0.5)
        loss = (
            config.LOSS_WEIGHTS["head_a"] * loss_mid
            + config.LOSS_WEIGHTS["head_b"] * loss_final
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Prediction = (Prob_HeadA + Prob_HeadB) / 2
    """
    model.eval()
    loss_meter = utils.AverageMeter()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)

            # Forward pass
            logits_mid, logits_final = model(images)

            # Calculate validation loss (using same joint loss for tracking)
            loss_mid = criterion(logits_mid, labels)
            loss_final = criterion(logits_final, labels)
            loss = (
                config.LOSS_WEIGHTS["head_a"] * loss_mid
                + config.LOSS_WEIGHTS["head_b"] * loss_final
            )

            loss_meter.update(loss.item(), images.size(0))

            # Compute probabilities
            # Use only final head for validation metric (Cite solution_lesson_node_00058)
            prob_final = torch.sigmoid(logits_final)

            all_targets.append(labels.cpu().numpy())
            all_preds.append(prob_final.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate ROC AUC
    auc_score = utils.calculate_roc_auc(all_targets, all_preds)

    return loss_meter.avg, auc_score


def run_training(seed, load_cached_data=True):
    """
    Runs the full training pipeline for a specific seed.

    Args:
        seed (int): The random seed to use.
        load_cached_data (bool): Whether to load data from cache.

    Returns:
        float: The best validation AUC achieved.
    """
    # 1. Setup
    utils.set_seed(seed)
    config.setup_directories()
    device = torch.device(config.DEVICE)

    print(f"Starting training for Seed {seed} on device {device}")

    # 2. Data Loading
    train_loader, val_loader, _ = dataset.get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        load_cached_data=load_cached_data,
    )

    # 3. Model Initialization
    net = model_lib.WideResNetPyramidal()
    net = net.to(device)

    # 4. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    # 5. Training Loop
    best_auc = 0.0
    best_epoch = 0
    save_path = os.path.join(config.WORKING_DIR, f"model_seed_{seed}.pth")

    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(
            net, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_auc = validate(net, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Logging
        print(
            f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.10f}"
        )

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1
            torch.save(net.state_dict(), save_path)
            # print(f"New best model saved to {save_path}")

    print(f"Seed {seed} finished. Best AUC: {best_auc:.10f} at Epoch {best_epoch}")
    return best_auc
