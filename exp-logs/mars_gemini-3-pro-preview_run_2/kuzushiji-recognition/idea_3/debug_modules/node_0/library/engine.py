import math
import sys
import torch
import numpy as np
from library.utils import F1Score


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses_list = []

    # Warmup scheduler for the first epoch to stabilize training
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        losses_list.append(loss_value)

        optimizer.zero_grad()
        losses.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

    return np.mean(losses_list)


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Evaluates the model on the validation set using F1 Score.
    """
    model.eval()
    metric = F1Score()

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        # F1Score class handles CPU conversion internally
        metric.update(outputs, targets)

    return metric.compute()


def fit(model, train_loader, val_loader, config):
    """
    Main training loop with optimizer, scheduler, and early stopping.
    """
    device = config.DEVICE
    model.to(device)

    # Initialize Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Initialize Scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.LR_STEPS, gamma=config.LR_GAMMA
    )

    best_f1 = 0.0
    patience = 5
    patience_counter = 0

    print(f"Start training for {config.NUM_EPOCHS} epochs")

    for epoch in range(config.NUM_EPOCHS):
        mean_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        lr_scheduler.step()

        val_stats = evaluate(model, val_loader, device)

        print(f"Epoch: {epoch}")
        print(f"Training Loss: {mean_loss}")
        print(f"Validation F1: {val_stats['f1']}")
        print(f"Validation Precision: {val_stats['precision']}")
        print(f"Validation Recall: {val_stats['recall']}")

        # Save best model
        if val_stats["f1"] > best_f1:
            best_f1 = val_stats["f1"]
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"Saved best model to {config.MODEL_PATH}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    return best_f1
