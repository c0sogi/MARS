import os
import torch
import torch.nn as nn
from timm.loss import SoftTargetCrossEntropy

from library.utils import MetricMonitor
from library.config import CFG


def train_one_epoch(epoch, model, train_loader, optimizer, device, mixup_fn=None):
    """
    Performs one epoch of training.
    """
    model.train()
    metric_monitor = MetricMonitor()

    # Use SoftTargetCrossEntropy if MixUp is active (targets are probabilities),
    # otherwise use standard CrossEntropy with label smoothing.
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=CFG.label_smoothing)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Apply MixUp/CutMix
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_monitor.update("Loss", loss.item())

    return metric_monitor.metrics["Loss"]["avg"]


def valid_one_epoch(epoch, model, val_loader, device):
    """
    Performs one epoch of validation.
    """
    model.eval()
    metric_monitor = MetricMonitor()
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Calculate Accuracy
            predicted = outputs.argmax(dim=1)
            accuracy = (predicted == targets).float().mean()

            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy.item())

    return (
        metric_monitor.metrics["Loss"]["avg"],
        metric_monitor.metrics["Accuracy"]["avg"],
    )


def train_model(
    model, train_loader, val_loader, optimizer, scheduler, device, mixup_fn=None
):
    """
    Orchestrates the training loop with Early Stopping.
    """
    best_acc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(CFG.output_dir, "best_model.pth")

    for epoch in range(CFG.epochs):
        # Run Training and Validation
        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, device, mixup_fn
        )
        val_loss, val_acc = valid_one_epoch(epoch, model, val_loader, device)

        # Step Scheduler
        if scheduler is not None:
            scheduler.step()

        # Print Metrics (Full Precision)
        print(
            f"Epoch: {epoch + 1} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Accuracy: {val_acc}"
        )

        # Early Stopping Logic
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= CFG.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    return best_acc
