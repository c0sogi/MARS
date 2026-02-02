import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library import utils, dataset, model


def train_one_epoch(net, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    net.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(net, dataloader, criterion, device):
    """
    Validates the model on the validation set.
    Returns the average CrossEntropyLoss and the calculated Log Loss metric.
    """
    net.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply softmax to get probabilities for Log Loss calculation
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Concatenate all batches
    if len(all_preds) > 0:
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_labels, axis=0)
        # Calculate Log Loss using the utility function (sklearn wrapper)
        # Note: y_true are indices, y_pred are probabilities. sklearn handles this.
        val_metric = utils.calculate_log_loss(
            y_true, y_pred, labels=list(range(Config.NUM_CLASSES))
        )
    else:
        val_metric = float("inf")

    return epoch_loss, val_metric


def run_training():
    """
    Orchestrates the Two-Phase Transfer Learning pipeline.
    """
    # 1. Setup
    utils.set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    print("Loading data...")
    dataloaders, class_names = dataset.get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # 3. Model Initialization
    print(f"Initializing model: {Config.MODEL_NAME}")
    net = model.get_model(num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED)
    net = net.to(device)

    # 4. Loss Function
    # Using CrossEntropyLoss which combines LogSoftmax and NLLLoss.
    # This is appropriate for multi-class classification with hard targets.
    criterion = nn.CrossEntropyLoss()

    # ==========================================
    # Phase 1: Warm-up
    # ==========================================
    print("\n" + "=" * 40)
    print(f"Phase 1: Warm-up (Frozen Backbone) for {Config.WARMUP_EPOCHS} epochs")
    print("=" * 40)

    model.freeze_backbone(net)

    # Optimizer for head only
    optimizer_warmup = optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()), lr=Config.WARMUP_LR
    )

    for epoch in range(Config.WARMUP_EPOCHS):
        train_loss = train_one_epoch(
            net, train_loader, criterion, optimizer_warmup, device
        )
        val_loss, val_metric = validate(net, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{Config.WARMUP_EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss:   {val_loss}")
        print(f"Val Metric: {val_metric}")

    # ==========================================
    # Phase 2: Fine-tuning
    # ==========================================
    print("\n" + "=" * 40)
    print(f"Phase 2: Fine-tuning (Unfrozen) for {Config.FINE_TUNE_EPOCHS} epochs")
    print("=" * 40)

    model.unfreeze_all(net)

    # Optimizer for all parameters with lower LR
    optimizer_finetune = optim.AdamW(
        net.parameters(), lr=Config.FINE_TUNE_LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_finetune, T_max=Config.FINE_TUNE_EPOCHS
    )

    best_metric = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    for epoch in range(Config.FINE_TUNE_EPOCHS):
        current_epoch = epoch + 1

        train_loss = train_one_epoch(
            net, train_loader, criterion, optimizer_finetune, device
        )
        val_loss, val_metric = validate(net, val_loader, criterion, device)

        scheduler.step()

        print(f"Epoch {current_epoch}/{Config.FINE_TUNE_EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss:   {val_loss}")
        print(f"Val Metric: {val_metric}")

        # Early Stopping and Model Checkpointing
        if val_metric < best_metric:
            print(
                f"Validation Metric improved from {best_metric} to {val_metric}. Saving model..."
            )
            best_metric = val_metric
            patience_counter = 0
            utils.save_checkpoint(
                net, optimizer_finetune, current_epoch, val_metric, best_model_path
            )
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"\nTraining complete. Best Validation Metric: {best_metric}")
    return best_model_path
