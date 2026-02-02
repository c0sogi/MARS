import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import set_seed, mixup_data, mixup_criterion, MetricMonitor
from library.dataset import get_dataset
from library.model import get_model


def train_one_epoch(model, train_loader, criterion, optimizer, device, config):
    model.train()
    metric_monitor = MetricMonitor()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Apply Mixup
        data, targets_a, targets_b, lam = mixup_data(
            data, target, config.MIXUP_ALPHA, device
        )

        optimizer.zero_grad()
        output = model(data)
        loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()

        # Calculate accuracy (approximate for mixup)
        # We compare prediction against the dominant label
        _, predicted = torch.max(output.data, 1)
        target_dom = targets_a if lam >= 0.5 else targets_b
        acc = (predicted == target_dom).sum().item() / data.size(0)

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", acc)

    return metric_monitor


def validate(model, val_loader, criterion, device, config):
    model.eval()
    metric_monitor = MetricMonitor()

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            _, predicted = torch.max(output.data, 1)
            acc = (predicted == target).sum().item() / data.size(0)

            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", acc)

    return metric_monitor


def generate_submission(config, device):
    print("Generating submission...")
    # Load Best Model
    model = get_model(config).to(device)

    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"Error: Best model not found at {config.BEST_MODEL_PATH}")
        return

    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    test_dataset = get_dataset("test", config=config)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    predictions = []

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            predictions.extend(preds.cpu().numpy())

    # Map IDs to Labels
    pred_labels = [config.ID2LABEL[p] for p in predictions]

    # Extract filenames from filepath (e.g., test/audio/clip_000.wav -> clip_000.wav)
    fnames = test_dataset.df["filepath"].apply(os.path.basename).tolist()

    # Create DataFrame
    df_sub = pd.DataFrame({"fname": fnames, "label": pred_labels})

    # Save
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")


def train_model():
    config = Config()

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # Data
    print("Loading datasets...")
    train_dataset = get_dataset("train", config=config)
    val_dataset = get_dataset("val", config=config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    print("Initializing model...")
    model = get_model(config).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    best_acc = 0.0
    patience = 10
    patience_counter = 0

    print("Starting training...")
    for epoch in range(config.EPOCHS):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config
        )
        val_metrics = validate(model, val_loader, criterion, device, config)

        scheduler.step()

        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        print(f"Train: {train_metrics}")
        print(f"Val:   {val_metrics}")

        val_acc = val_metrics.metrics["Accuracy"]["avg"]

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            patience_counter = 0
            print(f"New best model saved with Accuracy: {best_acc}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    print(f"Training complete. Best Validation Accuracy: {best_acc}")

    # Generate Submission
    generate_submission(config, device)


# Execute training
train_model()
