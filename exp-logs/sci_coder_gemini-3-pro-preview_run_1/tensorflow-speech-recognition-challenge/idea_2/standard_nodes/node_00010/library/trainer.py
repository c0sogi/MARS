import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.dataset import get_dataloaders
from library.model import ConvNeXtAudio


# Ensure reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.seed)


def mixup_data(x, y, alpha=0.4, device="cuda"):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, device, alpha):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Apply Mixup
        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha, device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Calculate weighted accuracy for monitoring
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (
            lam * predicted.eq(targets_a).sum().float()
            + (1 - lam) * predicted.eq(targets_b).sum().float()
        ).item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(load_cached_data=True, subset_size=None, patience=10):
    """
    Main training loop with Early Stopping.
    """
    Config.setup()
    device = torch.device(Config.device)
    print(f"Using device: {device}")

    # 1. Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data, subset_size=subset_size
    )

    # 2. Initialize Model
    model = ConvNeXtAudio(
        model_name=Config.model_name,
        num_classes=Config.num_classes,
        pretrained=Config.pretrained,
    )
    model = model.to(device)

    # 3. Setup Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.min_lr
    )

    best_acc = 0.0
    epochs_no_improve = 0

    print("Starting training...")

    for epoch in range(Config.epochs):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, Config.mixup_alpha
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{Config.epochs} - Time: {elapsed:.2f}s")
        print(f"Train Loss: {train_loss} | Train Acc: {train_acc}")
        # Print full precision as requested
        print(f"Val Loss: {val_loss} | Val Acc: {val_acc}")

        # Checkpoint & Early Stopping
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Config.model_save_path)
            epochs_no_improve = 0
            print(f"New best model saved with accuracy: {best_acc}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping triggered after {epochs_no_improve} epochs without improvement."
                )
                break

    print(f"Training complete. Best Validation Accuracy: {best_acc}")
    print(f"Best model saved to {Config.model_save_path}")

    # 4. Generate Submission
    predict_submission(test_loader, device)


def predict_submission(test_loader, device):
    """
    Load best model, predict on test set, and save submission.csv.
    """
    print("Generating submission...")

    # Load Model
    model = ConvNeXtAudio(
        model_name=Config.model_name,
        num_classes=Config.num_classes,
        pretrained=False,
    )
    model.load_state_dict(torch.load(Config.model_save_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []

    # Iterate through test loader
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            all_preds.extend(preds)

    # Map IDs to Labels
    pred_labels = [Config.id2label[p] for p in all_preds]

    # Get filenames from the dataset dataframe
    # test_loader is not shuffled, so order matches the dataset
    test_dataset = test_loader.dataset
    fnames = test_dataset.df["filepath"].apply(os.path.basename).tolist()

    if len(fnames) != len(pred_labels):
        print(
            f"Warning: Number of predictions ({len(pred_labels)}) does not match number of files ({len(fnames)})."
        )

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"fname": fnames, "label": pred_labels})

    # Save
    df_sub.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
