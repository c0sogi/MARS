import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import calculate_roc_auc


def get_class_weights(dataset):
    """
    Calculates class weights inversely proportional to class frequencies.
    Weight = Total_Samples / (Num_Classes * Class_Count)
    """
    # Check if dataset has targets (Train/Val)
    if not hasattr(dataset, "targets") or dataset.targets is None:
        return None

    # dataset.targets is (N, 4) float32, likely one-hot or probabilities
    # Convert to class indices for counting
    targets = dataset.targets
    y_indices = np.argmax(targets, axis=1)

    classes, counts = np.unique(y_indices, return_counts=True)
    n_classes = Config.NUM_CLASSES
    n_samples = len(y_indices)

    # Initialize weights container
    weights = np.zeros(n_classes)

    # Calculate weights
    for cls, count in zip(classes, counts):
        weights[cls] = n_samples / (n_classes * count)

    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        # CrossEntropyLoss expects class indices
        # targets shape: (B, 4) -> (B,)
        target_indices = torch.argmax(targets, dim=1)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, target_indices)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            target_indices = torch.argmax(targets, dim=1)

            outputs = model(images)
            loss = criterion(outputs, target_indices)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Metric
    auc = calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, auc


def fit(model, train_loader, val_loader=None, epochs=10, device="cuda", patience=5):
    """
    Main training loop.
    - Handles Discriminative Fine-Tuning optimizer setup.
    - Handles Cosine Annealing Scheduler.
    - Handles Early Stopping (if val_loader is provided).
    - If val_loader is None (Phase 2), trains for fixed 'epochs'.

    Returns:
        history (dict): Training logs.
        best_epoch (int): The epoch where best validation metric was achieved.
    """
    # 1. Setup Optimizer with Discriminative Learning Rates
    optimizer_params = model.get_optimizer_params()
    optimizer = torch.optim.AdamW(optimizer_params)

    # 2. Setup Scheduler (Cosine Annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. Setup Loss Function with Class Weights
    weights = get_class_weights(train_loader.dataset)
    if weights is not None:
        weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model.to(device)

    # Tracking variables
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    # Temporary path for saving best model during Early Stopping
    temp_model_path = os.path.join(Config.WORKING_DIR, "temp_best_model.pth")

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Step Scheduler
        scheduler.step()

        history["train_loss"].append(train_loss)
        log_msg = f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}"

        # Validation & Early Stopping
        if val_loader is not None:
            val_loss, val_auc = evaluate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)

            log_msg += f" - Val Loss: {val_loss:.6f} - Val AUC: {val_auc:.6f}"
            print(log_msg)

            # Checkpoint
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), temp_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch}. Best AUC: {best_auc:.6f} at epoch {best_epoch}."
                    )
                    break
        else:
            # Phase 2: No validation, just print train loss
            print(log_msg)
            # In Phase 2, we assume the last epoch is the target, or we just return the final state
            best_epoch = epoch

    # Load best model if we used validation
    if val_loader is not None and os.path.exists(temp_model_path):
        print(f"Loading best model from epoch {best_epoch}...")
        model.load_state_dict(torch.load(temp_model_path))
        # Clean up temp file
        try:
            os.remove(temp_model_path)
        except OSError:
            pass

    return history, best_epoch


def predict(model, dataloader, device):
    """
    Generates predictions for a dataset.
    """
    model.eval()
    model.to(device)
    all_preds = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds)


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print("Generating predictions for test set...")
    preds = predict(model, test_loader, device)

    # Load test metadata to get image IDs
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    image_ids = df_test["image_id"].values

    # Create DataFrame
    submission = pd.DataFrame(preds, columns=Config.TARGET_COLS)
    submission.insert(0, "image_id", image_ids)

    # Save
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
