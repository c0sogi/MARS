import os
import time
import numpy as np
import torch
import torch.nn as nn
from library.config import Config
from library.utils import calculate_metric, get_class_weights


def train_one_epoch(epoch, model, train_loader, optimizer, device, criterion):
    """
    Trains the model for one epoch using Automatic Mixed Precision (AMP).
    """
    model.train()
    running_loss = 0.0

    # Initialize GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler()

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward pass and optimization
        scaler.scale(loss).backward()

        # Gradient clipping
        if Config.MAX_GRAD_NORM > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, device, criterion):
    """
    Evaluates the model on the validation set.
    Computes Cross-Entropy Loss and Mean Column-wise ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)

            preds_list.append(probs.cpu().numpy())
            targets_list.append(labels.cpu().numpy())

    avg_loss = running_loss / len(val_loader)

    preds = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)

    # Convert targets to one-hot encoding for ROC AUC calculation
    # targets array contains class indices (0, 1, 2, 3)
    targets_one_hot = np.eye(Config.NUM_CLASSES)[targets]

    auc_score = calculate_metric(targets_one_hot, preds)

    return avg_loss, auc_score


def fit_model(model, train_loader, val_loader, optimizer, device, fold, model_name):
    """
    Manages the full training lifecycle:
    - Weighted Loss
    - Scheduler (Cosine Annealing)
    - Early Stopping
    - Checkpointing (saving best model)
    """
    # Load class weights for weighted CrossEntropyLoss
    class_weights = get_class_weights(load_cached_data=True)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Initialize Cosine Annealing Scheduler
    # T_max corresponds to the total number of epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.SCHEDULER_MIN_LR
    )

    best_score = -np.inf
    counter = 0  # For Early Stopping

    save_path = os.path.join(Config.WORK_DIR, f"{model_name}_fold_{fold}_best.pth")

    print(f"Starting training for {model_name} - Fold {fold}")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, device, criterion
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, device, criterion)

        # Step Scheduler
        if scheduler:
            scheduler.step()

        end_time = time.time()
        elapsed = end_time - start_time

        # Print metrics
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Time: {elapsed:.2f}s")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f} | Val AUC: {val_auc}")

        # Checkpoint and Early Stopping Logic
        if val_auc > best_score:
            print(
                f"Score Improved ({best_score} ---> {val_auc}). Saving model to {save_path}..."
            )
            best_score = val_auc
            torch.save(model.state_dict(), save_path)
            counter = 0
        else:
            counter += 1
            print(f"No improvement. EarlyStopping counter: {counter}/{Config.PATIENCE}")

        if counter >= Config.PATIENCE:
            print("Early Stopping triggered.")
            break

    print(f"Training complete. Best AUC: {best_score}")
    return best_score
