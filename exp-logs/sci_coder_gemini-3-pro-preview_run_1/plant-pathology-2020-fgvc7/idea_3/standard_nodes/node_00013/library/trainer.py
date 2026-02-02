import os
import time
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, get_class_weights, mixup_data
from library.data_loader import get_loaders
from library.model_factory import get_model
from library.loss_factory import get_loss


def train_one_epoch(epoch, model, loader, optimizer, criterion, device, config):
    """
    Trains the model for one epoch using Mixup regularization.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Apply Mixup Augmentation
        if config.use_mixup:
            images, labels_a, labels_b, lam = mixup_data(
                images, labels, config.mixup_alpha, device
            )
            outputs = model(images)
            loss = criterion(outputs, labels_a, labels_b, lam)
        else:
            # Fallback if mixup is disabled (though config enables it)
            # Pass labels as both targets with lambda=1.0
            outputs = model(images)
            loss = criterion(outputs, labels, labels, 1.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device, config):
    """
    Evaluates the model on the validation set and computes ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)

            # Calculate loss without mixup (lam=1.0 means 100% original labels)
            loss = criterion(outputs, labels, labels, 1.0)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply softmax to logits to get probabilities for AUC calculation
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    val_loss = running_loss / dataset_size

    # Concatenate predictions from all batches
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate Mean Column-wise ROC AUC
        try:
            val_auc = roc_auc_score(
                all_targets, all_preds, average="macro", multi_class="ovr"
            )
        except ValueError:
            # Handle edge cases (e.g., during debugging with very small batches) where not all classes are present
            val_auc = 0.0
    else:
        val_auc = 0.0

    return val_loss, val_auc


def run_fold(fold: int, config: Config):
    """
    Executes the training pipeline for a single fold.
    """
    print(f"Starting Fold {fold}")

    # Ensure reproducibility
    seed_everything(config.seed)

    # 1. Prepare Data
    train_loader, val_loader = get_loaders(fold, config)

    # Calculate class weights for the loss function based on training data distribution
    train_df = train_loader.dataset.df
    class_weights = get_class_weights(train_df, config.target_cols)

    # 2. Initialize Model
    model = get_model(config)
    model = model.to(config.device)

    # 3. Initialize Loss
    criterion = get_loss(config, class_weights)

    # 4. Initialize Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # 5. Initialize Scheduler
    # Removed scheduler to maintain constant learning rate (Cite Lesson 00005, 00008)

    # Training Loop
    best_auc = -1.0
    patience_counter = 0

    for epoch in range(config.epochs):
        start_time = time.time()

        # Train Step
        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, criterion, config.device, config
        )

        # Validation Step
        val_loss, val_auc = validate(
            model, val_loader, criterion, config.device, config
        )

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{config.epochs} - Time: {elapsed}s - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
        )

        # Checkpoint and Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0

            # Save the best model
            save_path = os.path.join(
                config.working_dir, f"{config.model_name}_fold_{fold}.pth"
            )
            torch.save(model.state_dict(), save_path)
            print(f"  New best model saved to {save_path}")
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Fold {fold} finished. Best Val AUC: {best_auc}")
    return best_auc
