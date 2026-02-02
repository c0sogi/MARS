import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.model import ModelEMA


def mixup_data(x, y, alpha=0.2, device="cuda"):
    """
    Applies Mixup augmentation to inputs and targets.
    Returns mixed inputs, mixed targets, and the lambda value.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]

    # Mix targets (assuming y is float for BCE)
    y_a, y_b = y, y[index]
    mixed_y = lam * y_a + (1 - lam) * y_b

    return mixed_x, mixed_y


def train_one_epoch(model, loader, optimizer, device, epoch, ema_model=None):
    """
    Trains the model for one epoch using Mixup and EMA updates.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        # Apply Mixup
        if Config.USE_MIXUP:
            images, labels = mixup_data(
                images, labels, alpha=Config.MIXUP_ALPHA, device=device
            )

        # Forward pass
        # Reshape labels to (N, 1) to match logit shape if necessary,
        # though BCEWithLogitsLoss usually handles (N) vs (N,1) with broadcasting,
        # explicit reshaping is safer.
        outputs = model(images).view(-1)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update EMA
        if ema_model and Config.USE_EMA:
            ema_model.update(model)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Uses the EMA model if provided (passed as 'model' argument).
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            batch_size = images.size(0)

            outputs = model(images).view(-1)
            loss = criterion(outputs, labels)

            probs = torch.sigmoid(outputs)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            all_targets.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate AUC
    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc_score = 0.5  # Handle edge case with single class in batch

    return epoch_loss, auc_score


def fit(
    model, train_loader, val_loader, optimizer, scheduler, device, num_epochs, patience
):
    """
    Main training loop with Early Stopping and Model EMA.
    """
    print(f"Starting training on device: {device}")

    # Initialize EMA
    ema_model = None
    if Config.USE_EMA:
        print("Initializing Model EMA...")
        ema_model = ModelEMA(model, decay=Config.EMA_DECAY, device=device)

    best_auc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, ema_model
        )

        # Step Scheduler (Cosine Annealing usually stepped per epoch)
        if scheduler:
            scheduler.step()

        # Validate
        # Use EMA model for validation if available
        eval_model = ema_model.get_model() if ema_model else model
        val_loss, val_auc = validate(eval_model, val_loader, device)

        end_time = time.time()
        epoch_mins = (end_time - start_time) / 60

        # Print metrics with full precision
        print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_mins:.2f}m")
        print(f"  Train Loss: {train_loss:.10f}")
        print(f"  Val Loss:   {val_loss:.10f}")
        print(f"  Val AUC:    {val_auc:.10f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.8f}")

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            print(
                f"  [Improvement] AUC increased from {best_auc:.10f} to {val_auc:.10f}. Saving model..."
            )
            best_auc = val_auc
            patience_counter = 0
            torch.save(eval_model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            patience_counter += 1
            print(f"  [No Improvement] Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc:.10f}")


def predict(model, test_loader, device):
    """
    Generates predictions for the test set using 8-view Test Time Augmentation (TTA).
    Saves predictions to submission.csv.
    """
    print("Starting inference with TTA (8 views)...")

    # Load best weights
    if os.path.exists(Config.BEST_MODEL_PATH):
        state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded best model checkpoint.")
    else:
        print("Warning: Best model checkpoint not found. Using current model weights.")

    model.eval()
    model.to(device)

    all_ids = []
    all_probs = []

    # Retrieve IDs from dataset
    # We assume the loader preserves order and dataset has 'ids' attribute
    # If not, we can read from test metadata again, but dataset.ids is safer
    if hasattr(test_loader.dataset, "ids"):
        all_ids = test_loader.dataset.ids
    else:
        # Fallback: Read metadata directly
        df_test = pd.read_csv(Config.TEST_METADATA)
        all_ids = df_test["id"].values

    # Store predictions
    predictions = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            batch_size = images.size(0)

            # TTA: 8 Views
            # 1. Original
            # 2. Rot 90
            # 3. Rot 180
            # 4. Rot 270
            # 5. HFlip
            # 6. HFlip + Rot 90
            # 7. HFlip + Rot 180
            # 8. HFlip + Rot 270

            # We accumulate probabilities
            batch_probs = torch.zeros(batch_size, device=device)

            # Define transformations
            transforms = [
                lambda x: x,  # Original
                lambda x: torch.rot90(x, 1, [2, 3]),  # Rot 90
                lambda x: torch.rot90(x, 2, [2, 3]),  # Rot 180
                lambda x: torch.rot90(x, 3, [2, 3]),  # Rot 270
                lambda x: torch.flip(x, [3]),  # HFlip
                lambda x: torch.rot90(torch.flip(x, [3]), 1, [2, 3]),  # HFlip + Rot 90
                lambda x: torch.rot90(torch.flip(x, [3]), 2, [2, 3]),  # HFlip + Rot 180
                lambda x: torch.rot90(torch.flip(x, [3]), 3, [2, 3]),  # HFlip + Rot 270
            ]

            if Config.USE_TTA:
                views = transforms
            else:
                views = [transforms[0]]  # Only original

            for transform in views:
                aug_images = transform(images)
                logits = model(aug_images).view(-1)
                probs = torch.sigmoid(logits)
                batch_probs += probs

            # Average probabilities
            batch_probs /= len(views)

            predictions.extend(batch_probs.cpu().numpy())

    # Create submission dataframe
    df_sub = pd.DataFrame({"id": all_ids, "label": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total predictions: {len(df_sub)}")
