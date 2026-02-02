import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import TrainConfig, PathConfig
from library.utils import calculate_auc, set_seed


def mixup_data(x, y, alpha=0.4, device="cuda"):
    """
    Applies Mixup augmentation to the batch.

    Args:
        x (Tensor): Input batch.
        y (Tensor): Target batch.
        alpha (float): Mixup alpha parameter.
        device (str): Device to move indices to.

    Returns:
        mixed_x (Tensor): Mixed inputs.
        y_a (Tensor): Targets of the first permutation.
        y_b (Tensor): Targets of the second permutation.
        lam (float): Lambda mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Trains the model for one epoch using Mixup and Weighted BCE Loss.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Training dataloader.
        optimizer (Optimizer): The optimizer.
        device (torch.device): Compute device.
        epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Loss function setup
    # The model outputs probabilities (sigmoid applied).
    # We use BCELoss with manual class weighting to handle imbalance.
    pos_weight = TrainConfig.POS_WEIGHT
    criterion = nn.BCELoss(reduction="none")

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)  # Ensure shape (B, 1)

        # Apply Mixup
        inputs, targets_a, targets_b, lam = mixup_data(
            inputs, targets, alpha=TrainConfig.MIXUP_ALPHA, device=device
        )

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Clamp for numerical stability with BCELoss (avoid log(0))
        outputs = torch.clamp(outputs, 1e-7, 1.0 - 1e-7)

        # Compute Weighted Mixup Loss
        # We calculate weights for both permutations:
        # Weight = pos_weight if target is 1, else 1.
        w_a = targets_a * (pos_weight - 1) + 1
        w_b = targets_b * (pos_weight - 1) + 1

        loss_a = (criterion(outputs, targets_a) * w_a).mean()
        loss_b = (criterion(outputs, targets_b) * w_b).mean()

        loss = lam * loss_a + (1 - lam) * loss_b

        # Backward
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        dataset_size += inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Validation dataloader.
        device (torch.device): Compute device.

    Returns:
        tuple: (average_weighted_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    pos_weight = TrainConfig.POS_WEIGHT
    criterion = nn.BCELoss(reduction="none")

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            outputs = torch.clamp(outputs, 1e-7, 1.0 - 1e-7)

            # Apply same weighting logic as training for consistent loss metric
            weights = targets * (pos_weight - 1) + 1
            loss = (criterion(outputs, targets) * weights).mean()

            running_loss += loss.item() * inputs.size(0)
            dataset_size += inputs.size(0)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

    avg_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    auc = calculate_auc(all_targets, all_preds)

    return avg_loss, auc


def fit_model(model, train_loader, val_loader, seed, epochs=None):
    """
    Full training loop for a single model instance.
    Handles optimization, scheduling, checkpointing, and early stopping.

    Args:
        model (nn.Module): Model instance to train.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        seed (int): Random seed for this run.
        epochs (int, optional): Number of epochs. Defaults to config.

    Returns:
        nn.Module: The trained model loaded with the best weights.
    """
    if epochs is None:
        epochs = TrainConfig.EPOCHS

    # Set seed for reproducibility
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=TrainConfig.LEARNING_RATE,
        weight_decay=TrainConfig.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=3
    )

    best_auc = 0.0
    patience_counter = 0

    # Ensure working directory exists
    os.makedirs(PathConfig.WORKING_DIR, exist_ok=True)
    save_path = os.path.join(PathConfig.WORKING_DIR, f"model_seed_{seed}.pth")

    print(f"Training model with seed {seed} on {device}...")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_auc = validate(model, val_loader, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}, Val AUC: {val_auc}"
        )

        scheduler.step(val_auc)

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= TrainConfig.PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Finished training seed {seed}. Best AUC: {best_auc}")

    # Load best model weights
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


def predict(model, loader, device):
    """
    Generates predictions for a dataloader.

    Args:
        model (nn.Module): Trained model.
        loader (DataLoader): Data loader (test set).
        device (torch.device): Compute device.

    Returns:
        tuple: (list_of_clip_ids, numpy_array_of_probabilities)
    """
    model.eval()
    all_probs = []
    all_clips = []

    with torch.no_grad():
        for inputs, clips in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_probs.append(outputs.cpu().numpy().flatten())
            all_clips.extend(clips)

    return all_clips, np.concatenate(all_probs)


def generate_submission(models, test_loader, output_file=None):
    """
    Generates submission file by averaging predictions from an ensemble of models.

    Args:
        models (list[nn.Module]): List of trained models.
        test_loader (DataLoader): Test data loader.
        output_file (str, optional): Path to save CSV. Defaults to config path.
    """
    if output_file is None:
        output_file = PathConfig.SUBMISSION_FILE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Collect predictions from all models
    ensemble_probs = []
    clips = None

    print(f"Generating predictions with {len(models)} models...")

    for i, model in enumerate(models):
        model = model.to(device)
        model.eval()
        c, p = predict(model, test_loader, device)

        if clips is None:
            clips = c
        else:
            # Verify alignment
            if c != clips:
                raise ValueError(f"Mismatch in test loader order for model {i}")

        ensemble_probs.append(p)

    # Average probabilities across the ensemble
    avg_probs = np.mean(ensemble_probs, axis=0)

    # Create DataFrame
    df = pd.DataFrame({"clip": clips, "probability": avg_probs})

    # Save submission
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
