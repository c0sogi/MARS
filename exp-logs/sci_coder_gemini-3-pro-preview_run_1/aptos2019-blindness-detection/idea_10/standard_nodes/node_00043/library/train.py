import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim import lr_scheduler

from library.config import Config, seed_everything
from library.dataset import get_loaders, mixup_data
from library.model import MultiScaleConvNeXt
from library.utils import quadratic_weighted_kappa, ModelEMA


def get_ordinal_targets(targets, num_classes, device):
    """
    Converts integer targets to ordinal binary vectors.
    Example for 5 classes (output dim 4):
    0 -> [0, 0, 0, 0]
    1 -> [1, 0, 0, 0]
    2 -> [1, 1, 0, 0]
    3 -> [1, 1, 1, 0]
    4 -> [1, 1, 1, 1]
    """
    # Create a range tensor [0, 1, 2, 3]
    range_tensor = torch.arange(num_classes - 1, device=device)
    # Expand targets to compare: [B, 1] > [1, 4]
    # Result is float tensor of shape [B, 4]
    return (targets.unsqueeze(1) > range_tensor).float()


def train_fn(model, ema_model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        # Mixup
        if Config.mixup_prob > 0 and np.random.random() < Config.mixup_prob:
            images, targets_a, targets_b, lam = mixup_data(
                images, targets, Config.mixup_alpha, device
            )

            # Convert integer targets to ordinal vectors
            targets_a_ordinal = get_ordinal_targets(
                targets_a, Config.num_classes, device
            )
            targets_b_ordinal = get_ordinal_targets(
                targets_b, Config.num_classes, device
            )

            optimizer.zero_grad()
            logits = model(images)

            # Mixup Loss
            loss = lam * criterion(logits, targets_a_ordinal) + (1 - lam) * criterion(
                logits, targets_b_ordinal
            )
        else:
            targets_ordinal = get_ordinal_targets(targets, Config.num_classes, device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets_ordinal)

        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        optimizer.step()

        # Update EMA
        if ema_model is not None:
            ema_model.update(model)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def eval_fn(model, loader, device):
    model.eval()
    preds = []
    true_labels = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)

            # Decode Ordinal Regression: Sum of Sigmoids
            # Probabilities for each threshold P(y > k)
            probs = torch.sigmoid(logits)

            # Expected value / Score = Sum of probabilities
            # e.g., [0.9, 0.8, 0.1, 0.0] -> 1.8 -> round to 2
            scores = probs.sum(dim=1)

            preds.append(scores.cpu().numpy())
            true_labels.append(targets.cpu().numpy())

    preds = np.concatenate(preds)
    true_labels = np.concatenate(true_labels)

    # Calculate Metric
    score = quadratic_weighted_kappa(true_labels, preds)
    return score, preds, true_labels


def inference_fn(model, loader, device):
    model.eval()
    all_preds = []

    # 4-View TTA
    # 1. Original
    # 2. Horizontal Flip
    # 3. Vertical Flip
    # 4. Rotate 180 (HFlip + VFlip)

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            batch_preds = []

            # Define TTA transformations
            # images shape: [B, C, H, W]
            transforms = [
                lambda x: x,  # Original
                lambda x: torch.flip(x, [3]),  # Horizontal Flip
                lambda x: torch.flip(x, [2]),  # Vertical Flip
                lambda x: torch.flip(x, [2, 3]),  # Rotate 180
            ]

            for t in transforms:
                aug_images = t(images)
                logits = model(aug_images)
                probs = torch.sigmoid(logits)
                scores = probs.sum(dim=1)  # Continuous score 0-4
                batch_preds.append(scores.cpu().numpy())

            # Average predictions across TTA views
            avg_preds = np.mean(batch_preds, axis=0)
            all_preds.append(avg_preds)

    return np.concatenate(all_preds)


def run_training(epochs=Config.epochs, load_cached_data=True):
    seed_everything(Config.seed)

    device = Config.device
    print(f"Using device: {device}")

    # Data Loaders
    train_loader, val_loader, test_loader = get_loaders(
        load_cached_data=load_cached_data
    )

    # Model Setup
    model = MultiScaleConvNeXt(pretrained=True).to(device)

    # EMA Setup
    ema_helper = None
    if Config.use_ema:
        ema_helper = ModelEMA(model, decay=Config.ema_decay)
        print("Model EMA enabled.")

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.min_lr
    )

    # Loss Function (Binary Cross Entropy for Ordinal Targets)
    criterion = nn.BCEWithLogitsLoss()

    best_score = -1.0
    best_model_path = os.path.join(Config.working_dir, "best_model.pth")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_fn(
            model, ema_helper, train_loader, criterion, optimizer, device, epoch
        )

        # Validation (Use EMA model if available)
        val_model = ema_helper.get_model() if ema_helper else model
        val_score, _, _ = eval_fn(val_model, val_loader, device)

        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - QWK: {val_score} - Time: {elapsed:.0f}s"
        )

        # Save Best Model
        if val_score > best_score:
            print(
                f"Validation Score Improved ({best_score} -> {val_score}). Saving model..."
            )
            best_score = val_score
            torch.save(val_model.state_dict(), best_model_path)

    print(f"Training complete. Best QWK: {best_score}")

    # --- Inference on Test Set ---
    print("Starting Inference on Test Set...")

    # Load Best Model
    inference_model = MultiScaleConvNeXt(pretrained=False)
    inference_model.load_state_dict(torch.load(best_model_path, map_location=device))
    inference_model.to(device)

    # Generate Predictions
    test_scores = inference_fn(inference_model, test_loader, device)

    # Round and Clip
    test_labels = np.round(test_scores).astype(int)
    test_labels = np.clip(test_labels, 0, 4)

    # Prepare Submission
    df_test = test_loader.dataset.df

    # Ensure submission directory exists
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Create DataFrame
    submission = pd.DataFrame({"id_code": df_test["id_code"], "diagnosis": test_labels})

    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    # Allow command line arguments or default to Config
    # For this implementation, we stick to Config but allow function calls
    run_training()
