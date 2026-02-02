import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional

from library.config import Config
from library.utils import quadratic_weighted_kappa, ModelEMA
from library.model import RetinopathyModel


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ema: Optional[ModelEMA] = None,
    scheduler=None,
):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        batch_size = images.size(0)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()

        if Config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        optimizer.step()

        # Update EMA
        if ema is not None:
            ema.update(model)

        # Update Scheduler if it's step-based (though Config implies epoch-based,
        # some implementations do step updates. We'll leave epoch updates to the main loop
        # unless it's OneCycleLR, but CosineAnnealing is typically epoch-based).

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
):
    """
    Evaluates the model on the validation set.
    Returns QWK score and average loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds_all = []
    targets_all = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Decode Ordinal Regression
            # 1. Sigmoid to get probabilities P(y > k)
            probs = torch.sigmoid(logits)
            # 2. Sum probabilities to get expected score (continuous 0-4)
            scores = probs.sum(dim=1)
            # 3. Round to nearest integer
            preds = scores.round().long()

            # Recover ground truth labels from ordinal targets
            # Summing the target vector [1, 1, 0, 0] gives 2.
            true_labels = targets.sum(dim=1).long()

            preds_all.append(preds.cpu().numpy())
            targets_all.append(true_labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    qwk = quadratic_weighted_kappa(targets_all, preds_all)

    return qwk, epoch_loss


def predict_tta(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
):
    """
    Generates predictions using 4-view Test Time Augmentation (TTA).
    Views: Original, Horizontal Flip, Vertical Flip, Rotate 180.
    """
    model.eval()
    preds_all = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # View 1: Original
            logits1 = model(images)
            probs1 = torch.sigmoid(logits1)

            # View 2: Horizontal Flip
            images_h = torch.flip(images, dims=[3])
            logits2 = model(images_h)
            probs2 = torch.sigmoid(logits2)

            # View 3: Vertical Flip
            images_v = torch.flip(images, dims=[2])
            logits3 = model(images_v)
            probs3 = torch.sigmoid(logits3)

            # View 4: Rotate 180 (Horizontal + Vertical Flip)
            images_r180 = torch.flip(images, dims=[2, 3])
            logits4 = model(images_r180)
            probs4 = torch.sigmoid(logits4)

            # Average probabilities
            avg_probs = (probs1 + probs2 + probs3 + probs4) / 4.0

            # Decode
            scores = avg_probs.sum(dim=1)
            preds = scores.round().long()

            preds_all.append(preds.cpu().numpy())

    return np.concatenate(preds_all)


def train_model(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    test_df: pd.DataFrame,
):
    """
    Main training loop.
    """
    device = Config.device
    print(f"Using device: {device}")

    # Initialize Model
    model = RetinopathyModel().to(device)

    # Initialize EMA
    ema = None
    if Config.use_ema:
        ema = ModelEMA(model, decay=Config.ema_decay, device=device)
        print(f"Model EMA enabled with decay {Config.ema_decay}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    # Scheduler
    # Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.min_lr
    )

    best_qwk = -1.0
    best_model_path = os.path.join(Config.working_dir, "best_model.pth")

    print("Starting training...")

    for epoch in range(Config.epochs):
        # Train
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            ema=ema,
            scheduler=scheduler,
        )

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Validate (Prefer EMA model for validation if available)
        val_model = ema.ema if ema else model
        val_qwk, val_loss = evaluate(val_model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.epochs} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val QWK: {val_qwk}"
        )

        # Save Best Model
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save(val_model.state_dict(), best_model_path)
            print(f"New best model saved with QWK: {best_qwk}")

    print(f"Training complete. Best QWK: {best_qwk}")

    # --- Inference on Test Set ---
    print("Starting inference on test set using best model (TTA enabled)...")

    # Load best model weights
    inference_model = RetinopathyModel().to(device)
    inference_model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Predict
    predictions = predict_tta(inference_model, test_loader, device)

    # Save Submission
    submission_path = os.path.join(Config.working_dir, "submission.csv")

    # Ensure prediction length matches dataframe
    if len(predictions) != len(test_df):
        print(
            f"Warning: Prediction count {len(predictions)} != Test DF length {len(test_df)}"
        )

    # Create submission DataFrame
    # Note: test_df from metadata/test.csv has 'id_code'.
    # Sample submission format is id_code, diagnosis
    submission_df = pd.DataFrame(
        {"id_code": test_df["id_code"], "diagnosis": predictions}
    )

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
