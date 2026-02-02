import os
import torch
import torch.nn as nn
import timm
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from library.config import Config
from library.utils import ModelEMA, calculate_f1, optimize_threshold
from library.loss import AsymmetricLoss
from library.dataset import get_dataloaders


class ArtworkClassifier(nn.Module):
    """
    Classifier wrapper for timm backbones.
    Supports creating different architectures for the Heterogeneous Ensemble.
    """

    def __init__(self, model_name, num_classes, pretrained=True):
        super(ArtworkClassifier, self).__init__()
        # Create backbone using timm
        # num_classes ensures the head is adapted to our target size (3474)
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        # Return logits. Sigmoid activation is handled by AsymmetricLoss during training
        # and explicitly applied during inference.
        return self.model(x)


def train_one_epoch(
    model, loader, optimizer, scheduler, criterion, device, scaler, ema=None
):
    """
    Trains the model for one epoch using Mixed Precision and Asymmetric Loss.
    """
    model.train()
    total_loss = 0.0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Mixed Precision Training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if ema:
            ema.update(model)

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns loss, F1 score, raw probabilities, and targets.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)

            total_loss += loss.item()
            # Convert logits to probabilities for metric calculation
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Micro F1 with default 0.5 threshold for monitoring progress
    # Note: Final threshold is optimized globally later
    bin_preds = (all_preds >= 0.5).astype(int)
    f1 = calculate_f1(bin_preds, all_targets)

    return total_loss / len(loader), f1, all_preds, all_targets


def inference(model, loader, device, use_tta=False):
    """
    Generates predictions for the test set.
    Supports Test-Time Augmentation (Horizontal Flip).
    """
    model.eval()
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            images, ids = batch
            images = images.to(device)

            with autocast():
                # Prediction 1: Original Image
                logits = model(images)
                probs = torch.sigmoid(logits)

                if use_tta:
                    # Prediction 2: Horizontally Flipped Image
                    images_flip = torch.flip(images, dims=[3])
                    logits_flip = model(images_flip)
                    probs_flip = torch.sigmoid(logits_flip)

                    # Average probabilities
                    probs = (probs + probs_flip) / 2.0

            all_probs.append(probs.cpu().numpy())
            all_ids.extend(ids)

    return np.concatenate(all_probs), all_ids


def run_training():
    """
    Orchestrates the training of the Heterogeneous Ensemble.
    1. Trains multiple models (ConvNeXt, Swin) defined in Config.
    2. Uses EMA and Asymmetric Loss.
    3. Aggregates predictions from all models.
    4. Optimizes threshold on Validation set.
    5. Generates final submission.
    """
    device = torch.device(Config.device)
    print(f"Using device: {device}")

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders()

    ensemble_val_probs = []
    ensemble_test_probs = []
    val_targets_cache = None

    # Iterate over models in the ensemble
    for model_name in Config.model_names:
        print(f"\n--- Training Model: {model_name} ---")

        # Initialize Model
        model = ArtworkClassifier(model_name, Config.num_classes).to(device)

        # Optimizer (AdamW)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
        )

        # Scheduler (OneCycleLR)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=Config.lr,
            steps_per_epoch=len(train_loader),
            epochs=Config.epochs,
            pct_start=Config.pct_start,
            div_factor=Config.div_factor,
            final_div_factor=Config.final_div_factor,
        )

        # Loss & Scaler
        criterion = AsymmetricLoss()
        scaler = GradScaler()

        # EMA
        ema = None
        if Config.use_ema:
            ema = ModelEMA(model, decay=Config.ema_decay, device=device)

        # Training Loop
        best_f1 = -1.0
        best_model_path = os.path.join(Config.working_dir, f"{model_name}_best.pth")

        for epoch in range(Config.epochs):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                criterion,
                device,
                scaler,
                ema,
            )

            # Validate (use EMA if available)
            val_model = ema.module if ema else model
            val_loss, val_f1, _, _ = validate(val_model, val_loader, criterion, device)

            # Print full precision metrics
            print(
                f"Epoch {epoch+1} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val F1: {val_f1}"
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(val_model.state_dict(), best_model_path)

        print(f"Finished training {model_name}. Best F1: {best_f1}")

        # Load Best Model for Inference
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # Generate Validation Probabilities (for ensemble thresholding)
        _, _, val_probs, val_targets = validate(model, val_loader, criterion, device)
        ensemble_val_probs.append(val_probs)
        val_targets_cache = val_targets

        # Generate Test Probabilities
        test_probs, test_ids = inference(
            model, test_loader, device, use_tta=Config.use_tta
        )
        ensemble_test_probs.append(test_probs)

        # Cleanup to free GPU memory
        del model, optimizer, scheduler, scaler, ema
        torch.cuda.empty_cache()

    # --- Ensemble Aggregation ---
    print("\n--- Aggregating Ensemble ---")

    # Average Probabilities
    avg_val_probs = np.mean(ensemble_val_probs, axis=0)
    avg_test_probs = np.mean(ensemble_test_probs, axis=0)

    # Optimize Threshold on Validation Set
    best_thresh, best_score = optimize_threshold(avg_val_probs, val_targets_cache)
    print(f"Optimal Threshold: {best_thresh} - Best Ensemble Val F1: {best_score}")

    # Apply Threshold to Test Set
    test_preds_bin = (avg_test_probs >= best_thresh).astype(int)

    # Generate Submission
    submission_rows = []
    for i, img_id in enumerate(test_ids):
        # Get indices where prediction is 1
        pred_indices = np.where(test_preds_bin[i] == 1)[0]
        # Format as space-separated string
        pred_str = " ".join(map(str, pred_indices))
        submission_rows.append({"id": img_id, "attribute_ids": pred_str})

    df_sub = pd.DataFrame(submission_rows)
    df_sub.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
