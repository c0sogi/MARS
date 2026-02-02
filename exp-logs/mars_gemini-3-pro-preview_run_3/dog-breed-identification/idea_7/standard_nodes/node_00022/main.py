import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR
import cv2

# Import library modules
from library.config import Config
from library.utils import seed_everything, save_clean_checkpoint
from library.data import get_dataloaders
from library.model_factory import create_model, get_llrd_params
from library.trainer import train_one_epoch, validate, update_swa, EarlyStopping
from library.calibration import TemperatureScaler


def pearson_corr(x, y):
    """Calculates Pearson correlation coefficient using NumPy."""
    if len(x) < 2:
        return 0.0
    mx = x.mean()
    my = y.mean()
    xm, ym = x - mx, y - my
    r_num = np.sum(xm * ym)
    r_den = np.sqrt(np.sum(xm**2)) * np.sqrt(np.sum(ym**2))
    if r_den == 0:
        return 0.0
    return r_num / r_den


def main():
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Optimize Config for Fast Baseline Execution on A100
    # 15 epochs is sufficient for fine-tuning on this dataset size (~7k images)
    Config.EPOCHS = 15
    Config.BATCH_SIZE = 64
    Config.NUM_WORKERS = 4

    print(f"Running on device: {device}")
    print(
        f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}, Models={Config.MODELS}"
    )

    # 2. Data Loading
    # Load cached data if available to speed up initialization
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        load_cached_data=True
    )
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # 3. Training & Calibration Loop
    trained_models = []
    calibrators = []
    val_probs_ensemble = None
    val_targets = None

    for model_name in Config.MODELS:
        print(f"\n{'='*40}")
        print(f"Training Model: {model_name}")
        print(f"{'='*40}")

        # A. Initialize Model
        model = create_model(
            model_name, num_classes=Config.NUM_CLASSES, pretrained=True
        )
        model = model.to(device)

        # B. Optimizer with Layer-Wise Learning Rate Decay (LLRD)
        param_groups = get_llrd_params(
            model,
            model_name,
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
            layer_decay=Config.LLRD_DECAY,
        )
        optimizer = optim.AdamW(param_groups)

        # C. Scheduler (Cosine Annealing)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
        )

        # D. SWA Setup
        swa_model = AveragedModel(model) if Config.USE_SWA else None
        swa_scheduler = (
            SWALR(optimizer, swa_lr=Config.SWA_LR) if Config.USE_SWA else None
        )

        # E. Training Loop
        criterion = nn.CrossEntropyLoss()
        best_val_loss = float("inf")
        best_model_path = os.path.join(Config.WORK_DIR, f"best_{model_name}.pth")

        early_stopper = EarlyStopping(patience=5, mode="min")

        for epoch in range(Config.EPOCHS):
            # Train
            train_loss = train_one_epoch(
                epoch, model, train_loader, optimizer, criterion, device, scheduler
            )

            # SWA Update (if applicable based on epoch)
            if Config.USE_SWA and epoch >= Config.SWA_START_EPOCH:
                update_swa(swa_model, model, swa_scheduler, epoch)

            # Validate
            val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_clean_checkpoint(model, best_model_path)
                print(f"  [Epoch {epoch}] New Best Val Loss: {best_val_loss:.6f}")

            early_stopper(val_loss)
            if early_stopper.early_stop:
                print("Early stopping triggered.")
                break

        # F. Load Best Model for Calibration
        print(f"Loading best checkpoint from {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        # G. Post-Hoc Calibration
        print("Calibrating model temperature...")
        # Get raw logits on validation set
        _, _, val_logits, val_labels = validate(model, val_loader, criterion, device)

        scaler = TemperatureScaler(device)
        scaler.fit(val_logits, val_labels)

        # Store model and calibrator for inference
        trained_models.append(model)
        calibrators.append(scaler)

        # Accumulate calibrated probabilities for ensemble evaluation
        calibrated_probs = scaler.predict_proba(val_logits).cpu()

        if val_probs_ensemble is None:
            val_probs_ensemble = torch.zeros_like(calibrated_probs)
            val_targets = val_labels

        val_probs_ensemble += calibrated_probs

    # 4. Ensemble Evaluation
    # Average probabilities across models
    val_probs_ensemble /= len(Config.MODELS)

    # Calculate Final Ensemble Metrics
    epsilon = 1e-15
    val_probs_clipped = torch.clamp(val_probs_ensemble, epsilon, 1 - epsilon)

    # Gather probabilities for the correct classes
    target_probs = val_probs_clipped.gather(1, val_targets.view(-1, 1)).squeeze()
    log_loss = -torch.log(target_probs).mean().item()

    # Accuracy
    _, preds = torch.max(val_probs_ensemble, 1)
    acc = (preds == val_targets).float().mean().item() * 100.0

    print(f"\n{'='*40}")
    print(f"Final Validation Metric: {log_loss}")
    print(f"Final Validation Accuracy: {acc:.2f}%")
    print(f"{'='*40}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate loss per sample
    individual_losses = -torch.log(target_probs).numpy()

    # Extract metadata features for correlation analysis
    val_df = val_loader.dataset.df
    file_sizes = []
    aspect_ratios = []

    full_paths = [os.path.join(Config.INPUT_DIR, p) for p in val_df["file_path"]]

    for path in full_paths:
        try:
            size = os.path.getsize(path)
            # Read image for aspect ratio
            img = cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
                ar = w / h if h > 0 else 0
            else:
                ar = 0
        except Exception:
            size = 0
            ar = 0

        file_sizes.append(size)
        aspect_ratios.append(ar)

    # Calculate correlations
    valid_mask = np.array(file_sizes) > 0

    if valid_mask.sum() > 0:
        corr_size = pearson_corr(
            individual_losses[valid_mask], np.array(file_sizes)[valid_mask]
        )
        corr_ar = pearson_corr(
            individual_losses[valid_mask], np.array(aspect_ratios)[valid_mask]
        )

        print(f"Correlation (Loss vs File Size): {corr_size:.6f}")
        print(f"Correlation (Loss vs Aspect Ratio): {corr_ar:.6f}")
    else:
        print("Could not calculate correlations (no valid data).")

    # 6. Submission Generation
    THRESHOLD = 0.14004325100369866

    if log_loss < THRESHOLD:
        print(
            f"\nValidation metric ({log_loss:.6f}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        all_test_probs = None

        # Iterate through models for inference
        for i, model in enumerate(trained_models):
            scaler = calibrators[i]
            model.eval()

            model_probs = []

            with torch.no_grad():
                for images, ids in test_loader:
                    images = images.to(device)

                    # Test-Time Augmentation (TTA)
                    # 1. Original Image
                    logits_orig = model(images)
                    logits_orig = scaler.forward(logits_orig)  # Apply calibration
                    probs_orig = torch.softmax(logits_orig, dim=1)

                    # 2. Horizontally Flipped Image
                    images_flip = torch.flip(images, [3])
                    logits_flip = model(images_flip)
                    logits_flip = scaler.forward(logits_flip)  # Apply calibration
                    probs_flip = torch.softmax(logits_flip, dim=1)

                    # Average TTA predictions
                    probs_avg = (probs_orig + probs_flip) / 2.0
                    model_probs.append(probs_avg.cpu())

            model_probs = torch.cat(model_probs, dim=0)

            if all_test_probs is None:
                all_test_probs = torch.zeros_like(model_probs)

            all_test_probs += model_probs

        # Average across heterogeneous ensemble
        all_test_probs /= len(trained_models)

        # Create Submission DataFrame
        test_ids = test_loader.dataset.ids
        sorted_breeds = [idx_to_class[i] for i in range(Config.NUM_CLASSES)]

        sub_df = pd.DataFrame(all_test_probs.numpy(), columns=sorted_breeds)
        sub_df.insert(0, "id", test_ids)

        submission_path = os.path.join(Config.OUTPUT_DIR, "submission.csv")
        sub_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nValidation metric ({log_loss:.6f}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
