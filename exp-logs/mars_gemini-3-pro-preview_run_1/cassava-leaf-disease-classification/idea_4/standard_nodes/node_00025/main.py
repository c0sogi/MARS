import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

# Library imports
from library.config import Config
from library.utils import seed_everything, get_logger, get_device
from library.data import get_loaders, get_test_loader
from library.model import CassavaClassifier, ModelEMA
from library.loss import SoftTargetCrossEntropy
from library.engine import train_one_epoch, validate, get_optimizer_llrd


def main():
    # --- 1. Configuration & Setup ---
    config = Config()

    # Restore full training schedule for ConvNeXt-Small
    # We allocate 12 epochs for coarse and 5 for fine to ensure convergence.
    config.epochs_coarse = 12
    config.epochs_fine = 5

    # Ensure output directories exist
    os.makedirs(config.output_dir, exist_ok=True)
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    # Initialize Logger
    logger = get_logger(os.path.join(config.output_dir, "run.log"))
    logger.info("Starting End-to-End Pipeline")

    # Set Seeds
    seed_everything(config.seed)

    # Device Setup
    device = get_device()
    logger.info(f"Using device: {device}")

    # --- 2. Model Initialization ---
    logger.info(f"Initializing Model: {config.model_name}")
    model = CassavaClassifier(config, pretrained=True)
    model.to(device)

    # Initialize EMA
    model_ema = ModelEMA(model, decay=config.ema_decay, device=device)

    # Loss Function & Scaler
    criterion = SoftTargetCrossEntropy()
    scaler = torch.amp.GradScaler("cuda")

    # --- 3. Phase 1: Warmup (Frozen Backbone) ---
    logger.info("\n=== Phase 1: Warmup ===")

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Get Coarse Loaders
    train_loader_coarse, val_loader_coarse, mixup_fn = get_loaders(
        config, phase="coarse"
    )

    # Optimizer (Head only)
    optimizer_warmup = torch.optim.AdamW(model.head.parameters(), lr=config.lr_coarse)

    for epoch in range(config.epochs_warmup):
        loss = train_one_epoch(
            epoch=epoch,
            model=model,
            loader=train_loader_coarse,
            optimizer=optimizer_warmup,
            criterion=criterion,
            device=device,
            config=config,
            scaler=scaler,
            model_ema=None,  # No EMA during warmup
            mixup_fn=mixup_fn,
            grad_accum_steps=config.grad_accum_steps_coarse,
        )
        logger.info(f"Warmup Epoch {epoch+1}/{config.epochs_warmup} - Loss: {loss:.4f}")

    # Unfreeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = True

    # --- 4. Phase 2: Coarse Training (384x384) ---
    logger.info("\n=== Phase 2: Coarse Training ===")

    optimizer = get_optimizer_llrd(model, config, config.lr_coarse)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.epochs_coarse, eta_min=config.min_lr_coarse
    )

    best_acc = 0.0
    best_model_path = os.path.join(config.output_dir, "best_model.pth")

    for epoch in range(config.epochs_coarse):
        train_loss = train_one_epoch(
            epoch=epoch,
            model=model,
            loader=train_loader_coarse,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=config,
            scaler=scaler,
            model_ema=model_ema,
            mixup_fn=mixup_fn,
            grad_accum_steps=config.grad_accum_steps_coarse,
        )

        # Validate using EMA model
        val_loss, val_acc = validate(
            model_ema.module, val_loader_coarse, criterion, device, config
        )
        scheduler.step()

        logger.info(
            f"Epoch {epoch+1}/{config.epochs_coarse} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model_ema.module.state_dict(), best_model_path)
            logger.info(f"New best model saved with accuracy: {best_acc:.4f}")

    # Reload best weights for next phase
    logger.info("Loading best coarse model for fine-tuning...")
    model.load_state_dict(torch.load(best_model_path))
    # Reset EMA to match the best model
    model_ema = ModelEMA(model, decay=config.ema_decay, device=device)

    # --- 5. Phase 3: Fine-Tuning (512x512) ---
    logger.info("\n=== Phase 3: Fine-Tuning ===")

    train_loader_fine, val_loader_fine, mixup_fn_fine = get_loaders(
        config, phase="fine"
    )

    optimizer = get_optimizer_llrd(model, config, config.lr_fine)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.epochs_fine, eta_min=config.min_lr_fine
    )

    for epoch in range(config.epochs_fine):
        train_loss = train_one_epoch(
            epoch=epoch,
            model=model,
            loader=train_loader_fine,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=config,
            scaler=scaler,
            model_ema=model_ema,
            mixup_fn=mixup_fn_fine,
            grad_accum_steps=config.grad_accum_steps_fine,
        )

        val_loss, val_acc = validate(
            model_ema.module, val_loader_fine, criterion, device, config
        )
        scheduler.step()

        logger.info(
            f"Fine Epoch {epoch+1}/{config.epochs_fine} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model_ema.module.state_dict(), best_model_path)
            logger.info(f"New best model saved with accuracy: {best_acc:.4f}")

    # --- 6. Final Evaluation & Failure Analysis ---
    logger.info("\n=== Final Evaluation & Failure Analysis ===")

    # Print Required Metric
    print(f"Final Validation Metric: {best_acc}")

    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Failure Analysis: Correlation between error and file size
    logger.info("Calculating error correlations...")
    val_metadata = pd.read_csv(config.val_metadata)

    # We need to get error per sample.
    # Use val_loader_fine (no shuffle)
    error_magnitudes = []

    with torch.no_grad():
        for inputs, targets in val_loader_fine:
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)

            # Get probability of the true class
            # targets is shape (N)
            # probs is shape (N, C)
            true_class_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            errors = 1.0 - true_class_probs
            error_magnitudes.extend(errors.cpu().tolist())

    val_metadata["error"] = error_magnitudes

    # Add file size
    val_metadata["file_size"] = val_metadata["file_path"].apply(
        lambda x: os.path.getsize(os.path.join(config.input_root, x))
    )

    corr = val_metadata["error"].corr(val_metadata["file_size"])
    print(f"Correlation between Error Magnitude and File Size: {corr:.4f}")

    # --- 7. Submission ---
    threshold = 0.9025367158754805
    if best_acc > threshold:
        logger.info("Validation threshold met. Generating submission...")

        test_loader = get_test_loader(config, phase="fine")
        test_metadata = pd.read_csv(config.test_metadata)
        image_ids = test_metadata["image_id"].values
        predictions = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)

                # TTA: Original + HFlip + VFlip
                with torch.amp.autocast("cuda"):
                    # 1. Original
                    out1 = model(inputs)
                    # 2. Horizontal Flip
                    out2 = model(torch.flip(inputs, dims=[3]))
                    # 3. Vertical Flip
                    out3 = model(torch.flip(inputs, dims=[2]))

                # Average Logits
                avg_logits = (out1 + out2 + out3) / 3.0
                preds = torch.argmax(avg_logits, dim=1).cpu().numpy()
                predictions.extend(preds)

        submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})

        sub_path = os.path.join(submission_dir, "submission.csv")
        submission_df.to_csv(sub_path, index=False)
        logger.info(f"Submission saved to {sub_path}")

    else:
        logger.info(f"Validation metric {best_acc} <= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
