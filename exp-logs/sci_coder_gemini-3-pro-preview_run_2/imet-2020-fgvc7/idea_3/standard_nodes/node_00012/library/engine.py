import os
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler

from library.config import Config
from library.utils import seed_everything, ModelEMA, optimize_threshold
from library.dataset import get_dataloaders
from library.loss import AsymmetricLoss
from library.models import ArtworkClassifier, train_one_epoch, validate, inference


def run(debug=Config.debug, epochs=Config.epochs, patience=3):
    """
    Orchestrates the training, evaluation, and submission generation for the
    Heterogeneous Ensemble of Artwork Classifiers.

    Args:
        debug (bool): If True, uses a small subset of data for debugging.
        epochs (int): Maximum number of training epochs per model.
        patience (int): Number of epochs with no improvement to wait before early stopping.
    """
    # 1. Setup Environment
    seed_everything(Config.seed)
    device = torch.device(Config.device)
    print(f"Engine initialized. Device: {device}")

    # Ensure directories exist
    os.makedirs(Config.working_dir, exist_ok=True)
    os.makedirs(Config.submission_dir, exist_ok=True)

    # 2. Prepare DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.batch_size, num_workers=Config.num_workers, debug=debug
    )

    # Storage for ensemble predictions
    ensemble_val_probs = []
    ensemble_test_probs = []
    val_targets_cache = None

    # 3. Train Each Model in the Ensemble
    for model_name in Config.model_names:
        print(f"\n--- Training Model: {model_name} ---")

        # Initialize Model
        model = ArtworkClassifier(model_name, Config.num_classes).to(device)

        # Initialize Optimizer (AdamW)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
        )

        # Initialize Scheduler (OneCycleLR)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=Config.lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=Config.pct_start,
            div_factor=Config.div_factor,
            final_div_factor=Config.final_div_factor,
        )

        # Initialize Loss and Scaler
        criterion = AsymmetricLoss()
        scaler = GradScaler()

        # Initialize EMA if enabled
        ema = None
        if Config.use_ema:
            ema = ModelEMA(model, decay=Config.ema_decay, device=device)

        # Training Loop Variables
        best_f1 = -1.0
        best_model_path = os.path.join(Config.working_dir, f"{model_name}_best.pth")
        patience_counter = 0

        for epoch in range(epochs):
            # Train one epoch
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

            # Validate
            # Use EMA model for validation if available
            eval_model = ema.module if ema else model
            val_loss, val_f1, _, _ = validate(eval_model, val_loader, criterion, device)

            # Print metrics (Full Precision)
            print(
                f"Epoch {epoch + 1} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val F1: {val_f1}"
            )

            # Early Stopping and Checkpointing
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(eval_model.state_dict(), best_model_path)
                patience_counter = 0  # Reset patience
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        print(f"Finished training {model_name}. Best F1: {best_f1}")

        # 4. Generate Predictions for Ensemble
        # Load the best model weights
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # Get Validation Probabilities (for threshold optimization)
        _, _, val_probs, val_targets = validate(model, val_loader, criterion, device)
        ensemble_val_probs.append(val_probs)
        val_targets_cache = val_targets

        # Get Test Probabilities (with Test-Time Augmentation)
        test_probs, test_ids = inference(
            model, test_loader, device, use_tta=Config.use_tta
        )
        ensemble_test_probs.append(test_probs)

        # Cleanup to free GPU memory
        del model, optimizer, scheduler, scaler, ema
        torch.cuda.empty_cache()

    # 5. Ensemble Aggregation and Submission
    print("\n--- Aggregating Ensemble ---")

    if not ensemble_val_probs:
        print("No models were trained successfully.")
        return

    # Average probabilities across all models
    avg_val_probs = np.mean(ensemble_val_probs, axis=0)
    avg_test_probs = np.mean(ensemble_test_probs, axis=0)

    # Optimize Threshold on Validation Set
    best_thresh, best_score = optimize_threshold(avg_val_probs, val_targets_cache)
    print(f"Optimal Threshold: {best_thresh} - Best Ensemble Val F1: {best_score}")

    # Apply Threshold to Test Set
    test_preds_bin = (avg_test_probs >= best_thresh).astype(int)

    # Format Submission
    submission_rows = []
    for i, img_id in enumerate(test_ids):
        # Get indices of active classes
        pred_indices = np.where(test_preds_bin[i] == 1)[0]
        # Convert to space-separated string
        pred_str = " ".join(map(str, pred_indices))
        submission_rows.append({"id": img_id, "attribute_ids": pred_str})

    # Save Submission
    df_sub = pd.DataFrame(submission_rows)
    df_sub.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
