import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import timm  # Imported for validation
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import provided library modules
from library.config import Config
from library.utils import (
    seed_everything,
    compute_class_weights,
    ModelEMA,
    calculate_metric,
)
from library.dataset import get_loaders, load_data
from library.models import AppleNet
from library.train import train_one_epoch, validate
from library.inference import predict_tta


def validate_backbones():
    """
    Cite debug_lesson_6: Validate Registry Dependencies at Initialization.
    Checks if all configured backbones support the required 'features_only=True' mode.
    """
    print("Validating model configurations...")
    for name in Config.BACKBONES:
        try:
            # Attempt to create the model with the required argument
            # We use a dummy img_size to ensure compatibility checks pass
            m = timm.create_model(name, pretrained=False, features_only=True)
            print(f"  [PASS] {name} supports features_only.")
            del m
        except RuntimeError as e:
            if "features_only not implemented" in str(e):
                print(
                    f"  [FAIL] {name} does not support features_only. Please remove it from Config."
                )
                raise e
            else:
                # Some other runtime error, re-raise
                raise e
        except Exception as e:
            print(f"  [FAIL] {name} failed initialization: {e}")
            raise e


def main():
    # 1. Setup and Configuration Overrides
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline execution
    Config.EPOCHS = 10
    Config.SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    print("Initializing pipeline...")

    # Validate backbones before doing any heavy lifting
    validate_backbones()

    # 2. Data Loading
    # Load cached data if available to save time
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)
    train_df, val_df, test_df = load_data(load_cached_data=True)

    # Compute class weights for balanced loss
    class_weights = compute_class_weights(train_df)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Store predictions for ensemble
    val_preds_ensemble = []
    test_preds_ensemble = []

    # 3. Training Loop (Heterogeneous Ensemble)
    for backbone in Config.BACKBONES:
        print(f"\n{'='*40}")
        print(f"Training Backbone: {backbone}")
        print(f"{'='*40}")

        # Initialize Model
        model = AppleNet(backbone, Config.NUM_CLASSES, pretrained=True).to(
            Config.DEVICE
        )

        # Initialize EMA and Optimizer
        ema = (
            ModelEMA(model, decay=Config.EMA_DECAY, device=Config.DEVICE)
            if Config.USE_EMA
            else None
        )
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler based on total steps
        total_steps = len(train_loader) * Config.EPOCHS
        scheduler = CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=Config.MIN_LR
        )

        scaler = GradScaler(enabled=Config.USE_AMP)

        # Training State
        best_score = -1.0
        best_model_path = os.path.join(Config.WORKING_DIR, f"{backbone}_best.pth")

        for epoch in range(Config.EPOCHS):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                criterion,
                Config.DEVICE,
                scaler,
                ema,
            )

            # Validate (use EMA model if available)
            val_model = ema.module if ema else model
            val_loss, val_score, _ = validate(
                val_model, val_loader, criterion, Config.DEVICE
            )

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_score:.4f}"
            )

            # Save Best Model
            if val_score > best_score:
                best_score = val_score
                torch.save(val_model.state_dict(), best_model_path)

        # 4. Inference
        print(f"Loading best weights for {backbone}...")
        model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
        model.eval()

        # Predict on Validation Set (for Ensemble Scoring)
        _, _, v_preds = validate(model, val_loader, criterion, Config.DEVICE)
        val_preds_ensemble.append(v_preds)

        # Predict on Test Set (with TTA)
        print(f"Running TTA Inference on Test Set...")
        t_preds = predict_tta(model, test_loader, Config.DEVICE)
        test_preds_ensemble.append(t_preds)

        # Cleanup to free memory
        del model, optimizer, scaler, scheduler, ema
        torch.cuda.empty_cache()

    # 5. Ensemble Evaluation
    print("\nComputing Ensemble Metrics...")
    # Average predictions across models
    avg_val_preds = np.mean(val_preds_ensemble, axis=0)

    # Get ground truth labels
    y_true = val_loader.dataset.y

    # Calculate Final Metric
    final_metric = calculate_metric(y_true, avg_val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate Mean Absolute Error per sample
    errors = np.mean(np.abs(y_true - avg_val_preds), axis=1)

    # Extract image meta-features
    brightness_vals = []
    contrast_vals = []

    # Iterate through validation dataframe to read images
    for idx, row in val_df.iterrows():
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightness_vals.append(np.mean(gray))
                contrast_vals.append(np.std(gray))
            else:
                brightness_vals.append(0.0)
                contrast_vals.append(0.0)
        else:
            brightness_vals.append(0.0)
            contrast_vals.append(0.0)

    # Compute Correlations
    if len(errors) == len(brightness_vals) and len(errors) > 1:
        corr_bright = np.corrcoef(errors, brightness_vals)[0, 1]
        corr_contrast = np.corrcoef(errors, contrast_vals)[0, 1]
        print(f"Correlation (Error vs Brightness): {corr_bright:.4f}")
        print(f"Correlation (Error vs Contrast): {corr_contrast:.4f}")
    else:
        print("Could not compute correlations due to data mismatch.")

    # 7. Submission
    # Note: The requirement "metric > 1.0" is impossible for ROC AUC (max 1.0).
    # Proceeding with a logical threshold (> 0.5) to ensure submission generation.
    if final_metric > 0.5:
        print("Metric satisfies threshold. Generating submission file.")
        avg_test_preds = np.mean(test_preds_ensemble, axis=0)

        submission = pd.DataFrame()
        submission["image_id"] = test_df["image_id"]

        for i, label in enumerate(Config.LABELS):
            submission[label] = avg_test_preds[:, i]

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"Metric {final_metric} is too low. Submission skipped.")


if __name__ == "__main__":
    main()
