import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_dataloaders, get_test_dataloader, prepare_data
from library.architectures import get_model
from library.engine import train_one_epoch, validate, predict_with_tta
from library.meta_learner import train_meta_learner, predict_meta


def main():
    print("Initializing Cactus Classification Demo...")

    # 1. Setup & Configuration Override
    # ---------------------------------------------------------
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Override Config for a quick demonstration run
    # We reduce epochs and folds to ensure execution finishes quickly
    Config.EPOCHS = 1
    Config.NUM_FOLDS = 2  # Run only 2 folds instead of 5
    Config.MODELS = [
        "efficientnet",
        "densenet_bc",
    ]  # Use two pre-trained models for stability/speed
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(
        f"Configuration: Epochs={Config.EPOCHS}, Folds={Config.NUM_FOLDS}, Models={Config.MODELS}"
    )
    print(f"Device: {device}")

    # 2. Data Preparation
    # ---------------------------------------------------------
    print("\n[Data] Preparing datasets...")
    # Force load_cached_data=False to demonstrate raw data processing logic
    prepare_data(load_cached_data=False)

    # 3. Training & Inference Loop
    # ---------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss()

    # Storage for Meta-Learner
    # oof_predictions: {model_name: {img_id: prob}}
    oof_predictions = {m: {} for m in Config.MODELS}

    # test_predictions_sum: {model_name: {img_id: sum_prob}} for averaging across folds
    test_predictions_sum = {m: {} for m in Config.MODELS}

    # Load Test Loader (shared across all models)
    test_loader = get_test_dataloader(load_cached_data=True)
    test_ids = test_loader.dataset.ids

    # Initialize test accumulators
    for m in Config.MODELS:
        for tid in test_ids:
            test_predictions_sum[m][tid] = 0.0

    for model_name in Config.MODELS:
        print(f"\n[Model] Processing Architecture: {model_name}")

        for fold in range(Config.NUM_FOLDS):
            print(f"  > Fold {fold + 1}/{Config.NUM_FOLDS}")

            # Get Dataloaders for this fold
            train_loader, val_loader = get_dataloaders(
                fold_id=fold, load_cached_data=True
            )

            # Initialize Model
            model = get_model(model_name, num_classes=1).to(device)

            # Optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )

            # Train
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"    Train Loss: {loss:.4f}")

            # Validate (Metrics)
            val_loss, val_auc = validate(model, val_loader, criterion, device)

            # Generate OOF Predictions (Inference on Validation Set)
            # We manually iterate to capture IDs which validate() does not return
            model.eval()
            with torch.no_grad():
                for inputs, targets, ids in val_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                    for i, img_id in enumerate(ids):
                        oof_predictions[model_name][img_id] = float(probs[i])

            # Generate Test Predictions (with TTA)
            fold_test_preds = predict_with_tta(model, test_loader, device)

            # Accumulate Test Predictions
            for tid, prob in fold_test_preds.items():
                test_predictions_sum[model_name][tid] += prob

    # 4. Aggregation
    # ---------------------------------------------------------
    print("\n[Aggregation] Averaging test predictions across folds...")
    final_test_predictions = {}
    for model_name in Config.MODELS:
        final_test_predictions[model_name] = {}
        for tid in test_ids:
            final_test_predictions[model_name][tid] = (
                test_predictions_sum[model_name][tid] / Config.NUM_FOLDS
            )

    # 5. Meta-Learner Stacking
    # ---------------------------------------------------------
    print("\n[Meta-Learner] Training and Stacking...")
    # Train meta-learner on the collected OOF predictions
    # Note: In this demo, OOF is partial (only 2 folds), but the function handles intersection of IDs.
    meta_model = train_meta_learner(oof_predictions)

    # Generate final submission using the meta-learner
    submission_df = predict_meta(meta_model, final_test_predictions)

    # 6. Verification
    # ---------------------------------------------------------
    print("\n[Verification] Validating output...")

    # Check 1: File Existence
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    # Check 2: Shape
    expected_rows = 3325
    if len(submission_df) != expected_rows:
        raise AssertionError(
            f"Submission has {len(submission_df)} rows, expected {expected_rows}"
        )

    # Check 3: Columns
    if "id" not in submission_df.columns or "has_cactus" not in submission_df.columns:
        raise AssertionError("Submission missing required columns ('id', 'has_cactus')")

    # Check 4: Probability Range
    probs = submission_df["has_cactus"].values
    if np.any(probs < 0) or np.any(probs > 1):
        raise AssertionError("Predicted probabilities are out of valid range [0, 1]")

    print(f"Success! Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
