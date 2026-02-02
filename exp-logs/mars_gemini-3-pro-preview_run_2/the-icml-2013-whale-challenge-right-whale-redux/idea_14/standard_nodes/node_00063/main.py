import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Import library modules
from library.config import Config
import library.data
import library.models
import library.engine
import library.stacking
import library.utils


def main():
    # --- 1. Configuration & Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure reproducibility
    library.utils.seed_everything(Config.SEED)

    # --- 2. Data Initialization & Split Reconstruction ---
    print("Initializing data and caching...")
    # Trigger caching by calling get_data_loaders for fold 0
    library.data.get_data_loaders(0, load_cached_data=True)

    # Load Train labels for CV splitting
    train_labels = np.load(Config.CACHE_TRAIN_LABELS)

    # Load Holdout Validation labels for final scoring
    val_labels = np.load(Config.CACHE_VAL_LABELS)

    # Reconstruct StratifiedKFold splits for TRAIN data only
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    # Split on train_labels
    splits = list(skf.split(np.zeros(len(train_labels)), train_labels))

    # Prepare containers
    # 1. train_oof_preds: OOF predictions for the Training Set (to train Meta-Learner)
    train_oof_preds = np.zeros((len(train_labels), len(Config.MODEL_NAMES)))

    # 2. val_bagged_preds: Bagged predictions for the Holdout Set (to evaluate Meta-Learner)
    val_bagged_preds = np.zeros((len(val_labels), len(Config.MODEL_NAMES)))

    # 3. test_bagged_preds: Bagged predictions for the Test Set (for submission)
    test_clips = np.load(Config.CACHE_TEST_CLIPS)
    test_bagged_preds = np.zeros((len(test_clips), len(Config.MODEL_NAMES)))

    # --- 3. Ensemble Training Loop ---
    for model_idx, model_name in enumerate(Config.MODEL_NAMES):
        print(f"\n" + "=" * 40)
        print(
            f"Processing Model {model_idx + 1}/{len(Config.MODEL_NAMES)}: {model_name}"
        )
        print("=" * 40)

        # Accumulators for bagging across folds
        current_model_val_sum = np.zeros(len(val_labels))
        current_model_test_sum = np.zeros(len(test_clips))

        for fold in range(Config.NUM_FOLDS):
            print(f"\n--- Fold {fold} ---")

            # Get DataLoaders (Train-CV split, plus Holdout loader)
            (
                train_loader,
                valid_oof_loader,
                holdout_loader,
                test_loader,
                _,
            ) = library.data.get_data_loaders(fold, load_cached_data=True)

            # Initialize Model
            model = library.models.get_model(model_name, pretrained=True).to(device)

            # Setup Optimizer and Scheduler
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.EPOCHS
            )

            # Define paths for checkpoints
            save_path = os.path.join(
                Config.WORKING_DIR, f"{model_name}_fold_{fold}.pth"
            )
            log_path = os.path.join(Config.WORKING_DIR, f"{model_name}_fold_{fold}.csv")

            # Train the model (Monitor OOF Validation Loss)
            library.engine.fit_model(
                model,
                train_loader,
                valid_oof_loader,
                optimizer,
                scheduler,
                device,
                Config.EPOCHS,
                save_path,
                log_path,
            )

            # Load the best checkpoint for inference
            library.utils.load_checkpoint(model, save_path, device=device)

            # 1. Generate OOF Predictions (for Meta-Learner Training)
            oof_probs = library.engine.predict(model, valid_oof_loader, device)

            # Map predictions back to the global Train OOF array
            _, valid_idx = splits[fold]
            train_oof_preds[valid_idx, model_idx] = oof_probs.flatten()

            # 2. Generate Holdout Predictions (Accumulate for Bagging)
            fold_val_probs = library.engine.predict(model, holdout_loader, device)
            current_model_val_sum += fold_val_probs.flatten()

            # 3. Generate Test Predictions (Accumulate for Bagging)
            fold_test_probs = library.engine.predict(model, test_loader, device)
            current_model_test_sum += fold_test_probs.flatten()

        # Average predictions for this model across all folds (Bagging)
        val_bagged_preds[:, model_idx] = current_model_val_sum / Config.NUM_FOLDS
        test_bagged_preds[:, model_idx] = current_model_test_sum / Config.NUM_FOLDS

    # --- 4. Stacking (Meta-Learner) ---
    print("\n" + "=" * 40)
    print("Training Meta-Learner (Train OOF -> Val Bagged)")
    print("=" * 40)

    # Train Meta-Learner on Train OOF predictions
    # This learns to combine the base models based on their cross-validated performance
    meta_model = library.stacking.train_meta_learner(train_oof_preds, train_labels)

    # Evaluate on Holdout Bagged predictions
    # This gives a robust estimate of ensemble performance on unseen data (Cite solution_lesson_node_00061)
    holdout_preds = library.stacking.predict_meta_learner(meta_model, val_bagged_preds)
    holdout_targets = val_labels

    # --- 5. Validation & Failure Analysis ---
    print("\n" + "=" * 40)
    print("Validation & Analysis")
    print("=" * 40)

    # Calculate Final Metric
    final_auc = roc_auc_score(holdout_targets, holdout_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    errors = np.abs(holdout_targets - holdout_preds)
    if np.std(errors) > 0 and np.std(holdout_targets) > 0:
        corr = np.corrcoef(errors, holdout_targets)[0, 1]
    else:
        corr = 0.0

    print(f"Correlation between Error and Target Label: {corr}")

    # --- 6. Submission ---
    threshold = 0.9959928858461402

    if final_auc > threshold:
        print(
            f"\nValidation metric ({final_auc:.6f}) > threshold ({threshold:.6f}). Generating submission..."
        )

        # Predict on the bagged test predictions using the trained meta-learner
        final_test_probs = library.stacking.predict_meta_learner(
            meta_model, test_bagged_preds
        )

        # Save submission
        library.engine.save_submission(
            final_test_probs, test_clips, Config.SUBMISSION_PATH
        )
    else:
        print(
            f"\nValidation metric ({final_auc:.6f}) did not meet threshold ({threshold:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    main()
