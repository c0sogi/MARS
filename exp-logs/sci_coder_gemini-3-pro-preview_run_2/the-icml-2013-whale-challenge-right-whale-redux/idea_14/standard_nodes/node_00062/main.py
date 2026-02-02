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

    # --- 2. Data Initialization ---
    print("Initializing data and caching...")
    # Trigger caching by calling loader for fold 0
    library.data.get_data_loaders(0, load_cached_data=True)

    # Load labels
    train_labels = np.load(Config.CACHE_TRAIN_LABELS)
    val_labels = np.load(Config.CACHE_VAL_LABELS)  # Hold-out labels

    # Reconstruct StratifiedKFold splits for TRAIN data
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    splits = list(skf.split(np.zeros(len(train_labels)), train_labels))

    # Prepare containers
    # OOF Predictions for Meta-Learner Training (on Train Set)
    oof_preds = np.zeros((len(train_labels), len(Config.MODEL_NAMES)))

    # Bagged Predictions for Meta-Learner Evaluation (on Hold-out Val Set)
    holdout_preds_accum = np.zeros((len(val_labels), len(Config.MODEL_NAMES)))

    # Bagged Predictions for Test Set
    test_clips = np.load(Config.CACHE_TEST_CLIPS)
    test_preds_accum = np.zeros((len(test_clips), len(Config.MODEL_NAMES)))

    # --- 3. Ensemble Training Loop ---
    for model_idx, model_name in enumerate(Config.MODEL_NAMES):
        print(f"\n" + "=" * 40)
        print(
            f"Processing Model {model_idx + 1}/{len(Config.MODEL_NAMES)}: {model_name}"
        )
        print("=" * 40)

        # Accumulators for this specific model across folds
        model_holdout_sum = np.zeros(len(val_labels))
        model_test_sum = np.zeros(len(test_clips))

        for fold in range(Config.NUM_FOLDS):
            print(f"\n--- Fold {fold} ---")

            # Get DataLoaders (Strict Train/Val split + Holdout)
            (
                train_loader,
                valid_loader,
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

            # Define paths
            save_path = os.path.join(
                Config.WORKING_DIR, f"{model_name}_fold_{fold}.pth"
            )
            log_path = os.path.join(Config.WORKING_DIR, f"{model_name}_fold_{fold}.csv")

            # Train the model (Early Stopping on Fold-Val Loss)
            library.engine.fit_model(
                model,
                train_loader,
                valid_loader,
                optimizer,
                scheduler,
                device,
                Config.EPOCHS,
                save_path,
                log_path,
            )

            # Load best checkpoint
            library.utils.load_checkpoint(model, save_path, device=device)

            # 1. Generate OOF Predictions (for Stacking Training)
            val_probs = library.engine.predict(model, valid_loader, device)
            _, valid_idx = splits[fold]
            oof_preds[valid_idx, model_idx] = val_probs.flatten()

            # 2. Generate Hold-out Predictions (for Stacking Evaluation - Bagging)
            # Cite Lesson 00061: Evaluate bagged predictions on hold-out set
            holdout_probs = library.engine.predict(model, holdout_loader, device)
            model_holdout_sum += holdout_probs.flatten()

            # 3. Generate Test Predictions (Bagging)
            test_probs = library.engine.predict(model, test_loader, device)
            model_test_sum += test_probs.flatten()

        # Average predictions across folds (Bagging)
        holdout_preds_accum[:, model_idx] = model_holdout_sum / Config.NUM_FOLDS
        test_preds_accum[:, model_idx] = model_test_sum / Config.NUM_FOLDS

    # --- 4. Stacking (Meta-Learner) ---
    print("\n" + "=" * 40)
    print("Training Meta-Learner (Stacking on OOF, Evaluating on Bagged Holdout)")
    print("=" * 40)

    # Train Meta-Learner on Train OOF Predictions
    # Cite Lesson 00053: Heterogeneous Stacking
    meta_model = library.stacking.train_meta_learner(oof_preds, train_labels)

    # Evaluate Meta-Learner on Bagged Hold-out Predictions
    holdout_final_probs = library.stacking.predict_meta_learner(
        meta_model, holdout_preds_accum
    )

    # --- 5. Validation & Failure Analysis ---
    print("\n" + "=" * 40)
    print("Validation & Analysis")
    print("=" * 40)

    # Calculate Final Metric on Hold-out Set
    final_auc = roc_auc_score(val_labels, holdout_final_probs)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    errors = np.abs(val_labels - holdout_final_probs)
    if np.std(errors) > 0 and np.std(val_labels) > 0:
        corr = np.corrcoef(errors, val_labels)[0, 1]
    else:
        corr = 0.0

    print(f"Correlation between Error and Target Label: {corr}")

    # --- 6. Submission ---
    threshold = 0.9959928858461402

    if final_auc > threshold:
        print(
            f"\nValidation metric ({final_auc:.6f}) > threshold ({threshold:.6f}). Generating submission..."
        )

        # Predict on the accumulated test predictions using the trained meta-learner
        final_test_probs = library.stacking.predict_meta_learner(
            meta_model, test_preds_accum
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
