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
    # Override epochs for a fast baseline execution
    Config.EPOCHS = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure reproducibility
    library.utils.seed_everything(Config.SEED)

    # --- 2. Data Initialization & Split Reconstruction ---
    print("Initializing data and caching...")
    # Call get_data_loaders for fold 0 to ensure data is cached in ./working
    # We ignore the return values here, we just want the side effect of caching
    library.data.get_data_loaders(0, load_cached_data=True)

    # Load metadata to determine sizes and split boundaries
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Load cached labels to reconstruct the full label vector
    # Note: data.py caches them as train_labels.npy and val_labels.npy
    train_labels = np.load(Config.CACHE_TRAIN_LABELS)
    val_labels = np.load(Config.CACHE_VAL_LABELS)

    # Concatenate to match the order in data.py (Train then Val)
    full_labels = np.concatenate([train_labels, val_labels], axis=0)
    total_samples = len(full_labels)

    # Reconstruct StratifiedKFold splits to map OOF preds correctly
    # data.py uses the full concatenated data for splitting
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    # We pass zeros as X because we only need the indices based on y
    splits = list(skf.split(np.zeros(total_samples), full_labels))

    # Prepare containers for Stacking
    # oof_preds: (n_samples, n_models)
    oof_preds = np.zeros((total_samples, len(Config.MODEL_NAMES)))

    # Load test clips to initialize test prediction container
    test_clips = np.load(Config.CACHE_TEST_CLIPS)
    # test_preds_accum: (n_test_samples, n_models) - stores averaged fold preds per model
    test_preds_accum = np.zeros((len(test_clips), len(Config.MODEL_NAMES)))

    # --- 3. Ensemble Training Loop ---
    for model_idx, model_name in enumerate(Config.MODEL_NAMES):
        print(f"\n" + "=" * 40)
        print(
            f"Processing Model {model_idx + 1}/{len(Config.MODEL_NAMES)}: {model_name}"
        )
        print("=" * 40)

        # Container for summing test predictions across folds for this specific model
        current_model_test_sum = np.zeros(len(test_clips))

        for fold in range(Config.NUM_FOLDS):
            print(f"\n--- Fold {fold} ---")

            # Get DataLoaders
            # Note: get_data_loaders resets seed, ensuring consistency with our manual splits
            train_loader, val_loader, test_loader, _ = library.data.get_data_loaders(
                fold, load_cached_data=True
            )

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

            # Train the model
            library.engine.fit_model(
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                device,
                Config.EPOCHS,
                save_path,
                log_path,
            )

            # Load the best checkpoint for inference
            library.utils.load_checkpoint(model, save_path, device=device)

            # Generate OOF Predictions
            # val_loader contains the validation data for this fold
            val_probs = library.engine.predict(model, val_loader, device)

            # Map predictions back to the global OOF array
            # The val_loader iterates over the data at indices 'valid_idx' from the split
            _, valid_idx = splits[fold]

            # Safety check for size alignment
            if len(val_probs) != len(valid_idx):
                raise ValueError(
                    f"Size mismatch: Preds {len(val_probs)} vs Indices {len(valid_idx)}"
                )

            oof_preds[valid_idx, model_idx] = val_probs.flatten()

            # Generate Test Predictions
            fold_test_probs = library.engine.predict(model, test_loader, device)
            current_model_test_sum += fold_test_probs.flatten()

        # Average test predictions for this model across all folds
        test_preds_accum[:, model_idx] = current_model_test_sum / Config.NUM_FOLDS

    # --- 4. Stacking (Meta-Learner) ---
    print("\n" + "=" * 40)
    print("Training Meta-Learner")
    print("=" * 40)

    # Train Logistic Regression on OOF predictions
    meta_model = library.stacking.train_meta_learner(oof_preds, full_labels)

    # Generate predictions on the full OOF set using the meta-learner
    meta_oof_probs = library.stacking.predict_meta_learner(meta_model, oof_preds)

    # --- 5. Validation & Failure Analysis ---
    print("\n" + "=" * 40)
    print("Validation & Analysis")
    print("=" * 40)

    # Isolate the predictions for the original 'val.csv' hold-out set
    # The full_data is [Train_Data, Val_Data], so Val starts after Train
    val_start_idx = len(train_labels)

    holdout_preds = meta_oof_probs[val_start_idx:]
    holdout_targets = full_labels[val_start_idx:]

    # Calculate Final Metric
    final_auc = roc_auc_score(holdout_targets, holdout_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis: Correlation between Error and Target
    # Error = |Target - Prediction|
    errors = np.abs(holdout_targets - holdout_preds)

    # Calculate Point-Biserial Correlation (Pearson between continuous error and binary target)
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

        # Predict on the accumulated test predictions
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
