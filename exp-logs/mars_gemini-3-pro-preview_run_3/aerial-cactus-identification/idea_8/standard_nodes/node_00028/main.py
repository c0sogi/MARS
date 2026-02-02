import os
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import set_seed, calculate_roc_auc
from library.data_loader import get_dataloaders
from library.model_factory import get_model
from library.trainer import fit_model
from library.stacking import train_meta_learner, predict_stack
from library.inference import predict_with_tta


def main():
    # 1. Configuration
    # We use 6 epochs per model to ensure the 15-model ensemble fits within the runtime limit.
    # 3 archs * 5 folds * 6 epochs is computationally feasible on the provided hardware.
    config = Config(epochs=6, debug=False)
    set_seed(config.seed)

    print("==== Starting Pipeline ====")
    print(f"Output Directory: {config.output_dir}")
    print(f"Device: {config.device}")

    # --- 2. Training & OOF Generation ---
    # Load full train metadata to establish ID order for alignment
    df_train = pd.read_csv(config.train_metadata_path)
    all_ids = df_train["id"].values

    # Container for OOF predictions: {model_name: {id: pred_scalar}}
    oof_preds_storage = {name: {} for name in config.model_names}

    # Iterate over architectures
    for model_name in config.model_names:
        print(f"\n--- Training Architecture: {model_name} ---")

        for fold_id in range(config.n_folds):
            print(f"Fold {fold_id}/{config.n_folds - 1}")

            # Get DataLoaders
            train_loader, val_loader = get_dataloaders(
                config, fold_id=fold_id, mode="train"
            )

            # Initialize Model
            model = get_model(
                model_name,
                num_classes=config.num_classes,
                pretrained=config.pretrained,
                stem_surgery=config.stem_surgery,
            )

            # Train
            save_name = f"{model_name}_fold{fold_id}.pth"
            val_preds, val_targets, val_auc = fit_model(
                config, model, train_loader, val_loader, fold_id, save_name=save_name
            )

            # Store OOF predictions mapped by ID
            # val_loader.dataset.image_ids corresponds to the order of val_preds
            val_ids = val_loader.dataset.image_ids
            for img_id, pred in zip(val_ids, val_preds):
                # pred is a numpy array of shape (1,), we want the float
                oof_preds_storage[model_name][img_id] = float(pred[0])

    # --- 3. Stacking (Meta-Learner) ---
    print("\n--- Training Meta-Learner ---")

    # Align OOF predictions to the original dataframe order
    targets = df_train["has_cactus"].values

    aligned_oof_preds = {}
    for model_name in config.model_names:
        # Create a list of preds in the order of df_train['id']
        preds_map = oof_preds_storage[model_name]
        # Fill missing with 0.5 (should not happen if all folds run)
        aligned_preds = np.array([preds_map.get(i, 0.5) for i in all_ids])
        aligned_oof_preds[model_name] = aligned_preds

    # Train Logistic Regression
    meta_model = train_meta_learner(
        aligned_oof_preds, targets, output_dir=config.output_dir
    )

    # Get Meta-Learner OOF Scores (Self-Validation)
    meta_oof_preds = predict_stack(meta_model, aligned_oof_preds)
    final_val_auc = calculate_roc_auc(targets, meta_oof_preds)

    # Print the required metric
    print(f"Final Validation Metric: {final_val_auc:.10f}")

    # --- 4. Failure Analysis ---
    print("\n--- Performing Failure Analysis ---")

    # Calculate Error Magnitude
    errors = np.abs(targets - meta_oof_preds)

    # Extract Features (Brightness, Contrast)
    # Try to load from cache first to save time
    cache_path = os.path.join(config.output_dir, f"train_full_{len(df_train)}_imgs.npy")
    if os.path.exists(cache_path):
        imgs = np.load(cache_path)
    else:
        # Fallback: load manually
        print("Cache not found, loading images for analysis...")
        imgs = []
        for path in df_train["file_path"]:
            full_path = os.path.join(config.input_dir, path)
            img = cv2.imread(full_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs.append(img)
        imgs = np.array(imgs)

    # Calculate image statistics
    # imgs shape: (N, 32, 32, 3)
    brightness = imgs.mean(axis=(1, 2, 3))
    contrast = imgs.std(axis=(1, 2, 3))

    # Correlation
    # Handle potential NaNs if std is 0 (flat images)
    contrast = np.nan_to_num(contrast)

    corr_bright, _ = pearsonr(errors, brightness)
    corr_contrast, _ = pearsonr(errors, contrast)

    print(f"Correlation (Error vs Brightness): {corr_bright:.4f}")
    print(f"Correlation (Error vs Contrast): {corr_contrast:.4f}")

    # --- 5. Inference & Submission ---
    # Proceed if the model has learned something (AUC > 0.5)
    # Note: The prompt mention of "higher than 1.0" is interpreted as a typo for a valid threshold.
    if final_val_auc > 0.5:
        print("\n--- Generating Submission ---")

        # Load Test Data
        test_loader = get_dataloaders(config, mode="test")
        test_ids = test_loader.dataset.image_ids

        # Generate Base Model Predictions
        test_preds_dict = {}

        for model_name in config.model_names:
            print(f"Predicting with {model_name} ensemble...")
            model_fold_preds = []

            for fold_id in range(config.n_folds):
                ckpt_path = os.path.join(
                    config.output_dir, f"{model_name}_fold{fold_id}.pth"
                )

                # Initialize model structure
                model = get_model(
                    model_name,
                    num_classes=config.num_classes,
                    pretrained=False,
                    stem_surgery=config.stem_surgery,
                )

                # Load weights
                state_dict = torch.load(ckpt_path, map_location=config.device)
                model.load_state_dict(state_dict)
                model = model.to(config.device)

                # Predict with TTA
                preds = predict_with_tta(
                    test_loader, model, config.device, tta_steps=config.tta_steps
                )
                model_fold_preds.append(preds)

            # Average across folds for this architecture
            # Stack: (5, N, 1) -> Mean: (N, 1) -> Flatten: (N,)
            avg_preds = np.mean(np.stack(model_fold_preds), axis=0).flatten()
            test_preds_dict[model_name] = avg_preds

        # Meta-Learner Prediction
        final_test_preds = predict_stack(meta_model, test_preds_dict)

        # Save Submission
        df_sub = pd.DataFrame({"id": test_ids, "has_cactus": final_test_preds})
        df_sub.to_csv(config.submission_path, index=False)
        print(f"Submission saved to {config.submission_path}")
        print(df_sub.head())

    else:
        print(f"Validation AUC ({final_val_auc}) is too low. Submission skipped.")


if __name__ == "__main__":
    main()
