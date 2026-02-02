import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from library.config import Config
from library.utils import (
    get_device,
    reconstruct_4_class_probabilities,
    get_binary_targets,
    seed_everything,
)
from library.data import get_test_loader, get_train_val_loaders, get_folds_data
from library.models import AppleNet


def predict_loader(model, loader, device):
    """
    Generates predictions for a data loader using TTA (Horizontal Flip).
    Returns raw probabilities (sigmoid applied).
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            # Unpack batch (handle both labeled and unlabeled loaders)
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device)

            # Forward pass 1: Original
            logits1 = model(images)
            probs1 = torch.sigmoid(logits1)

            # Forward pass 2: Horizontal Flip TTA
            images_flipped = torch.flip(images, dims=[3])
            logits2 = model(images_flipped)
            probs2 = torch.sigmoid(logits2)

            # Average probabilities
            avg_probs = (probs1 + probs2) / 2.0

            preds.append(avg_probs.cpu().numpy())

    return np.concatenate(preds, axis=0)


def get_oof_predictions(load_cached_data=True):
    """
    Generates or loads Out-Of-Fold predictions for all models and folds.
    Returns:
        np.array: (N_samples, N_architectures, 2)
    """
    cache_path = os.path.join(Config.WORKING_DIR, "oof_predictions.npy")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached OOF predictions from {cache_path}")
        return np.load(cache_path)

    print("Generating OOF predictions...")
    device = get_device()
    df = get_folds_data(load_cached_data=True)
    num_samples = len(df)
    num_architectures = len(Config.MODELS)

    # Initialize container: (N_samples, N_architectures, 2_targets)
    oof_preds = np.zeros((num_samples, num_architectures, 2))

    for model_idx, model_cfg in enumerate(Config.MODELS):
        model_name = model_cfg["name"]
        print(f"Processing OOF for architecture: {model_name}")

        for fold in range(Config.N_FOLDS):
            # Load Validation Loader
            _, val_loader = get_train_val_loaders(
                fold, model_cfg["img_size"], model_cfg["batch_size"]
            )

            # Load Model
            checkpoint_filename = f"best_model_{model_name}_fold_{fold}.pth"
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, checkpoint_filename)

            if not os.path.exists(checkpoint_path):
                print(
                    f"  Warning: Checkpoint {checkpoint_filename} not found. Skipping."
                )
                continue

            model = AppleNet(
                model_name=model_name,
                pretrained=False,
                dropout_rates=model_cfg.get("dropout_rates"),
            )
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)

            # Predict
            fold_preds = predict_loader(model, val_loader, device)

            # Map predictions to original DataFrame indices
            val_indices = df[df["fold"] == fold].index

            # Safety check
            if len(fold_preds) != len(val_indices):
                raise ValueError(
                    f"Prediction length ({len(fold_preds)}) does not match "
                    f"validation set length ({len(val_indices)}) for fold {fold}."
                )

            oof_preds[val_indices, model_idx, :] = fold_preds

            # Cleanup
            del model, checkpoint
            torch.cuda.empty_cache()

    # Cache results
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(cache_path, oof_preds)

    return oof_preds


def get_test_predictions_raw(load_cached_data=True):
    """
    Generates or loads raw test set predictions.
    Averages predictions across folds for each architecture.
    Returns:
        np.array: (N_test, N_architectures, 2)
    """
    cache_path = os.path.join(Config.WORKING_DIR, "test_predictions_raw.npy")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached raw test predictions from {cache_path}")
        return np.load(cache_path)

    print("Generating raw test predictions...")
    device = get_device()
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    num_test = len(test_df)
    num_architectures = len(Config.MODELS)

    test_preds = np.zeros((num_test, num_architectures, 2))

    for model_idx, model_cfg in enumerate(Config.MODELS):
        model_name = model_cfg["name"]
        print(f"Processing Test for architecture: {model_name}")

        # Accumulate predictions across folds
        arch_preds_sum = np.zeros((num_test, 2))
        folds_count = 0

        test_loader = get_test_loader(model_cfg["img_size"], model_cfg["batch_size"])

        for fold in range(Config.N_FOLDS):
            checkpoint_filename = f"best_model_{model_name}_fold_{fold}.pth"
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, checkpoint_filename)

            if not os.path.exists(checkpoint_path):
                continue

            model = AppleNet(
                model_name=model_name,
                pretrained=False,
                dropout_rates=model_cfg.get("dropout_rates"),
            )
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)

            fold_preds = predict_loader(model, test_loader, device)
            arch_preds_sum += fold_preds
            folds_count += 1

            del model, checkpoint
            torch.cuda.empty_cache()

        if folds_count > 0:
            test_preds[:, model_idx, :] = arch_preds_sum / folds_count
        else:
            print(
                f"Warning: No checkpoints found for {model_name}. Predictions will be 0."
            )

    # Cache results
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(cache_path, test_preds)

    return test_preds


def run_inference():
    """
    Main inference pipeline:
    1. Load/Generate OOF predictions.
    2. Load/Generate Test predictions.
    3. Train Rank-Based Stacking Meta-Learner.
    4. Generate final submission.
    """
    seed_everything(Config.SEED)

    # 1. Prepare Data for Meta-Learner
    # X_train: OOF predictions (N_train, N_arch, 2)
    oof_preds = get_oof_predictions(load_cached_data=True)

    # y_train: Ground truth binary targets
    df_folds = get_folds_data(load_cached_data=True)
    y_train = get_binary_targets(df_folds)

    # X_test: Raw test predictions (N_test, N_arch, 2)
    test_preds_raw = get_test_predictions_raw(load_cached_data=True)

    num_test = test_preds_raw.shape[0]
    final_test_preds = np.zeros((num_test, 2))  # [rust, scab]

    print("Training Rank-Calibrated Stacking Meta-Learners...")

    # Train a separate meta-learner for each binary target (Rust, Scab)
    target_names = Config.TARGET_COLS  # ["rust", "scab"]

    for i, target_name in enumerate(target_names):
        # Extract features for this target from all architectures
        # Shape: (N_train, N_arch)
        X_train_feat = oof_preds[:, :, i]
        X_test_feat = test_preds_raw[:, :, i]

        # Rank Normalization (Column-wise)
        # Convert probabilities to ranks (0 to 1) to handle calibration drift
        X_train_ranked = np.apply_along_axis(
            lambda x: rankdata(x, method="average"), axis=0, arr=X_train_feat
        )
        X_train_ranked = X_train_ranked / X_train_ranked.shape[0]

        X_test_ranked = np.apply_along_axis(
            lambda x: rankdata(x, method="average"), axis=0, arr=X_test_feat
        )
        X_test_ranked = X_test_ranked / X_test_ranked.shape[0]

        # Meta-Model: Logistic Regression
        meta_learner = LogisticRegression(random_state=Config.SEED)
        meta_learner.fit(X_train_ranked, y_train[:, i])

        # Predict on Test Set
        # predict_proba returns [prob_0, prob_1]
        final_test_preds[:, i] = meta_learner.predict_proba(X_test_ranked)[:, 1]

        print(
            f"  Target '{target_name}': Intercept={meta_learner.intercept_[0]:.4f}, "
            f"Coefficients={meta_learner.coef_[0]}"
        )

    # 2. Reconstruct 4-Class Probabilities
    # Rust (index 0) and Scab (index 1) from final_test_preds
    final_probs_4class = reconstruct_4_class_probabilities(
        final_test_preds[:, 0], final_test_preds[:, 1]
    )

    # 3. Create Submission File
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    submission_df = pd.DataFrame(
        {
            "image_id": test_df["image_id"],
            "healthy": final_probs_4class[:, 0],
            "multiple_diseases": final_probs_4class[:, 1],
            "rust": final_probs_4class[:, 2],
            "scab": final_probs_4class[:, 3],
        }
    )

    # Ensure correct column order
    cols_order = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    submission_df = submission_df[cols_order]

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
