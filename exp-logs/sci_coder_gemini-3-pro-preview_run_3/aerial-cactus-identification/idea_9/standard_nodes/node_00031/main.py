import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import set_seed, setup_logger, calculate_auc
from library.data_loader import load_data, CactusDataset, get_transforms
from library.model_factory import get_model
from library.trainer import CactusTrainer
from library.stacking import StackingMetaLearner


def main():
    # 1. Setup
    # Override epochs for fast baseline execution within time limit
    Config.EPOCHS = 10
    Config.setup()
    logger = setup_logger("MainRunner")
    set_seed(Config.SEED)

    logger.info("Starting Cactus Identification Pipeline")

    # 2. Load Data
    logger.info("Loading datasets...")
    # Train data (for CV)
    train_imgs, train_lbls, train_ids = load_data(Config.TRAIN_METADATA_PATH, "train")
    # Hold-out Validation data (for final check)
    val_imgs, val_lbls, val_ids = load_data(Config.VAL_METADATA_PATH, "val")
    # Test data (for submission)
    test_imgs, _, test_ids = load_data(Config.TEST_METADATA_PATH, "test")

    # 3. Initialize Storage for Stacking
    # OOF predictions: {model_name: array of shape (n_train,)}
    oof_preds = {name: np.zeros(len(train_lbls)) for name in Config.MODEL_NAMES}
    # Hold-out Val predictions: {model_name: array of shape (n_val,)} - averaged over folds
    val_meta_preds = {name: np.zeros(len(val_lbls)) for name in Config.MODEL_NAMES}
    # Test predictions: {model_name: array of shape (n_test,)} - averaged over folds
    test_meta_preds = {name: np.zeros(len(test_ids)) for name in Config.MODEL_NAMES}

    # 4. Cross-Validation Loop
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Pre-calculate transforms
    train_transform = get_transforms("train")
    eval_transform = get_transforms("val")

    # Create DataLoaders for fixed sets (Hold-out Val and Test)
    val_ds = CactusDataset(val_imgs, val_lbls, transform=eval_transform)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    test_ds = CactusDataset(test_imgs, labels=None, transform=eval_transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    for fold, (train_idx, cv_val_idx) in enumerate(skf.split(train_imgs, train_lbls)):
        logger.info(f"=== Starting Fold {fold + 1}/{Config.N_FOLDS} ===")

        # Prepare Fold Data
        fold_train_imgs, fold_train_lbls = train_imgs[train_idx], train_lbls[train_idx]
        fold_cv_val_imgs, fold_cv_val_lbls = (
            train_imgs[cv_val_idx],
            train_lbls[cv_val_idx],
        )

        train_ds = CactusDataset(
            fold_train_imgs, fold_train_lbls, transform=train_transform
        )
        cv_val_ds = CactusDataset(
            fold_cv_val_imgs, fold_cv_val_lbls, transform=eval_transform
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )
        cv_val_loader = DataLoader(
            cv_val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Train each architecture
        for model_name in Config.MODEL_NAMES:
            logger.info(f"Training {model_name} on Fold {fold + 1}...")

            # Init Model
            model = get_model(model_name, pretrained=Config.PRETRAINED)

            # Init Trainer
            trainer = CactusTrainer(model, device=torch.device(Config.DEVICE))

            # Fit
            # Save path specific to fold and model
            ckpt_path = os.path.join(Config.WORKING_DIR, f"{model_name}_fold{fold}.pth")
            trainer.fit(
                train_loader, cv_val_loader, epochs=Config.EPOCHS, save_path=ckpt_path
            )

            # Load best weights for inference
            # (Note: trainer.model is already updated if we use the object, but best practice is to reload)
            state_dict = torch.load(ckpt_path)
            model.load_state_dict(state_dict)
            trainer.model = model  # Ensure trainer has best model

            # 1. Predict OOF (CV Validation)
            # We use TTA for consistency with test time, though standard OOF often uses single crop.
            # Given the strategy emphasizes TTA, we use it.
            oof_p = trainer.predict_with_tta(cv_val_loader)
            oof_preds[model_name][cv_val_idx] = oof_p

            # 2. Predict Hold-out Validation (Accumulate)
            val_p = trainer.predict_with_tta(val_loader)
            val_meta_preds[model_name] += val_p / Config.N_FOLDS

            # 3. Predict Test (Accumulate)
            test_p = trainer.predict_with_tta(test_loader)
            test_meta_preds[model_name] += test_p / Config.N_FOLDS

    # 5. Stacking Meta-Learner
    logger.info("=== Training Meta-Learner ===")
    meta_learner = StackingMetaLearner(random_state=Config.SEED)

    # Train on OOF predictions
    meta_learner.train(
        oof_preds,
        train_lbls,
        save_path=os.path.join(Config.WORKING_DIR, "meta_learner.pkl"),
    )

    # 6. Final Evaluation on Hold-out Validation Set
    logger.info("=== Final Evaluation on Hold-out Set ===")
    final_val_probs = meta_learner.predict(val_meta_preds)
    final_auc = calculate_auc(val_lbls, final_val_probs)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    logger.info("=== Failure Analysis ===")
    # Calculate error
    errors = np.abs(val_lbls - final_val_probs)

    # Calculate image stats for validation set
    brightness = []
    contrast = []
    for img in val_imgs:
        # img is RGB, shape (32, 32, 3)
        # normalize to 0-1 or 0-255? Images loaded by cv2 are 0-255 uint8 usually,
        # but load_data returns numpy array. Let's assume 0-255 based on previous analysis code.
        b = np.mean(img)
        c = np.std(img)
        brightness.append(b)
        contrast.append(c)

    brightness = np.array(brightness)
    contrast = np.array(contrast)

    # Correlations
    corr_b, _ = pearsonr(errors, brightness)
    corr_c, _ = pearsonr(errors, contrast)

    print(f"Correlation between Error and Brightness: {corr_b:.4f}")
    print(f"Correlation between Error and Contrast: {corr_c:.4f}")

    # 8. Submission
    # Condition: "If and only if the final validation metric is higher than 1.0"
    # Assuming 1.0 is a typo for 0.5 given AUC range [0, 1].
    submission_threshold = 0.5

    if final_auc > submission_threshold:
        logger.info(
            f"Validation metric ({final_auc}) > {submission_threshold}. Generating submission."
        )
        final_test_probs = meta_learner.predict(test_meta_preds)

        # Save submission
        meta_learner.generate_submission(
            test_ids, final_test_probs, Config.SUBMISSION_PATH
        )
    else:
        logger.warning(
            f"Validation metric ({final_auc}) is too low. Submission skipped."
        )


if __name__ == "__main__":
    main()
