import os
import gc
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge

from library.config import Config
from library.utils import get_logger, seed_everything
from library.feature_engineering import generate_svd_features
from library.dataset import InsultDataset
from library.models import HybridModel
from library.engine import train_fold, inference_fn

logger = get_logger("pipeline")


def run_kfold_cv(model_name):
    """
    Runs Stratified K-Fold Cross-Validation for a specific model architecture.
    Handles data splitting, feature engineering, training, and inference.

    Args:
        model_name (str): The HuggingFace model identifier (e.g., 'microsoft/deberta-v3-large').

    Returns:
        tuple: (oof_preds, test_preds, y_labels)
            - oof_preds (np.ndarray): Out-of-fold predictions for the training set.
            - test_preds (np.ndarray): Averaged predictions for the test set.
            - y_labels (np.ndarray): Ground truth labels corresponding to oof_preds.
    """
    seed_everything(Config.seed)

    logger.info(f"Starting K-Fold CV for model: {model_name}")

    # 1. Load and Prepare Data
    # We combine train and validation metadata to perform our own Stratified K-Fold
    df_train_part = pd.read_csv(Config.TRAIN_PATH)
    df_val_part = pd.read_csv(Config.VAL_PATH)
    df_test = pd.read_csv(Config.TEST_PATH)

    df_train = pd.concat([df_train_part, df_val_part]).reset_index(drop=True)

    # Extract raw text and labels
    train_texts = df_train["Comment"].fillna("").astype(str).values
    train_labels = df_train["Insult"].values
    test_texts = df_test["Comment"].fillna("").astype(str).values

    # Initialize containers
    oof_preds = np.zeros(len(df_train))
    test_preds_list = []

    # 2. K-Fold Setup
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. Iterate Folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
        if fold not in Config.trn_folds:
            logger.info(f"Skipping Fold {fold} (not in Config.trn_folds)")
            continue

        logger.info(f"\n{'='*20} Running Fold {fold} for {model_name} {'='*20}")

        # Split Data
        X_train, X_val = train_texts[train_idx], train_texts[val_idx]
        y_train, y_val = train_labels[train_idx], train_labels[val_idx]

        # Generate Structural SVD Features
        # This function handles caching and strict isolation (fit on train, transform on val/test)
        train_svd, val_svd, test_svd = generate_svd_features(
            X_train, X_val, test_texts, fold_idx=fold, load_cached_data=True
        )

        # Create Datasets
        train_ds = InsultDataset(X_train, train_svd, tokenizer, labels=y_train)
        val_ds = InsultDataset(X_val, val_svd, tokenizer, labels=y_val)
        test_ds = InsultDataset(test_texts, test_svd, tokenizer)

        # Create DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Initialize Model
        device = Config.device
        # Use actual SVD dimension which might be smaller than Config.svd_components due to small dataset size
        actual_svd_dim = train_svd.shape[1]
        model = HybridModel(model_name, svd_dim=actual_svd_dim, pretrained=True)
        model.to(device)

        # Optimizer with Differential Learning Rates
        # Separate backbone parameters from custom head parameters
        backbone_params = list(model.backbone.named_parameters())
        backbone_ids = {id(p) for n, p in backbone_params}

        head_params = [
            (n, p) for n, p in model.named_parameters() if id(p) not in backbone_ids
        ]

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in backbone_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": Config.weight_decay,
                "lr": Config.lr_backbone,
            },
            {
                "params": [
                    p for n, p in backbone_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": Config.lr_backbone,
            },
            {
                "params": [
                    p for n, p in head_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": Config.weight_decay,
                "lr": Config.lr_head,
            },
            {
                "params": [
                    p for n, p in head_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": Config.lr_head,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, eps=1e-6)

        # Scheduler
        num_training_steps = len(train_loader) * Config.epochs
        num_warmup_steps = int(num_training_steps * Config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=Config.num_cycles,
        )

        # Train Fold
        train_fold(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            device,
            fold,
            model_name,
        )

        # Inference on Validation and Test
        # Load best model weights
        sanitized_name = model_name.split("/")[-1]
        model_path = os.path.join(Config.MODEL_DIR, f"{sanitized_name}_fold_{fold}.bin")
        logger.info(f"Loading best model from {model_path} for inference...")
        model.load_state_dict(torch.load(model_path, map_location=device))

        # Generate predictions
        val_preds_fold = inference_fn(val_loader, model, device).flatten()
        test_preds_fold = inference_fn(test_loader, model, device).flatten()

        # Store predictions
        oof_preds[val_idx] = val_preds_fold
        test_preds_list.append(test_preds_fold)

        # Cleanup to save memory
        del (
            model,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            test_loader,
            train_ds,
            val_ds,
            test_ds,
        )
        del train_svd, val_svd, test_svd
        torch.cuda.empty_cache()
        gc.collect()

    # Average Test Predictions across folds
    if test_preds_list:
        avg_test_preds = np.mean(test_preds_list, axis=0)
    else:
        logger.warning("No folds were run! Returning zeros.")
        avg_test_preds = np.zeros(len(df_test))

    return oof_preds, avg_test_preds, train_labels


def train_meta_learner(oof_preds_dict, y_true):
    """
    Trains a Ridge Regression meta-learner on the stacked OOF predictions.

    Args:
        oof_preds_dict (dict): Dictionary mapping model names to their OOF prediction arrays.
        y_true (np.ndarray): Ground truth labels.

    Returns:
        sklearn.linear_model.Ridge: The trained meta-model.
    """
    logger.info("Training Meta-Learner (Ridge Regression)...")

    # Create feature matrix: (n_samples, n_models)
    X_meta = np.column_stack([preds for preds in oof_preds_dict.values()])

    meta_model = Ridge(alpha=Config.meta_alpha, random_state=Config.seed)
    meta_model.fit(X_meta, y_true)

    logger.info(f"Meta-Learner Coefficients: {meta_model.coef_}")
    logger.info(f"Meta-Learner Intercept: {meta_model.intercept_}")

    return meta_model


def predict_meta_learner(meta_model, test_preds_dict):
    """
    Generates final predictions using the trained meta-learner.

    Args:
        meta_model: Trained Ridge model.
        test_preds_dict (dict): Dictionary mapping model names to their Test prediction arrays.

    Returns:
        np.ndarray: Final ensemble predictions clipped to [0, 1].
    """
    X_test_meta = np.column_stack([preds for preds in test_preds_dict.values()])
    final_preds = meta_model.predict(X_test_meta)

    # Clip probabilities to valid range
    final_preds = np.clip(final_preds, 0.0, 1.0)

    return final_preds


def save_submission(predictions):
    """
    Saves the final predictions to the submission CSV file.

    Args:
        predictions (np.ndarray): The probability scores for the test set.
    """
    logger.info(f"Saving submission to {Config.SUBMISSION_PATH}...")

    # Load test set to preserve structure
    df_test = pd.read_csv(Config.TEST_PATH)

    # Create submission dataframe
    submission = df_test.copy()
    submission["Insult"] = predictions

    # Reorder columns to match sample format: Insult, Date, Comment
    if "Date" in submission.columns and "Comment" in submission.columns:
        submission = submission[["Insult", "Date", "Comment"]]

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info("Submission saved successfully.")
