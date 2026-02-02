import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

from library.config import CFG
from library.utils import seed_everything, get_logger, get_score, OptimizedRounder
from library.features import FeatureEngineer
from library.data import get_loaders, get_test_loader
from library.neural_net import EssayModel
from library.classic_models import get_ridge_pipeline, get_lgbm_stacker
from library.engine import AWP, train_one_epoch, valid_one_epoch, inference_fn

logger = get_logger(os.path.join(CFG.output_dir, "workflow"))


def run_nn_fold(
    fold, train_loader, val_loader, test_loader, device, load_cached_data=True
):
    """
    Runs the Neural Network training and inference for a single fold.
    Manages caching of OOF and Test predictions.
    """
    cache_oof_path = os.path.join(CFG.output_dir, f"nn_oof_fold_{fold}.npy")
    cache_test_path = os.path.join(CFG.output_dir, f"nn_test_fold_{fold}.npy")
    cache_labels_path = os.path.join(CFG.output_dir, f"nn_labels_fold_{fold}.npy")

    # Check cache
    if (
        load_cached_data
        and os.path.exists(cache_oof_path)
        and os.path.exists(cache_test_path)
    ):
        logger.info(f"[NN Fold {fold}] Loading cached predictions...")
        val_preds = np.load(cache_oof_path)
        test_preds = np.load(cache_test_path)
        # Try loading labels if they exist (for validation scoring outside)
        if os.path.exists(cache_labels_path):
            val_labels = np.load(cache_labels_path)
        else:
            # If labels aren't cached, we extract them from loader (a bit slow but safe)
            val_labels = []
            for batch in val_loader:
                val_labels.append(batch["labels"].numpy())
            val_labels = np.concatenate(val_labels)
        return val_preds, test_preds, val_labels

    logger.info(f"[NN Fold {fold}] Training started...")

    # Initialize Model
    model = EssayModel(pretrained=True).to(device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )

    # Scheduler
    num_train_steps = int(len(train_loader) * CFG.epochs)
    num_warmup_steps = int(num_train_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # AWP
    awp = None
    if CFG.use_awp:
        awp = AWP(
            model,
            optimizer,
            adv_lr=CFG.awp_lr,
            adv_eps=CFG.awp_eps,
            start_epoch=CFG.awp_start_epoch,
        )

    # Training Loop
    best_loss = np.inf
    best_model_path = os.path.join(CFG.output_dir, f"nn_model_fold_{fold}.bin")

    for epoch in range(CFG.epochs):
        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, scheduler, device, awp
        )
        val_loss, val_preds, val_labels = valid_one_epoch(model, val_loader, device)

        logger.info(
            f"Fold {fold} | Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.15f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(
                f"New best model saved for Fold {fold} with loss {best_loss:.6f}"
            )

    # Load Best Model for Inference
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)

    # Generate OOF Predictions
    _, val_preds, val_labels = valid_one_epoch(model, val_loader, device)

    # Generate Test Predictions
    test_preds = inference_fn(model, test_loader, device)

    # Save to Cache
    np.save(cache_oof_path, val_preds)
    np.save(cache_test_path, test_preds)
    np.save(cache_labels_path, val_labels)

    # Cleanup
    del model, optimizer, scheduler, awp
    torch.cuda.empty_cache()
    gc.collect()

    return val_preds, test_preds, val_labels


def run_classic_branches(load_cached_data=True):
    """
    Runs the Feature Engineering and Ridge Regression branches.
    Returns aligned OOF and Test predictions for Lexical, Morphological, and Structural features.
    """
    cache_file = os.path.join(CFG.output_dir, "classic_features_oof_test.npz")

    if load_cached_data and os.path.exists(cache_file):
        logger.info("Loading cached classic features...")
        data = np.load(cache_file)
        return (
            data["lexical_oof"],
            data["morph_oof"],
            data["struct_oof"],
            data["lexical_test"],
            data["morph_test"],
            data["struct_test"],
            data["targets"],
        )

    logger.info("Computing classic features and models...")

    # 1. Load Data
    df_train = pd.read_csv(CFG.train_path)
    df_val = pd.read_csv(CFG.val_path)
    df_full = pd.concat([df_train, df_val], ignore_index=True)
    df_test = pd.read_csv(CFG.test_path)

    # 2. Structural Features
    fe = FeatureEngineer()
    # Note: FE handles internal caching based on split name
    feat_train_df = fe.extract_features(
        df_full, split_name="train_val_merged", load_cached_data=load_cached_data
    )
    feat_test_df = fe.extract_features(
        df_test, split_name="test", load_cached_data=load_cached_data
    )

    # Convert to numpy
    # We select specific numeric columns to be safe
    feature_cols = [
        c for c in feat_train_df.columns if c not in ["essay_id", "full_text", "score"]
    ]
    X_struct = feat_train_df[feature_cols].values.astype(np.float32)
    X_struct_test = feat_test_df[feature_cols].values.astype(np.float32)

    # 3. Ridge Models (Lexical & Morphological)
    targets = df_full["score"].values

    # Initialize OOF and Test arrays
    lexical_oof = np.zeros(len(df_full))
    morph_oof = np.zeros(len(df_full))

    lexical_test_preds = []
    morph_test_preds = []

    # Stratified K-Fold (Must match NN split)
    skf = StratifiedKFold(n_splits=CFG.num_folds, shuffle=True, random_state=CFG.seed)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_full, targets.astype(int))
    ):
        logger.info(f"Processing Classic Branch Fold {fold}...")

        # Get Data Splits
        X_train_text = df_full.loc[train_idx, "full_text"]
        y_train = targets[train_idx]
        X_val_text = df_full.loc[val_idx, "full_text"]
        X_test_text = df_test["full_text"]

        # --- Lexical (Word N-grams) ---
        word_pipe = get_ridge_pipeline(analyzer="word")
        word_pipe.fit(X_train_text, y_train)
        lexical_oof[val_idx] = word_pipe.predict(X_val_text)
        lexical_test_preds.append(word_pipe.predict(X_test_text))

        # --- Morphological (Char N-grams) ---
        char_pipe = get_ridge_pipeline(analyzer="char")
        char_pipe.fit(X_train_text, y_train)
        morph_oof[val_idx] = char_pipe.predict(X_val_text)
        morph_test_preds.append(char_pipe.predict(X_test_text))

    # Average Test Predictions
    lexical_test = np.mean(lexical_test_preds, axis=0)
    morph_test = np.mean(morph_test_preds, axis=0)

    # Save to Cache
    np.savez(
        cache_file,
        lexical_oof=lexical_oof,
        morph_oof=morph_oof,
        struct_oof=X_struct,
        lexical_test=lexical_test,
        morph_test=morph_test,
        struct_test=X_struct_test,
        targets=targets,
    )

    return (
        lexical_oof,
        morph_oof,
        X_struct,
        lexical_test,
        morph_test,
        X_struct_test,
        targets,
    )


def run_stacking(
    nn_oof,
    nn_test,
    lex_oof,
    morph_oof,
    struct_oof,
    lex_test,
    morph_test,
    struct_test,
    targets,
):
    """
    Trains the Meta-Learner (LightGBM) and optimizes thresholds.
    """
    logger.info("Running Stacking (Meta-Learner)...")

    # Construct Meta-Features
    # Shape: (N_samples, 3 + N_struct)
    X_meta = np.column_stack([nn_oof, lex_oof, morph_oof, struct_oof])
    X_test_meta = np.column_stack([nn_test, lex_test, morph_test, struct_test])

    # Train LightGBM
    # We train on all OOF data. For a more rigorous approach, we could do nested CV,
    # but training on full OOF is standard for the final submission model.
    model = get_lgbm_stacker()
    model.fit(X_meta, targets)

    # Predict
    final_oof_preds = model.predict(X_meta)
    final_test_preds = model.predict(X_test_meta)

    # Evaluate Stacking Performance (RMSE)
    rmse = np.sqrt(mean_squared_error(targets, final_oof_preds))
    logger.info(f"Stacking OOF RMSE: {rmse:.6f}")

    # Optimize Thresholds
    logger.info("Optimizing thresholds...")
    rounder = OptimizedRounder()
    rounder.fit(final_oof_preds, targets)

    optimized_coeffs = rounder.coefficients()
    logger.info(f"Optimized Coefficients: {optimized_coeffs}")

    # Calculate Final CV Score (QWK)
    final_oof_rounded = rounder.predict(final_oof_preds, optimized_coeffs)
    qwk = get_score(targets, final_oof_rounded)
    logger.info(f"Final CV QWK Score: {qwk:.6f}")

    return final_test_preds, rounder


def generate_submission(test_preds, rounder, essay_ids):
    """
    Applies thresholds and saves submission file.
    """
    final_scores = rounder.predict(test_preds)

    submission = pd.DataFrame(
        {"essay_id": essay_ids, "score": final_scores.astype(int)}
    )

    submission.to_csv(CFG.submission_file, index=False)
    logger.info(f"Submission saved to {CFG.submission_file}")
    logger.info(f"Submission head:\n{submission.head()}")


def main(load_cached_data=True):
    seed_everything(CFG.seed)

    # 1. Run Classic Branches
    # We run this first to ensure features are ready
    lex_oof, morph_oof, struct_oof, lex_test, morph_test, struct_test, targets = (
        run_classic_branches(load_cached_data)
    )

    # 2. Run Neural Network Branch (5 Folds)
    # We need to aggregate NN OOFs aligned with the targets
    nn_oof_full = np.zeros_like(targets, dtype=float)
    nn_test_preds_folds = []

    # Get Test Loader once
    test_loader, essay_ids = get_test_loader(load_cached_data)

    # We need to know the validation indices for each fold to place OOF preds correctly
    # Re-instantiate SKF to get indices
    skf = StratifiedKFold(n_splits=CFG.num_folds, shuffle=True, random_state=CFG.seed)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(targets)), targets.astype(int))
    ):
        # Get Loaders for this fold
        train_loader, val_loader = get_loaders(fold, load_cached_data)

        # Run Fold
        val_preds, test_preds, _ = run_nn_fold(
            fold, train_loader, val_loader, test_loader, CFG.device, load_cached_data
        )

        # Assign OOF
        # Ensure length matches
        if len(val_preds) != len(val_idx):
            raise ValueError(
                f"Fold {fold} prediction length mismatch: {len(val_preds)} vs {len(val_idx)}"
            )

        nn_oof_full[val_idx] = val_preds
        nn_test_preds_folds.append(test_preds)

    # Average NN Test Preds
    nn_test_avg = np.mean(nn_test_preds_folds, axis=0)

    # 3. Stacking & Thresholding
    final_test_preds, rounder = run_stacking(
        nn_oof_full,
        nn_test_avg,
        lex_oof,
        morph_oof,
        struct_oof,
        lex_test,
        morph_test,
        struct_test,
        targets,
    )

    # 4. Submission
    generate_submission(final_test_preds, rounder, essay_ids)
