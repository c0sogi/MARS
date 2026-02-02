import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from library.config import Config
from library.utils import get_logger, compute_qwk, seed_everything
from library.data import EssayDataset
from library.model import EssayModel

# Initialize logger
logger = get_logger(os.path.join(Config.output_dir, "stacking.log"))


def load_oof_data(config):
    """
    Loads OOF embeddings, targets, and IDs from the cache.
    """
    oof_ids_list = []
    oof_embeddings_list = []
    oof_targets_list = []

    for fold in range(config.n_folds):
        id_path = os.path.join(config.cache_dir, f"oof_ids_fold_{fold}.npy")
        emb_path = os.path.join(config.cache_dir, f"oof_embeddings_fold_{fold}.npy")
        target_path = os.path.join(config.cache_dir, f"oof_targets_fold_{fold}.npy")

        if not (
            os.path.exists(id_path)
            and os.path.exists(emb_path)
            and os.path.exists(target_path)
        ):
            raise FileNotFoundError(
                f"OOF data for fold {fold} not found in {config.cache_dir}. Run trainer first."
            )

        oof_ids_list.append(np.load(id_path, allow_pickle=True))
        oof_embeddings_list.append(np.load(emb_path))
        oof_targets_list.append(np.load(target_path))

    oof_ids = np.concatenate(oof_ids_list)
    oof_embeddings = np.concatenate(oof_embeddings_list, axis=0)
    oof_targets = np.concatenate(oof_targets_list)

    return oof_ids, oof_embeddings, oof_targets


def train_stacking(train_df, config):
    """
    Trains the LightGBM stacking head using 5-fold CV on the OOF embeddings.
    """
    logger.info("Loading OOF data for stacking...")
    oof_ids, oof_embeddings, oof_targets = load_oof_data(config)

    # Create OOF DataFrame
    # We create columns for embeddings: emb_0, emb_1, ...
    emb_cols = [f"emb_{i}" for i in range(oof_embeddings.shape[1])]

    # Construct a DataFrame for easy merging
    df_oof = pd.DataFrame(oof_embeddings, columns=emb_cols)
    df_oof["essay_id"] = oof_ids

    # Merge with train_df to align meta-features and fold info
    # train_df contains: essay_id, score, fold, word_count, char_count, etc.
    logger.info("Merging OOF data with meta-features...")

    # Select meta-features
    meta_cols = ["word_count", "char_count", "sentence_count", "unique_word_ratio"]

    # Ensure we don't duplicate columns if they exist in both
    cols_to_use = ["essay_id", "fold", "score"] + meta_cols
    df_merged = train_df[cols_to_use].merge(df_oof, on="essay_id", how="inner")

    if len(df_merged) != len(train_df):
        logger.warning(
            f"Mismatch in merged data length. Train: {len(train_df)}, Merged: {len(df_merged)}"
        )

    # Features for LightGBM
    feature_cols = emb_cols + meta_cols

    scores = []

    logger.info("Starting LightGBM Training (5-Fold CV)...")

    for fold in range(config.n_folds):
        logger.info(f"--- Stacking Fold {fold} ---")

        # Split Data
        train_idx = df_merged["fold"] != fold
        val_idx = df_merged["fold"] == fold

        X_train = df_merged.loc[train_idx, feature_cols]
        y_train = df_merged.loc[train_idx, "score"]

        X_val = df_merged.loc[val_idx, feature_cols]
        y_val = df_merged.loc[val_idx, "score"]

        # LightGBM Dataset
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # Callbacks
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=config.lgbm_params["early_stopping_rounds"]
            ),
            lgb.log_evaluation(period=100),
        ]

        # Train
        model = lgb.train(
            config.lgbm_params,
            dtrain,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Evaluate
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        score = compute_qwk(y_val, preds)
        scores.append(score)

        logger.info(f"Stacking Fold {fold} QWK: {score}")

        # Save Model
        model_path = os.path.join(config.model_dir, f"lgbm_fold_{fold}.txt")
        model.save_model(model_path)

    avg_score = np.mean(scores)
    logger.info(f"Average Stacking QWK: {avg_score}")

    return avg_score


def get_test_embeddings(test_df, config, load_cached_data=True):
    """
    Generates or loads embeddings for the test set using the 5 backbone models.
    """
    cache_path = os.path.join(config.cache_dir, "test_embeddings.npy")

    if load_cached_data and os.path.exists(cache_path):
        logger.info("Loading test embeddings from cache...")
        return np.load(cache_path)

    logger.info("Generating test embeddings (Ensemble of 5 Backbones)...")

    device = config.device
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    test_dataset = EssayDataset(test_df, tokenizer, config, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Array to store sum of embeddings
    aggregated_embeddings = None

    for fold in range(config.n_folds):
        logger.info(f"Inference with Backbone Fold {fold}...")

        model_path = os.path.join(config.model_dir, f"backbone_fold_{fold}.pth")
        if not os.path.exists(model_path):
            logger.warning(
                f"Model for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        model = EssayModel(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        fold_embeddings = []

        with torch.no_grad():
            for data in test_loader:
                input_ids = data["input_ids"].to(device)
                attention_mask = data["attention_mask"].to(device)
                meta_features = data["meta_features"].to(device)

                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=(
                        torch.bfloat16 if config.use_mixed_precision else torch.float32
                    ),
                ):
                    outputs = model(input_ids, attention_mask, meta_features)

                fold_embeddings.append(
                    outputs["embeddings"].detach().float().cpu().numpy()
                )

        fold_embeddings = np.concatenate(fold_embeddings, axis=0)

        if aggregated_embeddings is None:
            aggregated_embeddings = np.zeros_like(fold_embeddings)

        aggregated_embeddings += fold_embeddings

        # Cleanup
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Average
    avg_embeddings = aggregated_embeddings / config.n_folds

    # Cache
    np.save(cache_path, avg_embeddings)

    return avg_embeddings


def inference(test_df, config):
    """
    Performs full inference pipeline:
    1. Generate/Load Test Embeddings (Backbone Ensemble)
    2. Predict with LightGBM (Stacking Ensemble)
    3. Save Submission
    """
    logger.info("Starting Inference...")

    # 1. Get Embeddings
    test_embeddings = get_test_embeddings(test_df, config)

    # 2. Prepare Features
    emb_cols = [f"emb_{i}" for i in range(test_embeddings.shape[1])]
    meta_cols = ["word_count", "char_count", "sentence_count", "unique_word_ratio"]

    # Create DataFrame for features
    df_emb = pd.DataFrame(test_embeddings, columns=emb_cols)

    # Reset index of test_df to ensure alignment
    test_df = test_df.reset_index(drop=True)

    X_test = pd.concat([df_emb, test_df[meta_cols]], axis=1)

    # 3. Predict with LightGBM Ensemble
    final_preds = np.zeros(len(test_df))

    for fold in range(config.n_folds):
        model_path = os.path.join(config.model_dir, f"lgbm_fold_{fold}.txt")
        if not os.path.exists(model_path):
            logger.warning(f"LGBM model for fold {fold} not found. Skipping.")
            continue

        model = lgb.Booster(model_file=model_path)
        preds = model.predict(X_test)
        final_preds += preds

    final_preds /= config.n_folds

    # 4. Post-processing
    # Clip to [1, 6] and round
    final_preds = np.clip(final_preds, 1, 6)
    final_preds = np.round(final_preds).astype(int)

    # 5. Create Submission
    submission = pd.DataFrame({"essay_id": test_df["essay_id"], "score": final_preds})

    submission.to_csv(config.submission_path, index=False)
    logger.info(f"Submission saved to {config.submission_path}")
    logger.info(f"Submission head:\n{submission.head()}")

    return submission
