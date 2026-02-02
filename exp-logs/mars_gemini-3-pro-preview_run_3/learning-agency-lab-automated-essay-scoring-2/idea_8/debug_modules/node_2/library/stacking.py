import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from library.config import Config
from library.utils import get_logger, compute_qwk, seed_everything
from library.data import get_meta_features, get_dataloaders
from library.model import EssayModel

# Initialize logger
logger = get_logger(os.path.join(Config.WORKING_DIR, "output", "stacking.log"))


def load_oof_data():
    """
    Loads OOF embeddings from cache and aligns them with meta-features.
    Assumes OOF embeddings correspond to the fixed validation set defined in Config.
    """
    logger.info("Loading OOF Data for Stacking...")

    # 1. Load Validation Metadata for Targets and Meta-Features
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    if Config.DEBUG:
        val_df = val_df.iloc[:50]

    # Calculate Meta-Features
    # We re-calculate to ensure alignment and independence from data loader hashing
    meta_features = get_meta_features(val_df["full_text"])
    targets = val_df["score"].values.astype(float)

    # 2. Load and Average OOF Embeddings
    oof_embeddings_list = []
    found_folds = 0

    for fold in range(Config.N_FOLDS):
        # We look for embeddings saved by trainer.py
        # Note: trainer.py saves them as 'oof_embeddings_fold_{fold}.npy'
        path = os.path.join(Config.CACHE_DIR, f"oof_embeddings_fold_{fold}.npy")

        if os.path.exists(path):
            embeds = np.load(path)

            # Debug check: ensure length matches
            if len(embeds) != len(val_df):
                logger.warning(
                    f"Shape mismatch for fold {fold}: "
                    f"Embeddings {embeds.shape[0]} vs Metadata {len(val_df)}. "
                    "This is expected only in DEBUG mode if cache is stale."
                )
                if not Config.DEBUG:
                    raise ValueError(f"OOF embedding size mismatch for fold {fold}")
                else:
                    # Truncate for debug
                    embeds = embeds[: len(val_df)]

            oof_embeddings_list.append(embeds)
            found_folds += 1
        else:
            logger.warning(f"OOF embeddings for fold {fold} not found at {path}")

    if found_folds == 0:
        raise FileNotFoundError("No OOF embeddings found. Train backbones first.")

    # Average embeddings across available folds
    avg_embeddings = np.mean(oof_embeddings_list, axis=0)

    logger.info(f"Loaded OOF data. Embeddings shape: {avg_embeddings.shape}")

    return avg_embeddings, meta_features, targets


def get_test_embeddings(load_cached_data=True):
    """
    Generates or loads test set embeddings using the ensemble of backbone models.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "test_ensemble_embeddings.npy")

    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached test embeddings from {cache_path}")
        return np.load(cache_path)

    logger.info("Generating Test Embeddings (Inference)...")

    # Get Test Loader
    # We use get_dataloaders to ensure consistent preprocessing
    # We only need the test_loader
    _, _, test_loader = get_dataloaders(
        valid_batch_size=Config.VALID_BATCH_SIZE, load_cached_data=load_cached_data
    )

    device = Config.DEVICE
    fold_embeddings = []

    # Iterate through all folds
    for fold in range(Config.N_FOLDS):
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"backbone_fold_{fold}.pth")
        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint {ckpt_path} not found. Skipping fold {fold}.")
            continue

        logger.info(f"Running inference with backbone fold {fold}...")

        # Load Model
        model = EssayModel()
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        all_embeds = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_ids = batch["batch_ids"].to(device)

                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids, attention_mask, batch_ids)
                    embeddings = outputs["embeddings"]

                all_embeds.append(embeddings.float().cpu().numpy())

        # Concatenate batches for this fold
        fold_embeddings.append(np.concatenate(all_embeds, axis=0))

        # Cleanup
        del model, state_dict
        torch.cuda.empty_cache()
        gc.collect()

    if not fold_embeddings:
        raise RuntimeError("No models available for inference.")

    # Average across folds
    avg_embeddings = np.mean(fold_embeddings, axis=0)

    # Cache result
    np.save(cache_path, avg_embeddings)
    logger.info(f"Test embeddings generated and saved. Shape: {avg_embeddings.shape}")

    return avg_embeddings


def train_stacking(load_cached_data=True):
    """
    Trains the LightGBM stacking head.
    """
    seed_everything(Config.SEED)

    # 1. Prepare Data
    embeddings, meta_features, targets = load_oof_data()

    # Concatenate features: [Embeddings, MetaFeatures]
    X = np.hstack([embeddings, meta_features])
    y = targets

    logger.info(f"Stacking Training Data Shape: {X.shape}")

    # 2. Split for Early Stopping
    # We split the OOF data into Train/Val for the LightGBM model
    # This ensures we don't overfit the head to the OOF predictions
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=Config.SEED
    )

    # 3. Train LightGBM
    logger.info("Training LightGBM Stacking Model...")

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=Config.LGBM_PARAMS["early_stopping_rounds"]),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        Config.LGBM_PARAMS,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    # 4. Evaluate
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    qwk = compute_qwk(y_val, val_preds)

    logger.info(f"LightGBM Validation RMSE: {rmse:.6f}")
    logger.info(f"LightGBM Validation QWK: {qwk:.6f}")

    # 5. Save Model
    model_path = os.path.join(Config.WORKING_DIR, "lgbm_stacking.txt")
    model.save_model(model_path)
    logger.info(f"Stacking model saved to {model_path}")

    return model


def predict_stacking(model=None, load_cached_data=True):
    """
    Generates predictions for the test set using the stacking pipeline.
    """
    logger.info("Starting Stacking Inference...")

    # 1. Load Model if not provided
    if model is None:
        model_path = os.path.join(Config.WORKING_DIR, "lgbm_stacking.txt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LightGBM model not found at {model_path}")
        model = lgb.Booster(model_file=model_path)

    # 2. Prepare Test Data
    # Load Metadata
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    if Config.DEBUG:
        test_df = test_df.iloc[:20]

    # Meta Features
    meta_features = get_meta_features(test_df["full_text"])

    # Embeddings
    embeddings = get_test_embeddings(load_cached_data=load_cached_data)

    # Ensure alignment (crucial for debug mode)
    if len(embeddings) != len(meta_features):
        logger.warning(
            "Shape mismatch in test inference (likely due to DEBUG). Truncating."
        )
        min_len = min(len(embeddings), len(meta_features))
        embeddings = embeddings[:min_len]
        meta_features = meta_features[:min_len]
        test_df = test_df.iloc[:min_len]

    # Concatenate
    X_test = np.hstack([embeddings, meta_features])

    # 3. Predict
    logger.info("Predicting scores...")
    raw_preds = model.predict(X_test, num_iteration=model.best_iteration)

    # 4. Post-processing
    # Clip to [1, 6] and round
    final_scores = np.clip(raw_preds, 1, 6).round().astype(int)

    # 5. Save Submission
    submission = test_df[["essay_id"]].copy()
    submission["score"] = final_scores

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    logger.info(f"First 5 predictions:\n{submission.head()}")
