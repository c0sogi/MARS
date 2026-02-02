import os
import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Import LightGBM only if needed to avoid strict dependency issues if not installed
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

from library.configuration import Config, seed_everything
from library.utilities import get_logger, compute_metrics
from library.feature_engineering import FeatureEngineer
from library.dataset import load_supervised_data, get_tokenizer, Collate, EssayDataset
from library.modeling import EssayModel

logger = get_logger("MetaModeling")


class MetaLearner:
    """
    Level 2 Meta-Learner.
    Combines Level 1 predictions with explicit meta-features to correct bias.
    """

    def __init__(self, model_type="ridge"):
        self.model_type = model_type
        self.model = self._get_model()
        self.feature_cols = [
            "char_count",
            "word_count",
            "sentence_count",
            "unique_word_count",
            "avg_word_len",
        ]

    def _get_model(self):
        if self.model_type == "ridge":
            # Ridge Regression with Standard Scaler
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=1.0, random_state=Config.SEED)),
                ]
            )
        elif self.model_type == "lgbm":
            if LGBMRegressor is None:
                raise ImportError("LightGBM is not installed.")
            return LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                random_state=Config.SEED,
                verbose=-1,
                metric="rmse",
            )
        else:
            raise ValueError(f"Unknown meta-model type: {self.model_type}")

    def prepare_features(
        self, l1_preds: np.ndarray, meta_features_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Concatenates Level 1 predictions with meta-features.
        """
        # Ensure meta_features_df aligns with l1_preds
        meta_feats = meta_features_df[self.feature_cols].values

        # Reshape l1_preds to (N, 1)
        if len(l1_preds.shape) == 1:
            l1_preds = l1_preds.reshape(-1, 1)

        # Concatenate: [L1_Score, Meta_Feat_1, Meta_Feat_2, ...]
        return np.hstack([l1_preds, meta_feats])

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


def _predict_fold(model, dataloader, device):
    """
    Helper to run inference on a single fold model.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Autocast for mixed precision inference (faster)
            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)

            preds.append(logits.view(-1).cpu().numpy())

    return np.concatenate(preds)


def generate_level1_test_predictions(load_cached_data=True, debug=False):
    """
    Generates Level 1 predictions for the Test set by averaging outputs from all 5 folds.
    Implements caching to avoid re-running inference.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "test_L1_preds.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached Level 1 Test predictions from {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info("Generating Level 1 Test predictions (Inference on 5 Folds)...")

    # 2. Load Test Data
    tokenizer = get_tokenizer()
    test_dataset = load_supervised_data(
        "test", tokenizer, load_cached_data=load_cached_data, debug=debug
    )
    test_df = test_dataset.df

    collate_fn = Collate(tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 3. Inference Loop
    fold_preds = []

    for fold in range(Config.NUM_FOLDS):
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"model_fold_{fold}.pth")

        if not os.path.exists(checkpoint_path):
            logger.warning(
                f"Checkpoint for fold {fold} not found at {checkpoint_path}. Skipping."
            )
            continue

        logger.info(f"Inference with Fold {fold+1}/{Config.NUM_FOLDS}...")

        # Load Model
        model = EssayModel(
            pretrained=False
        )  # No need to load pretrained weights, we load state_dict
        state_dict = torch.load(checkpoint_path, map_location=Config.DEVICE)
        model.load_state_dict(state_dict)
        model.to(Config.DEVICE)

        # Predict
        preds = _predict_fold(model, test_loader, Config.DEVICE)
        fold_preds.append(preds)

        # Cleanup
        del model, state_dict
        torch.cuda.empty_cache()
        gc.collect()

    if not fold_preds:
        raise RuntimeError("No models found for inference.")

    # 4. Average Predictions
    avg_preds = np.mean(fold_preds, axis=0)

    # 5. Create DataFrame and Cache
    result_df = test_df[["essay_id", "full_text"]].copy()
    result_df["pred_score"] = avg_preds

    try:
        result_df.to_parquet(cache_path, index=False)
        logger.info(f"Saved Level 1 Test predictions to {cache_path}")
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

    return result_df


def run_stacking(debug=False, load_cached_data=True):
    """
    Main function to execute Stage 3: Stacking.
    """
    seed_everything(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Load OOF Predictions (Training Data for Meta-Learner)
    oof_path = os.path.join(Config.WORKING_DIR, "oof_predictions.csv")
    if not os.path.exists(oof_path):
        raise FileNotFoundError(
            f"OOF predictions not found at {oof_path}. Run training first."
        )

    logger.info(f"Loading OOF predictions from {oof_path}")
    oof_df = pd.read_csv(oof_path)

    if debug:
        oof_df = oof_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 2. Load/Generate Level 1 Test Predictions (Test Data for Meta-Learner)
    test_pred_df = generate_level1_test_predictions(
        load_cached_data=load_cached_data, debug=debug
    )

    # 3. Feature Engineering (Meta-Features)
    fe = FeatureEngineer()

    # Extract features for OOF (Train)
    # We use a unique partition name 'oof_meta' to cache these specific features
    logger.info("Extracting meta-features for OOF data...")
    oof_feats = fe.process_and_cache(
        oof_df, "oof_meta", load_cached_data=load_cached_data
    )

    # Extract features for Test
    logger.info("Extracting meta-features for Test data...")
    test_feats = fe.process_and_cache(
        test_pred_df, "test_meta", load_cached_data=load_cached_data
    )

    # 4. Prepare Data for Meta-Learner
    meta_learner = MetaLearner(model_type=Config.META_MODEL_TYPE)

    # Train Inputs
    X_train = meta_learner.prepare_features(oof_df["pred_score"].values, oof_feats)
    y_train = oof_df["score"].values

    # Test Inputs
    X_test = meta_learner.prepare_features(
        test_pred_df["pred_score"].values, test_feats
    )

    logger.info(f"Meta-Learner Training Data Shape: {X_train.shape}")
    logger.info(f"Meta-Learner Test Data Shape: {X_test.shape}")

    # 5. Train Meta-Learner
    logger.info(f"Training Meta-Learner ({Config.META_MODEL_TYPE})...")
    meta_learner.fit(X_train, y_train)

    # Evaluate on Train (Sanity Check)
    train_preds = meta_learner.predict(X_train)
    mse = mean_squared_error(y_train, train_preds)
    train_metrics = compute_metrics(y_train, train_preds)
    logger.info(f"Meta-Learner Train MSE: {mse}")
    logger.info(f"Meta-Learner Train QWK: {train_metrics['qwk']}")

    # 6. Predict on Test
    logger.info("Predicting on Test set...")
    raw_test_preds = meta_learner.predict(X_test)

    # 7. Post-Processing
    # Clip to [1, 6] and round to nearest integer
    final_preds = np.clip(raw_test_preds, 1, 6).round().astype(int)

    # 8. Create Submission
    submission_df = pd.DataFrame(
        {"essay_id": test_pred_df["essay_id"], "score": final_preds}
    )

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)

    logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")
    logger.info(f"First 5 predictions:\n{submission_df.head()}")

    return submission_df
