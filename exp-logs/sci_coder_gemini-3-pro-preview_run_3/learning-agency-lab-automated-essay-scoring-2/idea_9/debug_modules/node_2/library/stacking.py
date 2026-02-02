import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import get_logger, compute_qwk, get_cache_path
from library.data import load_data_from_metadata

# Initialize logger
logger = get_logger("stacking")


class FeatureEngineer:
    """
    Handles the extraction and caching of scalar meta-features from essay texts.
    Features include: char_count, word_count, sentence_count, unique_word_count, etc.
    """

    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _extract(self, df):
        """
        Internal method to compute features from a dataframe.
        """
        df = df.copy()
        # Ensure full_text is string
        texts = df["full_text"].astype(str).fillna("")

        # 1. Length features
        df["char_count"] = texts.apply(len)
        df["word_count"] = texts.apply(lambda x: len(x.split()))

        # 2. Sentence approximation (counting punctuation)
        df["sentence_count"] = texts.apply(
            lambda x: x.count(".") + x.count("?") + x.count("!")
        )

        # 3. Vocabulary richness
        def get_unique_count(text):
            words = text.split()
            if len(words) == 0:
                return 0
            return len(set(words))

        df["unique_word_count"] = texts.apply(get_unique_count)

        # 4. Ratios
        df["word_len_avg"] = df["char_count"] / (df["word_count"] + 1e-6)
        df["unique_ratio"] = df["unique_word_count"] / (df["word_count"] + 1e-6)

        # Select only feature columns
        feature_cols = [
            "char_count",
            "word_count",
            "sentence_count",
            "unique_word_count",
            "word_len_avg",
            "unique_ratio",
        ]
        return df[feature_cols]

    def get_features(self, split="train", load_cached_data=True):
        """
        Retrieves meta-features for a specific split.
        Implements caching logic using parquet files.

        Args:
            split (str): 'train' or 'test' (or 'val' mapped to train logic if needed).
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: Dataframe containing meta-features.
        """
        # Define cache path
        # We use a simple config dict to hash, ensuring version control if logic changes
        config_hash_obj = {
            "feature_version": "v1",
            "split": split,
            "features": ["char", "word", "sent", "unique", "ratios"],
        }
        cache_path = get_cache_path(f"{split}_meta_features", config_hash_obj).replace(
            ".npy", ".parquet"
        )

        # 1. Try Load
        if load_cached_data and os.path.exists(cache_path):
            logger.info(f"Loading cached meta-features for '{split}' from {cache_path}")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Re-computing.")

        # 2. Compute
        logger.info(f"Computing meta-features for '{split}'...")
        # Map 'val' to 'train' metadata if necessary, but usually we treat train/val as one for stacking
        # or separate. Here we load based on the split name provided.
        # Note: 'train' split in metadata is the 80% split. 'val' is 20%.
        # If we want features for the full training set (train+val), we might need to concat.
        # However, the standard pipeline usually provides OOFs aligned with the specific folds.
        # For simplicity in this module, we load the specific metadata file requested.

        df = load_data_from_metadata(split)
        features_df = self._extract(df)

        # 3. Save
        logger.info(f"Saving meta-features to {cache_path}")
        features_df.to_parquet(cache_path, index=False)

        return features_df


class LGBMStacker:
    """
    Implements the LightGBM Stacking Meta-Learner.
    Trains on OOF predictions + Meta-features.
    Predicts on Test predictions + Meta-features.
    """

    def __init__(self):
        self.params = Config.LGBM_PARAMS.copy()
        self.models = []  # To store trained fold models
        self.feature_engineer = FeatureEngineer()

    def make_dataset(self, oof_dict, split="train", load_cached_data=True):
        """
        Constructs the input dataset (X) for the stacker.
        Concatenates model predictions with meta-features.

        Args:
            oof_dict (dict): Dictionary {model_name: np.array of predictions}.
                             Arrays must be aligned with the metadata for 'split'.
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Cache flag for feature engineering.

        Returns:
            pd.DataFrame: The feature matrix X.
        """
        # Get meta-features
        meta_df = self.feature_engineer.get_features(split, load_cached_data)

        # Create DataFrame from OOF dictionary
        # We assume oof_dict values are 1D arrays or (N,1) arrays
        data = {}
        for model_name, preds in oof_dict.items():
            data[f"pred_{model_name}"] = preds.flatten()

        oof_df = pd.DataFrame(data)

        # Concatenate
        # Reset indices to ensure alignment
        meta_df = meta_df.reset_index(drop=True)
        oof_df = oof_df.reset_index(drop=True)

        X = pd.concat([oof_df, meta_df], axis=1)
        return X

    def train(self, train_oof_dict, train_targets, n_folds=5):
        """
        Trains the LightGBM model using Stratified K-Fold CV.

        Args:
            train_oof_dict (dict): Dictionary of OOF predictions for the training set.
            train_targets (array-like): True target scores.
            n_folds (int): Number of folds for CV.

        Returns:
            float: The average QWK score across folds.
        """
        logger.info("Preparing training data for Stacker...")

        # We need to combine train and val metadata to reconstruct the full training set
        # if the OOFs provided cover the whole dataset (which is typical for stacking).
        # However, based on the provided library, we have separate train/val metadata.
        # We assume `train_oof_dict` corresponds to the concatenated (Train + Val) or
        # just the set we want to train on.
        # For this implementation, we will assume the caller provides OOFs aligned with
        # a specific set of features. To be safe, we will load 'train' features.
        # NOTE: In a robust pipeline, we should handle the alignment carefully.
        # Here, we assume `train_oof_dict` aligns with `load_data_from_metadata('train')`
        # OR the user passes the correct split name.
        # Let's assume we are training on the 'train' split defined in metadata.

        X = self.make_dataset(train_oof_dict, split="train", load_cached_data=True)
        y = np.array(train_targets, dtype=int)

        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

        oof_preds = np.zeros(len(X))
        scores = []
        self.models = []

        logger.info(f"Starting Stacker Training (LGBM) with {n_folds} folds...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create LGBM Datasets
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            # Train
            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),  # Disable verbose logging
            ]

            model = lgb.train(
                self.params,
                dtrain,
                num_boost_round=2000,
                valid_sets=[dtrain, dval],
                callbacks=callbacks,
            )

            self.models.append(model)

            # Predict
            val_pred = model.predict(X_val, num_iteration=model.best_iteration)
            oof_preds[val_idx] = val_pred

            # Score
            qwk = compute_qwk(y_val, val_pred)
            scores.append(qwk)
            logger.info(f"Stacker Fold {fold+1} QWK: {qwk}")

        # Overall Score
        overall_qwk = compute_qwk(y, oof_preds)
        logger.info(f"Stacker Mean QWK: {np.mean(scores)}")
        logger.info(f"Stacker OOF QWK: {overall_qwk}")

        return overall_qwk

    def predict(self, test_pred_dict):
        """
        Generates predictions for the test set using the trained fold models.
        Averages the predictions from all fold models.

        Args:
            test_pred_dict (dict): Dictionary of model predictions for the test set.

        Returns:
            np.array: Final predicted scores.
        """
        logger.info("Generating test predictions...")
        X_test = self.make_dataset(test_pred_dict, split="test", load_cached_data=True)

        final_preds = np.zeros(len(X_test))

        for model in self.models:
            preds = model.predict(X_test, num_iteration=model.best_iteration)
            final_preds += preds

        final_preds /= len(self.models)

        return final_preds

    def run_inference_and_submit(self, test_pred_dict):
        """
        Full inference pipeline:
        1. Predicts on test data.
        2. Post-processes predictions (clip, round).
        3. Saves to submission.csv.
        """
        raw_preds = self.predict(test_pred_dict)

        # Post-processing: Clip to [1, 6] and round
        final_preds = np.clip(np.round(raw_preds), 1, 6).astype(int)

        # Load test metadata to get essay_ids
        df_test = load_data_from_metadata("test")

        submission = pd.DataFrame(
            {"essay_id": df_test["essay_id"], "score": final_preds}
        )

        # Save
        save_path = Config.SUBMISSION_FILE
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        submission.to_csv(save_path, index=False)

        logger.info(f"Submission saved to {save_path}. Shape: {submission.shape}")
        logger.info(f"Head:\n{submission.head()}")
