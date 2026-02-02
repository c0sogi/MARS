import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from library.config import Config
from library.utils import seed_everything, get_config_hash
from library.feature_extraction import FeatureExtractor
from library.dimensionality_reduction import IndependentPCA
from library.models import get_base_models, get_meta_learner


def merge_data_dicts(dict1, dict2):
    """
    Merges two feature dictionaries (e.g., train and val) by concatenating
    their numpy arrays along the first axis.
    """
    merged = {}
    # Assuming both dicts have the same keys
    for key in dict1.keys():
        val1 = dict1.get(key)
        val2 = dict2.get(key)

        if val1 is None or val2 is None:
            merged[key] = None
        elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            merged[key] = np.concatenate([val1, val2], axis=0)
        else:
            # Fallback for non-array items, though not expected for features/targets
            merged[key] = val1
    return merged


def slice_data_dict(data_dict, indices):
    """
    Slices all numpy arrays in the data dictionary using the provided indices.
    Used for creating fold-specific datasets.
    """
    sliced = {}
    for key, value in data_dict.items():
        if value is None:
            sliced[key] = None
        elif isinstance(value, np.ndarray):
            sliced[key] = value[indices]
        else:
            sliced[key] = value
    return sliced


class CrossValidator:
    """
    Manages the K-Fold Cross-Validation process for Stacking.
    Ensures PCA and Models are fitted strictly on fold training data to avoid leakage.
    """

    def __init__(self, n_folds=Config.N_FOLDS, seed=Config.SEED):
        self.n_folds = n_folds
        self.seed = seed
        self.working_dir = Config.WORKING_DIR

    def run_cv(self, data_dict, load_cached_data=True):
        """
        Runs K-Fold CV to generate Out-of-Fold (OOF) predictions.

        Args:
            data_dict: Dictionary containing concatenated Train + Val data.
            load_cached_data: If True, attempts to load OOF preds from disk.

        Returns:
            oof_preds_df: DataFrame containing OOF predictions for each base model.
            targets: The true target values corresponding to the data.
        """
        # Construct a cache key based on Config and Data IDs
        config_hash = get_config_hash(Config)
        # Hash the IDs to ensure the data hasn't changed
        ids_hash = pd.util.hash_pandas_object(pd.Series(data_dict["ids"])).sum()
        cache_path = os.path.join(
            self.working_dir, f"oof_preds_{config_hash}_{ids_hash}.npz"
        )

        targets = data_dict["targets"]

        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached OOF predictions from {cache_path}")
            loaded = np.load(cache_path, allow_pickle=True)
            columns = loaded["columns"]
            oof_values = loaded["oof_preds"]
            return pd.DataFrame(oof_values, columns=columns), targets

        # 2. Run CV
        print(f"Starting {self.n_folds}-Fold Cross-Validation...")
        seed_everything(self.seed)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        model_names = get_base_models().keys()
        n_samples = len(data_dict["ids"])

        # Initialize OOF container
        oof_preds_df = pd.DataFrame(
            np.zeros((n_samples, len(model_names))), columns=model_names
        )

        for fold, (train_idx, val_idx) in enumerate(kf.split(data_dict["ids"])):
            print(f"  Processing Fold {fold + 1}/{self.n_folds}...")

            # Slice data
            train_fold = slice_data_dict(data_dict, train_idx)
            val_fold = slice_data_dict(data_dict, val_idx)

            # Fit Independent PCA on Train Fold ONLY
            pca = IndependentPCA(variance_threshold=Config.PCA_VARIANCE, seed=self.seed)
            pca.fit(train_fold)

            # Transform
            X_train = pca.transform(train_fold)
            y_train = train_fold["targets"]
            X_val = pca.transform(val_fold)

            # Train Base Models
            base_models = get_base_models()
            for name, model in base_models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                oof_preds_df.loc[val_idx, name] = preds

        # 3. Report Metrics
        print("\n=== CV Performance (RMSE) ===")
        for name in model_names:
            rmse = np.sqrt(mean_squared_error(targets, oof_preds_df[name]))
            print(f"{name}: {rmse:.6f}")

        # 4. Save Cache
        os.makedirs(self.working_dir, exist_ok=True)
        np.savez(
            cache_path,
            oof_preds=oof_preds_df.values,
            columns=list(oof_preds_df.columns),
            targets=targets,
        )
        print(f"Saved OOF predictions to {cache_path}")

        return oof_preds_df, targets


class FinalTrainer:
    """
    Handles the final training on the full dataset and prediction on the test set.
    """

    def __init__(self, seed=Config.SEED):
        self.seed = seed

    def train_and_predict(
        self, train_data_raw, test_data_raw, oof_preds_df, oof_targets
    ):
        """
        Trains Meta-Learner, retrains Base Models on full data, and generates submission.
        """
        print("\n=== Final Training & Inference ===")
        seed_everything(self.seed)

        # 1. Train Meta-Learner (Level 2)
        print("Training Meta-Learner (Stacking)...")
        meta_learner = get_meta_learner()
        meta_learner.fit(oof_preds_df.values, oof_targets)

        # Check Meta-Learner fit
        meta_oof_pred = meta_learner.predict(oof_preds_df.values)
        meta_rmse = np.sqrt(mean_squared_error(oof_targets, meta_oof_pred))
        print(f"Meta-Learner OOF RMSE: {meta_rmse:.6f}")

        # 2. Global Feature Compression
        print("Fitting Global PCA on Full Training Data...")
        pca = IndependentPCA(variance_threshold=Config.PCA_VARIANCE, seed=self.seed)
        pca.fit(train_data_raw)

        X_full = pca.transform(train_data_raw)
        y_full = train_data_raw["targets"]
        X_test = pca.transform(test_data_raw)

        print(f"Final Feature Dimension: {X_full.shape[1]}")

        # 3. Retrain Base Models (Level 1)
        print("Retraining Base Models on Full Training Data...")
        base_models = get_base_models()

        # Container for test predictions from base models
        test_base_preds = pd.DataFrame(
            np.zeros((len(test_data_raw["ids"]), len(base_models))),
            columns=base_models.keys(),
        )

        for name, model in base_models.items():
            # print(f"  Retraining {name}...")
            model.fit(X_full, y_full)
            test_base_preds[name] = model.predict(X_test)

        # 4. Final Prediction
        print("Generating Final Predictions via Meta-Learner...")
        final_preds = meta_learner.predict(test_base_preds.values)

        # Clip to valid range
        final_preds = np.clip(final_preds, 1.0, 100.0)

        # 5. Create Submission
        submission = pd.DataFrame(
            {"Id": test_data_raw["ids"], "Pawpularity": final_preds}
        )

        submission_path = Config.SUBMISSION_PATH
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

        return submission


def train_eval_pipeline(load_cached_data=True):
    """
    Main pipeline function to execute the training and evaluation process.
    """
    # 1. Feature Extraction
    # Extracts features for Train, Val, and Test sets using the defined backbones.
    extractor = FeatureExtractor()
    train_raw, val_raw, test_raw = extractor.extract_and_cache_features(
        load_cached_data=load_cached_data
    )

    # 2. Data Merging
    # Combine Train and Val sets to use all available labeled data for Cross-Validation.
    full_train_raw = merge_data_dicts(train_raw, val_raw)
    print(f"Merged Train+Val samples: {len(full_train_raw['ids'])}")

    # 3. Cross-Validation (Stacking Level 1)
    # Generate OOF predictions to train the Meta-Learner.
    cv = CrossValidator()
    oof_preds, targets = cv.run_cv(full_train_raw, load_cached_data=load_cached_data)

    # 4. Final Training & Submission
    # Train Meta-Learner, retrain Base Models on full data, predict Test set.
    trainer = FinalTrainer()
    trainer.train_and_predict(full_train_raw, test_raw, oof_preds, targets)
