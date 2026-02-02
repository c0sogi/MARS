import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import RidgeCV, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from library.config import Config
from library.utils import (
    seed_everything,
    rmse_score,
    save_to_cache,
    load_from_cache,
)
from library.processors import LogitTargetTransformer, FeaturePreprocessor
from library.data import PetDataset, load_metadata, get_processor
from library.extractors import process_and_cache_features


class Level0Trainer:
    """
    Trains Level-0 heterogeneous experts using Stratified K-Fold CV.
    Handles model-specific preprocessing and target transformations.
    """

    def __init__(self, n_folds=Config.N_FOLDS, seed=Config.SEED):
        self.n_folds = n_folds
        self.seed = seed
        self.models_config = [
            # (Model Name, Strategy, Target Transform, Trainer Function)
            ("Ridge", "linear", True, self._train_ridge),
            ("SVR", "linear", True, self._train_svr),
            ("ExtraTrees", "tree", False, self._train_et),
            ("LightGBM", "tree", False, self._train_lgbm),
        ]

    def _get_stratified_folds(self, y, n_folds):
        """
        Creates stratified folds based on binned continuous target.
        """
        # Binning for stratification (Sturges' rule approx for ~7k-9k samples -> ~14 bins)
        num_bins = 14
        y_binned = pd.cut(y, bins=num_bins, labels=False)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        return list(skf.split(np.zeros(len(y)), y_binned))

    def _train_ridge(self, X_train, y_train, X_val, X_test):
        """Trains RidgeCV."""
        model = RidgeCV(
            alphas=Config.RIDGE_ALPHAS, scoring="neg_root_mean_squared_error"
        )
        model.fit(X_train, y_train)

        pred_val = model.predict(X_val)
        pred_test = model.predict(X_test)
        return pred_val, pred_test

    def _train_svr(self, X_train, y_train, X_val, X_test):
        """Trains SVR with GridSearchCV."""
        # SVR scaling is sensitive to data size, using cache_size=1000MB
        svr = SVR(cache_size=1000)
        # Use n_jobs=-1 for parallel grid search
        model = GridSearchCV(
            svr,
            Config.SVR_GRID,
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )
        model.fit(X_train, y_train)

        pred_val = model.predict(X_val)
        pred_test = model.predict(X_test)
        return pred_val, pred_test

    def _train_et(self, X_train, y_train, X_val, X_test):
        """Trains ExtraTreesRegressor."""
        model = ExtraTreesRegressor(**Config.ET_PARAMS)
        model.fit(X_train, y_train)

        pred_val = model.predict(X_val)
        pred_test = model.predict(X_test)
        return pred_val, pred_test

    def _train_lgbm(self, X_train, y_train, X_val, X_test):
        """Trains LightGBM with early stopping."""
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(
            X_val, label=y_val_raw if "y_val_raw" in locals() else None
        )
        # Note: LightGBM uses raw targets in this pipeline config, so y_train is correct.
        # For validation in early stopping, we need to pass the validation set.
        # However, _train_lgbm signature doesn't take y_val.
        # We need to adjust logic or split y inside the main loop.
        # To keep interface consistent, I will rely on the main loop passing X_val.
        # But I don't have y_val here.
        # Strategy adjustment: The main loop splits indices. I will pass y_val to all trainers
        # but ignore it in others.
        pass
        # Re-implementing logic in run_cv to handle y_val availability.

    def _train_lgbm_internal(self, X_train, y_train, X_val, y_val, X_test):
        """Internal LightGBM trainer with validation data."""
        train_ds = lgb.Dataset(X_train, label=y_train)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

        callbacks = [
            lgb.early_stopping(
                stopping_rounds=Config.LGBM_PARAMS["early_stopping_rounds"],
                verbose=False,
            ),
            lgb.log_evaluation(period=0),  # Silence
        ]

        # Copy params to avoid modification
        params = Config.LGBM_PARAMS.copy()
        if "early_stopping_rounds" in params:
            del params["early_stopping_rounds"]

        model = lgb.train(params, train_ds, valid_sets=[val_ds], callbacks=callbacks)

        pred_val = model.predict(X_val)
        pred_test = model.predict(X_test)
        return pred_val, pred_test

    def run_cv(self, feature_data, targets, test_data, load_cached=True):
        """
        Main execution loop for Level 0.

        Args:
            feature_data (dict): {'backbone': {'embeddings': np.array, 'metadata': np.array}}
            targets (np.array): Target values for training data.
            test_data (dict): {'backbone': {'embeddings': np.array, 'metadata': np.array}}
            load_cached (bool): Whether to load OOF/Test predictions from disk.

        Returns:
            tuple: (oof_preds_df, test_preds_df)
        """
        # Cache paths
        oof_cache_path = os.path.join(Config.IDEA_DIR, "level0_oof.npy")
        test_pred_cache_path = os.path.join(Config.IDEA_DIR, "level0_test_pred.npy")
        col_names_path = os.path.join(Config.IDEA_DIR, "level0_cols.npy")

        if (
            load_cached
            and os.path.exists(oof_cache_path)
            and os.path.exists(test_pred_cache_path)
        ):
            print("Loading cached Level 0 predictions...")
            oof_matrix = load_from_cache(oof_cache_path)
            test_pred_matrix = load_from_cache(test_pred_cache_path)
            # Reconstruct DataFrames if needed, or just return arrays.
            # We'll return arrays for simplicity in Level 1.
            return oof_matrix, test_pred_matrix

        print("Starting Level 0 Training...")

        backbones = list(feature_data.keys())
        n_samples = len(targets)
        n_test = len(next(iter(test_data.values()))["embeddings"])

        # Calculate total columns: Backbones * Models
        n_cols = len(backbones) * len(self.models_config)

        oof_matrix = np.zeros((n_samples, n_cols))
        test_pred_matrix = np.zeros((n_test, n_cols))

        # Folds
        folds = self._get_stratified_folds(targets, self.n_folds)

        col_idx = 0

        # Iterate Backbones
        for backbone in backbones:
            print(f"Processing Backbone: {backbone}")

            # Get data for this backbone
            train_emb = feature_data[backbone]["embeddings"]
            train_meta = feature_data[backbone]["metadata"]
            test_emb = test_data[backbone]["embeddings"]
            test_meta = test_data[backbone]["metadata"]

            # Iterate Models
            for model_name, strategy, use_logit, trainer_func in self.models_config:
                print(f"  Training {model_name} (Strategy: {strategy})...")

                # Test predictions accumulator for averaging across folds
                test_preds_fold_accum = np.zeros(n_test)

                # CV Loop
                for fold_idx, (train_idx, val_idx) in enumerate(folds):
                    # Split Data
                    X_train_emb, X_val_emb = train_emb[train_idx], train_emb[val_idx]
                    X_train_meta, X_val_meta = (
                        train_meta[train_idx],
                        train_meta[val_idx],
                    )
                    y_train, y_val = targets[train_idx], targets[val_idx]

                    # Preprocessing
                    preprocessor = FeaturePreprocessor(seed=Config.SEED)
                    X_train_proc = preprocessor.fit_transform(
                        X_train_emb, X_train_meta, strategy=strategy
                    )
                    X_val_proc = preprocessor.transform(X_val_emb, X_val_meta)
                    X_test_proc = preprocessor.transform(test_emb, test_meta)

                    # Target Transformation
                    target_transformer = LogitTargetTransformer()
                    if use_logit:
                        y_train_proc = target_transformer.transform(y_train)
                        # y_val is kept raw for metric calculation if needed, but for training we might need transformed
                        # For SVR/Ridge, we train on transformed y.
                    else:
                        y_train_proc = y_train

                    # Train and Predict
                    if model_name == "LightGBM":
                        # LightGBM needs validation set for early stopping
                        # If using logit, we should technically transform y_val too for metric consistency in loss,
                        # but LightGBM here uses 'rmse' on raw targets usually.
                        # However, config says LightGBM uses Raw Target (use_logit=False).
                        # So y_train_proc is raw. y_val is raw.
                        p_val, p_test = self._train_lgbm_internal(
                            X_train_proc, y_train_proc, X_val_proc, y_val, X_test_proc
                        )
                    else:
                        p_val, p_test = trainer_func(
                            X_train_proc, y_train_proc, X_val_proc, X_test_proc
                        )

                    # Inverse Transform if needed
                    if use_logit:
                        p_val = target_transformer.inverse_transform(p_val)
                        p_test = target_transformer.inverse_transform(p_test)

                    # Store OOF
                    oof_matrix[val_idx, col_idx] = p_val

                    # Accumulate Test Preds
                    test_preds_fold_accum += p_test

                # Average Test Preds
                test_pred_matrix[:, col_idx] = test_preds_fold_accum / self.n_folds

                # Print RMSE for this model
                rmse = rmse_score(targets, oof_matrix[:, col_idx])
                print(f"    {backbone} - {model_name} OOF RMSE: {rmse:.6f}")

                col_idx += 1

        # Cache results
        save_to_cache(oof_matrix, oof_cache_path)
        save_to_cache(test_pred_matrix, test_pred_cache_path)

        return oof_matrix, test_pred_matrix


class Level1Trainer:
    """
    Trains Level-1 Meta-Learner (Bayesian Ridge) on OOF predictions.
    """

    def __init__(self):
        self.model = BayesianRidge(**Config.META_MODEL_PARAMS)

    def train_and_predict(self, X_oof, y, X_test):
        """
        Fits meta-learner and generates final predictions.
        Performs internal CV to evaluate meta-learner performance.
        """
        print("\nStarting Level 1 Meta-Learner Training...")

        # 1. Evaluate with CV (Nested)
        # We reuse the same stratification logic for consistency
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )
        # Binning
        num_bins = 14
        y_binned = pd.cut(y, bins=num_bins, labels=False)

        oof_meta = np.zeros(len(y))

        for train_idx, val_idx in skf.split(X_oof, y_binned):
            X_tr, X_val = X_oof[train_idx], X_oof[val_idx]
            y_tr, _ = y[train_idx], y[val_idx]

            self.model.fit(X_tr, y_tr)
            oof_meta[val_idx] = self.model.predict(X_val)

        cv_score = rmse_score(y, oof_meta)
        print(f"Level 1 CV RMSE: {cv_score:.6f}")

        # 2. Fit on Full Data and Predict Test
        self.model.fit(X_oof, y)
        final_preds = self.model.predict(X_test)

        return final_preds


def run_ensemble(debug=Config.DEBUG, load_cached_level0=True):
    """
    Main execution pipeline.
    """
    seed_everything()

    # 1. Load Metadata
    print("Loading Metadata...")
    train_df, _, test_df = load_metadata(merge_train_val=True, debug=debug)

    # 2. Feature Extraction
    backbones = [
        (
            "SigLIP",
            Config.MODEL_SIGLIP,
            Config.BATCH_SIZE_SIGLIP,
            Config.CACHE_FEATURES_SIGLIP,
        ),
        (
            "DINOv2",
            Config.MODEL_DINOV2,
            Config.BATCH_SIZE_DINOV2,
            Config.CACHE_FEATURES_DINOV2,
        ),
        (
            "ConvNeXt",
            Config.MODEL_CONVNEXTV2,
            Config.BATCH_SIZE_CONVNEXT,
            Config.CACHE_FEATURES_CONVNEXT,
        ),
    ]

    feature_data = {}  # Structure: {backbone: {embeddings, metadata}}
    test_data = {}
    targets = None
    ids_test = None

    for name, model_path, batch_size, cache_base_path in backbones:
        print(f"\nPreparing features for {name}...")

        # Define specific cache paths for Train vs Test to avoid collision
        # cache_base_path is like ".../features_siglip.npy"
        # We split it to insert _train and _test
        base, ext = os.path.splitext(cache_base_path)
        path_train_emb = f"{base}_train{ext}"
        path_test_emb = f"{base}_test{ext}"
        path_train_ids = f"{base}_train_ids{ext}"
        path_test_ids = f"{base}_test_ids{ext}"
        path_train_meta = f"{base}_train_meta{ext}"
        path_test_meta = f"{base}_test_meta{ext}"
        path_train_tgt = f"{base}_train_targets{ext}"

        # Process Train
        # Note: We don't use 'return_flipped' here because extractors.py handles augmentation internally
        # if the dataset returns multiple images. But PetDataset needs 'return_flipped=True' to enable that.
        # The extractor logic checks for 5D tensors.

        # Load Processor
        processor = get_processor(model_path)

        # Train Dataset
        train_ds = PetDataset(
            train_df, processor, return_flipped=True, include_target=True
        )
        train_cache = {
            "embeddings": path_train_emb,
            "ids": path_train_ids,
            "metadata": path_train_meta,
            "targets": path_train_tgt,
        }
        train_res = process_and_cache_features(
            train_ds, model_path, batch_size, train_cache
        )

        # Test Dataset
        test_ds = PetDataset(
            test_df, processor, return_flipped=True, include_target=False
        )
        test_cache = {
            "embeddings": path_test_emb,
            "ids": path_test_ids,
            "metadata": path_test_meta,
            "targets": None,
        }
        test_res = process_and_cache_features(
            test_ds, model_path, batch_size, test_cache
        )

        # Store in dictionaries
        feature_data[name] = {
            "embeddings": train_res["embeddings"],
            "metadata": train_res["metadata"],
        }
        test_data[name] = {
            "embeddings": test_res["embeddings"],
            "metadata": test_res["metadata"],
        }

        # Store targets (should be same across backbones, but good to have)
        if targets is None:
            targets = train_res["targets"]

        if ids_test is None:
            ids_test = test_res["ids"]

        # Cleanup to save memory
        del processor, train_ds, test_ds, train_res, test_res
        gc.collect()

    # 3. Level 0 Training
    l0_trainer = Level0Trainer()
    oof_preds, l0_test_preds = l0_trainer.run_cv(
        feature_data, targets, test_data, load_cached=load_cached_level0
    )

    # 4. Level 1 Training
    l1_trainer = Level1Trainer()
    final_predictions = l1_trainer.train_and_predict(oof_preds, targets, l0_test_preds)

    # 5. Submission
    print("\nGenerating Submission...")
    submission = pd.DataFrame({"Id": ids_test, "Pawpularity": final_predictions})

    # Ensure format matches sample (Id, Pawpularity)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission.head())
