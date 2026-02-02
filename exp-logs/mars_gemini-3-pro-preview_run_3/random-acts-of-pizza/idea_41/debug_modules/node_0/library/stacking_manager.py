import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

from library.config import Config
from library.utils import (
    get_logger,
    Timer,
    save_model,
    load_model,
)
from library.data_loader import load_and_preprocess_data
from library.feature_engineering import (
    LatentUserClusterer,
    SparseFeaturizer,
    TextEmbedder,
    MetadataSelector,
)
from library.model_factory import get_base_models, get_meta_learner

logger = get_logger("StackingManager")


class StackingPipeline:
    def __init__(self):
        self.base_models_config = get_base_models()
        self.meta_learner = get_meta_learner()
        self.n_folds = Config.N_FOLDS
        self.random_state = Config.RANDOM_STATE

        # Placeholders for fitted feature engineers (for final inference)
        self.final_fe = {"clusterer": None, "featurizer": None, "selector": None}

        # Placeholders for fitted models
        self.final_base_models = {}

        # Cache for frozen embeddings to avoid re-computing
        self.embeddings = {"train": None, "val": None, "test": None}

    def _get_model_features(self, model_name, feature_store, split_type="train"):
        """
        Constructs the specific feature matrix (X) for a given model based on its config.
        """
        config = self.base_models_config[model_name]
        required_sets = config["feature_sets"]
        is_sparse = config["sparse"]

        parts = []
        for fs in required_sets:
            data = feature_store[fs]
            # Ensure 2D
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            parts.append(data)

        if is_sparse:
            # Check if any part is sparse, if so use sparse hstack
            # If all are dense but model expects sparse, sparse.hstack works on dense too
            return sp.hstack(parts).tocsr()
        else:
            return np.hstack(parts)

    def _fit_transform_fe(
        self, df_train, df_val=None, df_test=None, fit=True, fe_objs=None
    ):
        """
        Runs the dynamic feature engineering pipeline.
        If fit=True, fits new objects on df_train.
        If fit=False, uses provided fe_objs.
        Returns dictionaries of features for provided dataframes.
        """

        # Initialize or use provided objects
        if fit:
            clusterer = LatentUserClusterer()
            featurizer = SparseFeaturizer()
            selector = MetadataSelector()

            # FIT
            # 1. Latent User Clustering
            clusterer.fit(df_train[Config.SUBREDDIT_COL])

            # 2. Sparse Featurizer
            featurizer.fit(df_train["text_concat"], df_train[Config.SUBREDDIT_COL])

            # Transform train to get latent features needed for MetadataSelector fit
            train_latent = clusterer.transform(df_train[Config.SUBREDDIT_COL])

            # 3. Metadata Selector
            selector.fit(df_train, train_latent)

            current_fe = {
                "clusterer": clusterer,
                "featurizer": featurizer,
                "selector": selector,
            }
        else:
            current_fe = fe_objs
            clusterer = current_fe["clusterer"]
            featurizer = current_fe["featurizer"]
            selector = current_fe["selector"]

        # Helper to transform a single df
        def transform_single(df):
            if df is None:
                return None

            # Latent
            latent = clusterer.transform(df[Config.SUBREDDIT_COL])

            # Sparse
            sparse_dict = featurizer.transform(
                df["text_concat"], df[Config.SUBREDDIT_COL]
            )

            # Metadata
            meta = selector.transform(df, latent)

            return {
                "lexical": sparse_dict["lexical"],
                "behavioral": sparse_dict["behavioral"],
                "metadata": meta,
                # Semantic is handled separately via pre-computed cache
            }

        feats_train = transform_single(df_train)
        feats_val = transform_single(df_val)
        feats_test = transform_single(df_test)

        return feats_train, feats_val, feats_test, current_fe

    def run(self):
        logger("Starting Stacking Pipeline...")

        # 1. Load Data
        train_df, val_df, test_df = load_and_preprocess_data(load_cached_data=True)
        y_train_full = train_df[Config.TARGET_COL].values

        # 2. Pre-compute Frozen Embeddings (Optimization)
        # These are deterministic and independent, so no leakage calculating them upfront.
        with Timer("Pre-computing Text Embeddings"):
            embedder = TextEmbedder()
            self.embeddings["train"] = embedder.transform(train_df["text_concat"])
            self.embeddings["val"] = embedder.transform(val_df["text_concat"])
            self.embeddings["test"] = embedder.transform(test_df["text_concat"])
            del embedder  # Free memory

        # 3. Level 1: Generate OOF Predictions
        logger(f"Starting Level 1: {self.n_folds}-Fold Cross-Validation...")

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        # Initialize OOF matrix
        model_names = list(self.base_models_config.keys())
        oof_preds = pd.DataFrame(0.0, index=train_df.index, columns=model_names)

        # Store scores
        fold_scores = {name: [] for name in model_names}

        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_train_full)):
            logger(f"  Processing Fold {fold + 1}/{self.n_folds}...")

            # Split Data
            fold_train_df = train_df.iloc[train_idx].copy()
            fold_val_df = train_df.iloc[val_idx].copy()

            y_fold_train = y_train_full[train_idx]
            y_fold_val = y_train_full[val_idx]

            # Dynamic Feature Engineering (Fit on Fold Train)
            # We don't transform test here, only fold train/val
            fts_train, fts_val, _, _ = self._fit_transform_fe(
                fold_train_df, fold_val_df, fit=True
            )

            # Inject pre-computed embeddings
            fts_train["semantic"] = self.embeddings["train"][train_idx]
            fts_val["semantic"] = self.embeddings["train"][val_idx]

            # Train Base Models
            for name, config in self.base_models_config.items():
                # Prepare X
                X_ft_train = self._get_model_features(name, fts_train)
                X_ft_val = self._get_model_features(name, fts_val)

                # Clone estimator
                model = clone(config["estimator"])

                # Fit
                # Check for early stopping support (XGB/LGBM)
                if (
                    "xgb" in str(type(model)).lower()
                    or "lgbm" in str(type(model)).lower()
                ):
                    model.fit(
                        X_ft_train,
                        y_fold_train,
                        eval_set=[(X_ft_val, y_fold_val)],
                        # early_stopping_rounds is deprecated in some versions but supported in kwargs or constructor
                        # We assume standard sklearn API wrapper behavior where it's often in fit params
                        # or handled via callbacks. For simplicity with standard wrappers:
                    )
                else:
                    model.fit(X_ft_train, y_fold_train)

                # Predict
                probs = model.predict_proba(X_ft_val)[:, 1]

                # Store OOF
                oof_preds.iloc[val_idx, oof_preds.columns.get_loc(name)] = probs

                # Score
                score = roc_auc_score(y_fold_val, probs)
                fold_scores[name].append(score)

        # Print OOF Scores
        logger("Level 1 OOF Performance:")
        for name, scores in fold_scores.items():
            mean_score = np.mean(scores)
            logger(f"  {name}: Mean AUC = {mean_score}")

        # 4. Level 2: Train Meta-Learner
        logger("Training Level 2 Meta-Learner on OOF predictions...")
        self.meta_learner.fit(oof_preds, y_train_full)

        # Check Meta-Learner Performance on OOF (approximate)
        meta_oof_probs = self.meta_learner.predict_proba(oof_preds)[:, 1]
        meta_score = roc_auc_score(y_train_full, meta_oof_probs)
        logger(
            f"  Meta-Learner OOF AUC: {meta_oof_probs}"
        )  # Printing array is messy, let's print score
        logger(f"  Meta-Learner OOF AUC Score: {meta_score}")

        # 5. Final Retraining on Full Train
        logger("Retraining Base Models on Full Training Set...")

        # Fit FE on Full Train
        # We use the global validation set (val_df) for early stopping during this phase
        full_feats_train, full_feats_val, full_feats_test, final_fe_objs = (
            self._fit_transform_fe(train_df, val_df, test_df, fit=True)
        )
        self.final_fe = final_fe_objs

        # Inject embeddings
        full_feats_train["semantic"] = self.embeddings["train"]
        full_feats_val["semantic"] = self.embeddings["val"]
        full_feats_test["semantic"] = self.embeddings["test"]

        y_val = val_df[Config.TARGET_COL].values

        # Train Models
        for name, config in self.base_models_config.items():
            logger(f"  Retraining {name}...")
            X_train = self._get_model_features(name, full_feats_train)
            X_val = self._get_model_features(name, full_feats_val)

            model = clone(config["estimator"])

            if "xgb" in str(type(model)).lower() or "lgbm" in str(type(model)).lower():
                # Use global validation set for early stopping
                model.fit(
                    X_train,
                    y_train_full,
                    eval_set=[(X_val, y_val)],
                )
            else:
                model.fit(X_train, y_train_full)

            self.final_base_models[name] = model

        # 6. Generate Submission
        logger("Generating Final Predictions...")

        # Base model predictions on Test
        test_base_preds = pd.DataFrame(0.0, index=test_df.index, columns=model_names)

        for name, model in self.final_base_models.items():
            X_test = self._get_model_features(name, full_feats_test)
            test_base_preds[name] = model.predict_proba(X_test)[:, 1]

        # Meta model prediction
        final_probs = self.meta_learner.predict_proba(test_base_preds)[:, 1]

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {
                "request_id": test_df[Config.ID_COL],
                "requester_received_pizza": final_probs,
            }
        )

        # Save
        save_path = Config.SUBMISSION_PATH
        logger(f"Saving submission to {save_path}...")
        submission.to_csv(save_path, index=False)
        logger("Pipeline Completed Successfully.")
