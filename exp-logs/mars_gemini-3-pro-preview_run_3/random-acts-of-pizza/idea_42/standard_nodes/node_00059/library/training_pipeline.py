import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from library import config, utils, data_loader, feature_engineering, model_definitions

# Initialize logger
logger = utils.setup_logging("training_pipeline")


def _prepare_model_inputs(dynamic_features, static_features, indices=None):
    """
    Helper to assemble the specific feature views for the Pent-View architecture.

    Args:
        dynamic_features (dict): Output from DynamicFeatureExtractor.transform
        static_features (dict): Output from StaticFeatureExtractor.extract
        indices (np.array, optional): Indices to slice static features (if not already sliced).

    Returns:
        dict: Mapping of model_key -> feature_matrix (X)
    """
    # Unpack Dynamic Features
    X_lexical = dynamic_features["X_lexical"]
    X_behavioral = dynamic_features["X_behavioral_sparse"]
    X_community_latent = dynamic_features["X_community_latent"]
    X_metadata_scaled = dynamic_features["X_metadata_scaled"]

    # Unpack Static Features (Slice if necessary)
    if indices is not None:
        embeddings = static_features["embeddings"][indices]
    else:
        embeddings = static_features["embeddings"]

    # 1. Sparse Lexical Branch (Lexical Bagger)
    # View: TF-IDF Text + Scaled Metadata
    # Note: We stack sparse and dense.
    X_lexical_view = sp.hstack([X_lexical, sp.csr_matrix(X_metadata_scaled)])

    # 2. Sparse Behavioral Branch (Community Bagger)
    # View: TF-IDF Subreddits + Scaled Metadata
    X_community_view = sp.hstack([X_behavioral, sp.csr_matrix(X_metadata_scaled)])

    # 3. Dense Semantic Branch (Semantic Booster & Bagger)
    # View: Dense Embeddings + Latent Community Topics (NMF) + Scaled Metadata
    # All are dense arrays.
    X_semantic_view = np.hstack([embeddings, X_community_latent, X_metadata_scaled])

    # 4. Contextual Branch (Metadata Anchor)
    # View: Scaled Metadata Only
    X_contextual_view = X_metadata_scaled

    return {
        "lexical_rf": X_lexical_view,
        "community_rf": X_community_view,
        "semantic_xgb": X_semantic_view,
        "semantic_rf": X_semantic_view,
        "metadata_lr": X_contextual_view,
    }


class StackingTrainer:
    """
    Manages the Level 1 Cross-Validation Loop (OOF Generation).
    """

    def __init__(self, n_folds=config.N_FOLDS, random_state=config.SEED):
        self.n_folds = n_folds
        self.random_state = random_state
        self.models_def = model_definitions.get_level1_models()
        self.model_keys = list(self.models_def.keys())

    def run_cv(self, train_df, static_features):
        """
        Executes the CV loop.

        Returns:
            oof_preds (pd.DataFrame): OOF predictions for all models.
            y_train (pd.Series): Aligned target values.
        """
        logger.info(f"Starting {self.n_folds}-Fold Cross-Validation...")

        y = train_df["requester_received_pizza"].values
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        # Initialize OOF matrix
        oof_matrix = np.zeros((len(train_df), len(self.model_keys)))

        # Static metadata dataframe is needed for DynamicExtractor fitting
        static_meta_df = static_features["metadata"]

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, y)):
            logger.info(f"Processing Fold {fold_idx + 1}/{self.n_folds}")

            # 1. Slice Data
            X_train_df = train_df.iloc[train_idx].reset_index(drop=True)
            X_val_df = train_df.iloc[val_idx].reset_index(drop=True)
            y_fold_train = y[train_idx]
            y_fold_val = y[val_idx]

            # Slice Static Metadata for Dynamic Fitting
            meta_train_df = static_meta_df.iloc[train_idx].reset_index(drop=True)
            meta_val_df = static_meta_df.iloc[val_idx].reset_index(drop=True)

            # 2. Fit Dynamic Features on Fold Train (Leakage Prevention)
            dynamic_extractor = feature_engineering.DynamicFeatureExtractor()
            dynamic_extractor.fit(X_train_df, meta_train_df)

            # 3. Transform Train and Val
            dynamic_train = dynamic_extractor.transform(X_train_df, meta_train_df)
            dynamic_val = dynamic_extractor.transform(X_val_df, meta_val_df)

            # 4. Prepare Model Inputs
            # We pass indices to _prepare_model_inputs to slice the static embeddings correctly
            inputs_train = _prepare_model_inputs(
                dynamic_train, static_features, train_idx
            )
            inputs_val = _prepare_model_inputs(dynamic_val, static_features, val_idx)

            # 5. Train and Predict
            for i, key in enumerate(self.model_keys):
                model = model_definitions.get_level1_models()[key]  # Get fresh instance
                X_tr = inputs_train[key]
                X_va = inputs_val[key]

                # Handle XGBoost specific params
                if key == "semantic_xgb":
                    # Calculate scale_pos_weight
                    n_pos = np.sum(y_fold_train)
                    n_neg = len(y_fold_train) - n_pos
                    scale_weight = n_neg / n_pos if n_pos > 0 else 1.0
                    model.set_params(scale_pos_weight=scale_weight)

                    # Train with early stopping (using fold val)
                    # Note: For CV, we use the fold validation set for ES to get a good OOF model
                    model.fit(
                        X_tr, y_fold_train, eval_set=[(X_va, y_fold_val)], verbose=False
                    )
                else:
                    model.fit(X_tr, y_fold_train)

                # Predict
                if hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X_va)[:, 1]
                else:
                    preds = model.predict(X_va)  # Fallback

                oof_matrix[val_idx, i] = preds

        # Calculate and Print Metrics
        logger.info("CV Complete. OOF Scores:")
        for i, key in enumerate(self.model_keys):
            auc = roc_auc_score(y, oof_matrix[:, i])
            print(f"Model {key} OOF AUC: {auc}")

        return pd.DataFrame(oof_matrix, columns=self.model_keys), y


class FinalRetrainer:
    """
    Handles Final Retraining on Full Data and Test Prediction.
    """

    def __init__(self):
        self.models_def = model_definitions.get_level1_models()
        self.model_keys = list(self.models_def.keys())
        self.meta_learner = model_definitions.get_meta_learner()

    def run(
        self,
        train_df,
        val_df,
        test_df,
        static_train,
        static_val,
        static_test,
        oof_df,
        y_train_oof,
    ):
        """
        Args:
            train_df: Global Train set (~80%)
            val_df: Global Val set (~20%)
            test_df: Test set
            static_train: Static features for train
            static_val: Static features for val
            static_test: Static features for test
            oof_df: OOF predictions from StackingTrainer
            y_train_oof: Target for OOF (aligned with train_df)
        """
        logger.info("Starting Final Retraining Protocol...")

        # 1. Train Meta-Learner
        logger.info("Training Level 2 Meta-Learner on OOF predictions...")
        self.meta_learner.fit(oof_df, y_train_oof)

        # 2. Prepare Full Training Data (Train + Val) for Dynamic Fitting
        # We concatenate DFs and Metadata for fitting the NMF/TF-IDF
        full_train_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
        full_meta_df = pd.concat(
            [static_train["metadata"], static_val["metadata"]],
            axis=0,
            ignore_index=True,
        )
        y_full = full_train_df["requester_received_pizza"].values

        # Concatenate embeddings for input preparation
        full_embeddings = np.vstack(
            [static_train["embeddings"], static_val["embeddings"]]
        )
        static_full = {"embeddings": full_embeddings, "metadata": full_meta_df}

        # 3. Fit Dynamic Extractor on Full Data
        logger.info("Fitting DynamicFeatureExtractor on Full Training Set...")
        dynamic_extractor = feature_engineering.DynamicFeatureExtractor()
        dynamic_extractor.fit(full_train_df, full_meta_df)

        # 4. Transform All Sets
        logger.info("Transforming all datasets...")
        # Full Train (for RF/Linear)
        dyn_full = dynamic_extractor.transform(full_train_df, full_meta_df)
        inputs_full = _prepare_model_inputs(dyn_full, static_full)

        # Train Only (for XGB training)
        dyn_train = dynamic_extractor.transform(train_df, static_train["metadata"])
        inputs_train = _prepare_model_inputs(dyn_train, static_train)

        # Val Only (for XGB Early Stopping)
        dyn_val = dynamic_extractor.transform(val_df, static_val["metadata"])
        inputs_val = _prepare_model_inputs(dyn_val, static_val)

        # Test Set
        dyn_test = dynamic_extractor.transform(test_df, static_test["metadata"])
        inputs_test = _prepare_model_inputs(dyn_test, static_test)

        # 5. Retrain Level 1 Models
        test_preds_l1 = np.zeros((len(test_df), len(self.model_keys)))

        for i, key in enumerate(self.model_keys):
            logger.info(f"Retraining {key}...")
            model = model_definitions.get_level1_models()[key]  # Fresh instance

            if key == "semantic_xgb":
                # XGBoost: Train on Train, ES on Val
                y_tr = train_df["requester_received_pizza"].values
                y_va = val_df["requester_received_pizza"].values

                # Calc scale_pos_weight
                n_pos = np.sum(y_tr)
                n_neg = len(y_tr) - n_pos
                scale_weight = n_neg / n_pos if n_pos > 0 else 1.0
                model.set_params(scale_pos_weight=scale_weight)

                model.fit(
                    inputs_train[key],
                    y_tr,
                    eval_set=[(inputs_val[key], y_va)],
                    verbose=False,
                )
            else:
                # RF/Linear: Train on Full (Train + Val)
                model.fit(inputs_full[key], y_full)

            # Predict on Test
            if hasattr(model, "predict_proba"):
                test_preds_l1[:, i] = model.predict_proba(inputs_test[key])[:, 1]
            else:
                test_preds_l1[:, i] = model.predict(inputs_test[key])

        # 6. Final Prediction via Meta-Learner
        logger.info("Generating Final Predictions via Meta-Learner...")
        final_probs = self.meta_learner.predict_proba(test_preds_l1)[:, 1]

        # 7. Save Submission
        submission = pd.DataFrame(
            {
                "request_id": test_df["request_id"],
                "requester_received_pizza": final_probs,
            }
        )

        utils.save_data_cache(
            submission, config.SUBMISSION_FILE_PATH
        )  # Handles dir creation
        # Also save as CSV explicitly for competition format (header included by to_csv default)
        submission.to_csv(config.SUBMISSION_FILE_PATH, index=False)
        logger.info(f"Submission saved to {config.SUBMISSION_FILE_PATH}")


def run_training_pipeline(debug_size=config.DEBUG_SAMPLE_SIZE):
    """
    Main entry point for the training pipeline.
    """
    utils.set_seed(config.SEED)

    # 1. Load Data
    train_df, val_df = data_loader.load_dataset("train", debug_size=debug_size)
    test_df = data_loader.load_dataset("test", debug_size=debug_size)

    # 2. Static Feature Extraction (Cached)
    static_extractor = feature_engineering.StaticFeatureExtractor()

    # Note: We extract static features for the specific splits.
    # Ideally, we should have a consistent cache key.
    # We use 'train' for the train_df loaded from metadata/train.parquet
    # We use 'val' for the val_df loaded from metadata/val.parquet
    static_train = static_extractor.extract(train_df, "train_split")
    static_val = static_extractor.extract(val_df, "val_split")
    static_test = static_extractor.extract(test_df, "test")

    # 3. Level 1 Stacking (OOF Generation on Train Split)
    # We run CV on the 'train_df' (80% split) to generate OOFs for training the meta-learner.
    stacker = StackingTrainer()
    oof_df, y_train_oof = stacker.run_cv(train_df, static_train)

    # Save OOF for analysis
    utils.save_data_cache(
        oof_df, os.path.join(config.PREDICTIONS_DIR, "oof_predictions.parquet")
    )

    # 4. Final Retraining and Submission
    retrainer = FinalRetrainer()
    retrainer.run(
        train_df,
        val_df,
        test_df,
        static_train,
        static_val,
        static_test,
        oof_df,
        y_train_oof,
    )

    logger.info("Pipeline completed successfully.")
