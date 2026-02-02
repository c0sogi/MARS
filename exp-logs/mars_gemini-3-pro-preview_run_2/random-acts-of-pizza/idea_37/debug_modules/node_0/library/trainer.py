import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import setup_logger, set_seed
from library.data_loader import load_dataset
from library.feature_engine import EmbeddingGenerator
from library.processors import FoldProcessor
from library.model_definitions import get_bagged_lr_pipeline, get_hyperparameter_grid


class CrossValidator:
    """
    Orchestrates the Telescoping Multi-View Consensus Ensemble (TMVCE) training pipeline.
    """

    def __init__(self):
        """
        Initializes the CrossValidator with logging and configuration.
        """
        self.logger = setup_logger(
            os.path.join(Config.WORKING_DIR, "training.log"), name="trainer"
        )
        self.n_splits = Config.N_SPLITS
        self.random_seed = Config.RANDOM_SEED

        # Ensure reproducibility
        set_seed(self.random_seed)

    def prepare_data(self):
        """
        Loads data, merges training splits for CV, and generates embeddings.

        Returns:
            tuple: (df_train, anchor_train, aux_train, df_test, anchor_test, aux_test)
        """
        self.logger.info("Loading and preparing data...")

        # Load datasets (cached if available)
        df_train_split, df_val_split, df_test = load_dataset(load_cached_data=True)

        # Merge provided train and val splits to perform our own Stratified K-Fold CV
        df_train = pd.concat([df_train_split, df_val_split], axis=0).reset_index(
            drop=True
        )

        # Generate Embeddings
        embedder = EmbeddingGenerator()
        embeddings = embedder.generate_dataset_embeddings(
            df_train_split, df_val_split, df_test, load_cached_data=True
        )

        # Merge embeddings corresponding to the merged df_train
        # Concatenate train and val embeddings along the 0-axis (samples)
        anchor_train = np.vstack(
            [embeddings["anchor"]["train"], embeddings["anchor"]["val"]]
        )
        aux_train = np.vstack([embeddings["aux"]["train"], embeddings["aux"]["val"]])

        # Test embeddings
        anchor_test = embeddings["anchor"]["test"]
        aux_test = embeddings["aux"]["test"]

        self.logger.info(f"Full Training Set Shape: {df_train.shape}")
        self.logger.info(f"Test Set Shape: {df_test.shape}")

        return df_train, anchor_train, aux_train, df_test, anchor_test, aux_test

    def tune_and_train(self, X, y, pipeline_name="Pipeline"):
        """
        Performs Grid Search to find best hyperparameters for the Bagged Logistic Regression
        and returns the best estimator.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            pipeline_name (str): Name for logging.

        Returns:
            estimator: The best fitted model from Grid Search.
        """
        # Retrieve hyperparameter grid
        raw_grid = get_hyperparameter_grid()

        # Prefix parameters with 'estimator__' because BaggingClassifier wraps the estimator
        param_grid = {f"estimator__{k}": v for k, v in raw_grid.items()}

        # Initialize base pipeline
        # We set n_jobs=1 for the bagging classifier to allow GridSearchCV to parallelize efficiently
        base_model = get_bagged_lr_pipeline(n_jobs=1)

        # Configure Grid Search
        # Using 3-fold inner CV for hyperparameter selection
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0,
        )

        self.logger.info(f"Starting Grid Search for {pipeline_name}...")
        grid_search.fit(X, y)

        self.logger.info(
            f"{pipeline_name} Best Score (Inner CV): {grid_search.best_score_}"
        )
        self.logger.info(f"{pipeline_name} Best Params: {grid_search.best_params_}")

        return grid_search.best_estimator_

    def run(self):
        """
        Executes the 5-Fold Stratified Cross-Validation loop.
        """
        # 1. Prepare Data
        (
            df_train,
            anchor_train_full,
            aux_train_full,
            df_test,
            anchor_test,
            aux_test,
        ) = self.prepare_data()

        y = df_train[Config.TARGET_COL].values.astype(int)

        # 2. Setup CV
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_seed
        )

        # Containers for predictions
        oof_preds = np.zeros(len(df_train))
        test_preds_accum = np.zeros(len(df_test))

        self.logger.info(
            f"Starting {self.n_splits}-Fold Stratified Cross-Validation..."
        )

        for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, y)):
            fold_num = fold + 1
            self.logger.info(f"--- Fold {fold_num}/{self.n_splits} ---")

            # 3. Split Data for this Fold
            df_fold_train = df_train.iloc[train_idx]
            df_fold_val = df_train.iloc[val_idx]
            y_fold_train = y[train_idx]
            y_fold_val = y[val_idx]

            # Split Embeddings
            anchor_fold_train = anchor_train_full[train_idx]
            aux_fold_train = aux_train_full[train_idx]
            anchor_fold_val = anchor_train_full[val_idx]
            aux_fold_val = aux_train_full[val_idx]

            # 4. Feature Processing (Stateful per fold)
            processor = FoldProcessor()

            # Fit on training data and transform training data
            views_train = processor.fit_transform(
                df_fold_train, anchor_fold_train, aux_fold_train
            )

            # Transform validation data
            views_val = processor.transform(df_fold_val, anchor_fold_val, aux_fold_val)

            # Transform test data (using this fold's processor)
            views_test = processor.transform(df_test, anchor_test, aux_test)

            # 5. Pipeline A: Parsimonious Expert (Anchor + Meta)
            model_a = self.tune_and_train(
                views_train["view_A"], y_fold_train, pipeline_name="Pipeline A"
            )

            # 6. Pipeline B: Augmented Expert (Anchor + Aux + Meta)
            model_b = self.tune_and_train(
                views_train["view_B"], y_fold_train, pipeline_name="Pipeline B"
            )

            # 7. Inference & Consensus

            # Validation Inference
            prob_a_val = model_a.predict_proba(views_val["view_A"])[:, 1]
            prob_b_val = model_b.predict_proba(views_val["view_B"])[:, 1]

            # Soft Voting Consensus
            prob_val_consensus = 0.5 * prob_a_val + 0.5 * prob_b_val
            oof_preds[val_idx] = prob_val_consensus

            fold_auc = roc_auc_score(y_fold_val, prob_val_consensus)
            self.logger.info(f"Fold {fold_num} AUC: {fold_auc}")

            # Test Inference (CV-Bagging)
            prob_a_test = model_a.predict_proba(views_test["view_A"])[:, 1]
            prob_b_test = model_b.predict_proba(views_test["view_B"])[:, 1]
            prob_test_consensus = 0.5 * prob_a_test + 0.5 * prob_b_test

            test_preds_accum += prob_test_consensus

        # 8. Evaluation
        overall_auc = roc_auc_score(y, oof_preds)
        self.logger.info(f"Overall OOF AUC: {overall_auc}")

        # 9. Submission Generation
        avg_test_preds = test_preds_accum / self.n_splits
        self.save_submission(df_test, avg_test_preds)

    def save_submission(self, df_test, preds):
        """
        Saves the predictions to a CSV file in the required format.

        Args:
            df_test (pd.DataFrame): Test dataframe containing request_id.
            preds (np.ndarray): Predicted probabilities.
        """
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        submission = pd.DataFrame(
            {"request_id": df_test["request_id"], "requester_received_pizza": preds}
        )

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
