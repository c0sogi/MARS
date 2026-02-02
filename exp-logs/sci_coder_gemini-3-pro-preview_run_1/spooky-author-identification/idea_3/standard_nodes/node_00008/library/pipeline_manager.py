import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from library.config import Config
from library.utils import seed_everything, save_submission
from library.data_processing import load_data, MetaFeatureExtractor
from library.model_tfidf import TfidfExpert
from library.model_transformer import Trainer
from library.model_stacking import StackingMetaLearner


class CrossValidationRunner:
    """
    Handles the 5-Fold Cross-Validation process to generate Out-of-Fold (OOF) predictions
    for the Level 1 base models.
    """

    def __init__(self, df, n_folds=Config.N_FOLDS):
        self.df = df
        self.n_folds = n_folds
        self.skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=Config.SEED
        )

    def run(self):
        """
        Executes the CV loop.

        Returns:
            tuple: (oof_tfidf, oof_transformer)
                   Two numpy arrays of shape (n_samples, 3) containing OOF probabilities.
        """
        # Initialize OOF arrays
        n_samples = len(self.df)
        n_classes = len(Config.LABELS)
        oof_tfidf = np.zeros((n_samples, n_classes))
        oof_transformer = np.zeros((n_samples, n_classes))

        print(
            f"Starting {self.n_folds}-Fold Cross-Validation on {n_samples} samples..."
        )

        for fold, (train_idx, val_idx) in enumerate(
            self.skf.split(self.df, self.df["author"])
        ):
            print(f"\n=== Fold {fold + 1}/{self.n_folds} ===")

            # Create Fold Data
            train_fold = self.df.iloc[train_idx].reset_index(drop=True)
            val_fold = self.df.iloc[val_idx].reset_index(drop=True)

            # --- 1. TF-IDF Expert ---
            print(">> Training TF-IDF Expert...")
            tfidf_model = TfidfExpert()

            # Force re-computation of features for this specific split to avoid leakage
            # We pass val_fold as the second argument (validation set)
            # We pass val_fold as the third argument (dummy test set) just to satisfy signature
            X_train, X_val, _ = tfidf_model.get_features(
                train_fold["text"],
                val_fold["text"],
                val_fold["text"],
                load_cached_data=False,
            )

            y_train = train_fold["author"].map(Config.LABEL2ID).values

            tfidf_model.fit(X_train, y_train)
            probs_tfidf = tfidf_model.predict_proba(X_val)
            oof_tfidf[val_idx] = probs_tfidf

            # --- 2. Transformer Expert ---
            print(">> Training Transformer Expert...")
            transformer_trainer = Trainer()

            # The Trainer handles tokenization and dataset creation internally
            transformer_trainer.fit(
                train_fold["text"],
                train_fold["author"],
                val_fold["text"],
                val_fold["author"],
                fold_idx=fold,
            )

            # Generate predictions for the validation fold
            probs_transformer = transformer_trainer.predict(val_fold["text"])
            oof_transformer[val_idx] = probs_transformer

        return oof_tfidf, oof_transformer


class PipelineManager:
    """
    Orchestrates the entire pipeline:
    1. Load Data
    2. Run CV to get OOF predictions
    3. Train Meta-Learner
    4. Retrain Level 1 models on full data
    5. Generate Final Submission
    """

    def __init__(self):
        seed_everything()
        self.train_df_orig, self.val_df_orig, self.test_df = load_data()

        # Combine original train and validation sets for maximum data utility
        # during Cross-Validation and Final Training
        self.full_train_df = pd.concat(
            [self.train_df_orig, self.val_df_orig]
        ).reset_index(drop=True)

        self.meta_learner = StackingMetaLearner()

    def run_cv_and_train_meta(self):
        """
        Runs CV to generate OOFs, extracts meta-features, and trains the XGBoost Meta-Learner.
        """
        print("\n" + "=" * 40)
        print("PHASE 1: Cross-Validation & Meta-Learner Training")
        print("=" * 40)

        # 1. Generate OOF Predictions
        cv_runner = CrossValidationRunner(self.full_train_df)
        oof_tfidf, oof_transformer = cv_runner.run()

        # Save OOFs for debugging/safety
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        np.save(os.path.join(Config.WORKING_DIR, "oof_tfidf.npy"), oof_tfidf)
        np.save(
            os.path.join(Config.WORKING_DIR, "oof_transformer.npy"), oof_transformer
        )

        # 2. Extract Meta-Features for Full Train
        print("\nExtracting Meta-Features for full training set...")
        meta_extractor = MetaFeatureExtractor()
        # Use a unique identifier for caching
        meta_features = meta_extractor.get_features(
            self.full_train_df, "full_train_merged"
        )

        # 3. Prepare Stacking Data
        # Combine L1 probs and meta-features
        X_stack = self.meta_learner.prepare_meta_features(
            [oof_tfidf, oof_transformer], meta_features
        )
        y_stack = self.full_train_df["author"].map(Config.LABEL2ID).values

        # 4. Train Meta-Learner
        # We need a validation set for XGBoost early stopping.
        # We split the OOF data 80/20.
        X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
            X_stack, y_stack, test_size=0.2, random_state=Config.SEED, stratify=y_stack
        )

        self.meta_learner.fit(X_meta_train, y_meta_train, X_meta_val, y_meta_val)

        return self.meta_learner

    def retrain_l1_and_predict_test(self):
        """
        Retrains L1 models on the full dataset, generates test features,
        and uses the trained Meta-Learner for final prediction.
        """
        print("\n" + "=" * 40)
        print("PHASE 2: Full Retraining & Test Prediction")
        print("=" * 40)

        # --- 1. TF-IDF Expert ---
        print(">> Retraining TF-IDF Expert on full data...")
        tfidf_model = TfidfExpert()

        # We need features for full_train (to fit) and test (to predict)
        # We pass test_df['text'] as the dummy validation set
        X_train, _, X_test = tfidf_model.get_features(
            self.full_train_df["text"],
            self.test_df["text"],
            self.test_df["text"],
            load_cached_data=False,
        )

        y_train = self.full_train_df["author"].map(Config.LABEL2ID).values
        tfidf_model.fit(X_train, y_train)

        print("Predicting Test TF-IDF...")
        test_probs_tfidf = tfidf_model.predict_proba(X_test)

        # --- 2. Transformer Expert ---
        print(">> Retraining Transformer Expert on full data...")
        transformer_trainer = Trainer()

        # To use early stopping effectively, we still need a validation split.
        # We split the full training data 90/10.
        tr_split, val_split = train_test_split(
            self.full_train_df,
            test_size=0.1,
            random_state=Config.SEED,
            stratify=self.full_train_df["author"],
        )

        transformer_trainer.fit(
            tr_split["text"],
            tr_split["author"],
            val_split["text"],
            val_split["author"],
            fold_idx="final",
        )

        print("Predicting Test Transformer...")
        test_probs_transformer = transformer_trainer.predict(self.test_df["text"])

        # --- 3. Meta-Features ---
        print("Extracting Meta-Features for Test set...")
        meta_extractor = MetaFeatureExtractor()
        test_meta_features = meta_extractor.get_features(self.test_df, "test_final")

        # --- 4. Final Stacking & Prediction ---
        print("Generating Final Predictions via Meta-Learner...")
        X_test_stack = self.meta_learner.prepare_meta_features(
            [test_probs_tfidf, test_probs_transformer], test_meta_features
        )

        final_probs = self.meta_learner.predict(X_test_stack)

        # --- 5. Save Submission ---
        save_submission(self.test_df["id"].values, final_probs)

    def execute(self):
        """
        Main entry point.
        """
        self.run_cv_and_train_meta()
        self.retrain_l1_and_predict_test()
