import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, save_submission, timer
from library.data_loader import load_dataset
from library.feature_engineering import FeaturePipeline
from library.models import HeptViewEnsemble


class Trainer:
    def __init__(self):
        """
        Initializes the Trainer with FeaturePipeline and HeptViewEnsemble.
        """
        set_seed(Config.SEED)
        self.pipeline = FeaturePipeline()
        self.ensemble = HeptViewEnsemble()

    def run(self, load_cached_data=True, limit=None):
        """
        Executes the full training and prediction pipeline.

        Args:
            load_cached_data (bool): Whether to load data/features from cache.
            limit (int, optional): Limit the number of samples for debugging.
        """
        print(f"Starting Trainer Run (Limit={limit})...")

        # =========================================================================
        # 1. Data Loading
        # =========================================================================
        with timer("Data Loading"):
            # Load raw data dictionaries (ids, metadata, text, community, y)
            train_raw = load_dataset(
                "train", load_cached_data=load_cached_data, limit=limit
            )
            val_raw = load_dataset(
                "val", load_cached_data=load_cached_data, limit=limit
            )
            test_raw = load_dataset(
                "test", load_cached_data=load_cached_data, limit=limit
            )

        # =========================================================================
        # 2. Feature Engineering
        # =========================================================================
        with timer("Feature Engineering"):
            # Fit pipeline on training data
            self.pipeline.fit(train_raw)

            # Transform all splits into feature views (Lexical, Behavioral, Semantic, Metadata)
            train_features = self.pipeline.transform(
                train_raw, "train", load_cached_data=load_cached_data
            )
            val_features = self.pipeline.transform(
                val_raw, "val", load_cached_data=load_cached_data
            )
            test_features = self.pipeline.transform(
                test_raw, "test", load_cached_data=load_cached_data
            )

        # Extract targets
        y_train = train_raw["y"]
        y_val = val_raw["y"]

        # =========================================================================
        # 3. Level 1: Out-of-Fold (OOF) Training
        # =========================================================================
        with timer("Level 1 OOF Training"):
            # Generates OOF predictions using 5-Fold CV on the training set
            oof_preds = self.ensemble.train_oof(train_features, y_train)

        # =========================================================================
        # 4. Level 2: Meta-Learner Training
        # =========================================================================
        with timer("Level 2 Meta-Learner Training"):
            # Trains the Logistic Regression meta-learner on OOF predictions
            self.ensemble.train_meta(oof_preds, y_train)

        # =========================================================================
        # 5. Final Retraining (Validation-Guided)
        # =========================================================================
        with timer("Final Retraining"):
            # Retrains base learners:
            # - RF/Linear: Train on (Train + Val)
            # - XGB/LGBM: Train on Train, Early Stop on Val
            self.ensemble.train_final(train_features, y_train, val_features, y_val)

        # =========================================================================
        # 6. Prediction and Submission
        # =========================================================================
        with timer("Prediction and Submission"):
            # Generate final probabilities for the test set
            final_predictions = self.ensemble.predict(test_features)

            # Save submission file
            save_submission(final_predictions, test_raw["ids"])

        print("Trainer run completed successfully.")
