import os
from library.config import SEED
from library.utils import get_logger, timer, set_seed
from library.feature_factory import FeatureFactory
from library.models import StackingEnsemble

logger = get_logger("trainer")


class Trainer:
    """
    Orchestrates the training and submission pipeline.
    Delegates core logic to FeatureFactory and StackingEnsemble as defined in the library.
    """

    def __init__(self):
        self.feature_factory = FeatureFactory()
        self.model = StackingEnsemble()

    def train(self, load_cached_data=True):
        """
        Runs the feature generation and model training process.

        Args:
            load_cached_data (bool): If True, attempts to load features from cache.
                                     If False or cache miss, regenerates features.

        Returns:
            tuple: (trained_model, data_dict)
        """
        set_seed(SEED)

        # 1. Feature Generation
        # FeatureFactory handles caching, preprocessing, and view generation.
        with timer("Feature Generation"):
            data_dict = self.feature_factory.run(load_cached_data=load_cached_data)

        # 2. Model Training
        # StackingEnsemble.fit() implements:
        # - Stratified K-Fold CV for OOF predictions (Level 1)
        # - Meta-Learner training (Level 2)
        # - Final Retraining of Base Learners (Full Train/Val or Train w/ Early Stopping)
        with timer("Model Training"):
            self.model.fit(data_dict)

        return self.model, data_dict

    def generate_submission(self, data_dict):
        """
        Generates predictions for the test set and saves the submission file.
        """
        with timer("Submission Generation"):
            self.model.generate_submission(data_dict)

    def run(self, load_cached_data=True):
        """
        Executes the full pipeline: Feature Generation -> Training -> Submission.
        """
        model, data_dict = self.train(load_cached_data=load_cached_data)
        self.generate_submission(data_dict)
