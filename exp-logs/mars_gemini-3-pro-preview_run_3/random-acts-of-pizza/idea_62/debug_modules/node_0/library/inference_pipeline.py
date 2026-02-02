import pandas as pd
from library.config import Config
from library.utils import setup_logger, set_seed
from library.data_loader import load_union_dataset
from library.feature_engineering import FeatureEngineeringPipeline
from library.meta_learner import Level2MetaLearner

logger = setup_logger("inference_pipeline")


class InferencePipeline:
    """
    Orchestrates the inference phase of the Conservative Granular Hept-View Stacking Ensemble.

    This pipeline:
    1. Loads the Union Dataset (Train + Val) and Test Dataset.
    2. Runs the Feature Engineering Pipeline to generate/load feature matrices.
       (Note: Training data is required to fit vectorizers if cache is missing).
    3. Invokes the Level 2 Meta Learner to generate predictions using the
       Consistent Hybrid Inference protocol (CV-Bagging for volatile, Single Model for stable).
    4. Saves the submission file.
    """

    def __init__(self):
        self.random_state = Config.RANDOM_STATE

    def run(self, load_cached_data: bool = True, debug_size: int = None):
        """
        Executes the inference pipeline.

        Args:
            load_cached_data (bool): Whether to attempt loading features/data from cache.
            debug_size (int, optional): If provided, limits dataset size for debugging.

        Returns:
            pd.DataFrame: The generated submission dataframe.
        """
        set_seed(self.random_state)
        logger.info("Starting Inference Pipeline...")

        # 1. Load Data
        # We load the union dataset as well because the FeatureEngineeringPipeline
        # requires it to fit vectorizers and scalers if the features are not cached.
        logger.info("Loading datasets...")
        train_df, test_df = load_union_dataset(
            load_cached_data=load_cached_data, debug_size=debug_size
        )

        # 2. Feature Engineering
        # Generates (or loads) the 4 views: Lexical, Community, Semantic, Metadata
        logger.info("Running Feature Engineering Pipeline...")
        fe_pipeline = FeatureEngineeringPipeline()
        features_dict = fe_pipeline.run(
            train_df, test_df, load_cached_data=load_cached_data
        )

        # 3. Prepare Test IDs
        # Ensure we have the IDs corresponding to the test rows
        if Config.ID_COL not in test_df.columns:
            raise ValueError(f"Test dataframe missing ID column: {Config.ID_COL}")
        test_ids = test_df[Config.ID_COL].values

        # 4. Generate Submission
        # The MetaLearner handles the logic of loading the correct model types
        # (Volatile vs Stable) and stacking them.
        logger.info("Generating predictions via Level 2 Meta Learner...")
        meta_learner = Level2MetaLearner()
        submission_df = meta_learner.generate_submission(features_dict, test_ids)

        logger.info("Inference Pipeline Completed Successfully.")
        return submission_df
