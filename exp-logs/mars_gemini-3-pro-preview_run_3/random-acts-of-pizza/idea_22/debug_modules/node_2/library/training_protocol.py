import pandas as pd
from library.config import Config
from library.utils import load_data
from library.feature_engineering import FeaturePipeline
from library.model_definitions import HexEnsemble


def run_training_protocol(debug=False, load_cache=True):
    """
    Executes the Hex-View Hybrid-Topology Stacking Ensemble training protocol.

    This function orchestrates:
    1. Feature Engineering (Lexical, Behavioral, Semantic, Manifold, Metadata)
    2. Data Loading (Test IDs)
    3. Model Training (Validation-Guided Retraining Protocol via HexEnsemble)
    4. Submission Generation

    Args:
        debug (bool): If True, enables debug mode (subsampled data via Config).
        load_cache (bool): If True, attempts to load pre-computed features from cache.

    Returns:
        HexEnsemble: The trained ensemble model instance.
    """
    # 1. Configuration Setup
    if debug:
        Config.DEBUG = True
        print("DEBUG Mode Enabled: Running on subsampled data.")

    # 2. Feature Engineering Pipeline
    # Generates or loads the feature dictionaries for train, val, and test splits
    print("Starting Feature Pipeline...")
    pipeline = FeaturePipeline(load_cached_data=load_cache)
    data = pipeline.run()

    # 3. Load Test IDs
    # The feature pipeline returns matrices; we need the original IDs for the submission file.
    print("Loading Test IDs for submission...")
    df_test = load_data("test")
    test_ids = df_test[Config.ID_COL].values

    # 4. Model Initialization
    print("Initializing HexEnsemble...")
    model = HexEnsemble()

    # 5. Training Protocol
    # The fit method implements the Validation-Guided Retraining:
    # - Phase 1: 5-Fold CV to generate OOF predictions.
    # - Phase 2: Train Meta-Learner on OOF predictions.
    # - Phase 3: Retrain Base Learners on Full Data (Train + Val).
    print("Executing Validation-Guided Retraining Protocol...")
    model.fit(data["train"], data["val"])

    # 6. Submission Generation
    # Generates predictions on the test set and saves to ./submission/submission.csv
    print("Generating final submission...")
    model.generate_submission(data["test"], test_ids)

    print("Protocol execution complete.")
    return model
