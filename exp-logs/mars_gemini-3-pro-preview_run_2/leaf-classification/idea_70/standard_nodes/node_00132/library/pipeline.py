import os
import numpy as np
import pandas as pd
from library.utils import set_seed
from library.preprocessing import Preprocessor
from library.models import GreedyEnsembleSelector


def run_smpge_pipeline(load_cached_data=True, max_ensemble_size=30, tolerance=1e-6):
    """
    Executes the Stratified-Manifold Precision-Generative Ensemble (SMPGE) pipeline.

    Args:
        load_cached_data (bool): If True, attempts to load preprocessed data from cache.
        max_ensemble_size (int): Maximum number of experts to select in the ensemble.
        tolerance (float): Minimum improvement in log loss required to add an expert.

    Returns:
        pd.DataFrame: The generated submission dataframe.
    """
    # 1. Setup
    set_seed(42)
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    print("==================================================")
    print("Starting SMPGE Pipeline")
    print("==================================================")

    # 2. Data Loading and Preprocessing
    # The Preprocessor handles loading raw data, feature extraction,
    # manifold transformations, and caching.
    print("\n[Step 1/4] Initializing Data Preprocessing...")
    preprocessor = Preprocessor()
    data = preprocessor.get_data(load_cached_data=load_cached_data)

    # Extract class labels for submission header (sklearn sorts classes alphabetically)
    # y_train contains the species strings
    classes = np.unique(data["y_train"])
    print(f"Data loaded. Number of classes: {len(classes)}")
    print(f"Training samples: {len(data['y_train'])}")
    print(f"Validation samples: {len(data['y_val'])}")
    print(f"Test samples: {len(data['test_ids'])}")

    # 3. Model Training and Selection
    # Initialize the selector which manages the expert library
    print("\n[Step 2/4] Initializing Greedy Ensemble Selector...")
    selector = GreedyEnsembleSelector(
        max_ensemble_size=max_ensemble_size, tolerance=tolerance
    )

    # Fit the selector:
    # - Phase 1: Train all candidates on Train, Select on Val
    # - Phase 2: Retrain selected experts on Train + Val (via refit)
    print("Running Selection Phase...")
    selector.fit(data)
    print("Running Retraining Phase...")
    selector.refit(data)

    # 4. Inference
    print("\n[Step 3/4] Generating Test Predictions...")
    # Predict using the retrained ensemble on the test set views
    test_probs = selector.predict(data)

    # 5. Submission Generation
    print("\n[Step 4/4] Formatting Submission...")

    # Verify shape
    if test_probs.shape[1] != len(classes):
        raise ValueError(
            f"Output shape {test_probs.shape} does not match number of classes {len(classes)}"
        )

    # Create DataFrame
    # Columns must be the species names
    submission = pd.DataFrame(test_probs, columns=classes)

    # Insert ID column at the beginning
    submission.insert(0, "id", data["test_ids"])

    # Save to file
    submission_path = os.path.join(submission_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"Submission saved successfully to: {submission_path}")
    print("Pipeline Complete.")

    return submission
