import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config, set_seed
from library.utils import setup_logger, load_object
from library.trainer import run_training
from library.inference import generate_submission
from library.feature_extraction import FeaturePreprocessor


def main():
    # 1. Setup
    # Ensure reproducibility
    set_seed(Config.SEED)
    logger = setup_logger("runfile")

    logger.info("Starting Asymmetric Dual-Backbone Consensus (ADBC) Pipeline")

    # 2. Training Phase
    # Trains models using 5-fold CV on the training split (metadata/train.csv)
    # Returns the paths to the 5 saved fold models
    logger.info(">>> Phase 1: Training")
    model_paths = run_training(debug=False)

    # 3. Validation Phase
    # Evaluate on the hold-out validation split (metadata/val.csv)
    logger.info(">>> Phase 2: Validation")

    # Initialize preprocessor to load validation data
    preprocessor = FeaturePreprocessor()
    val_data = preprocessor.get_data(split="val", load_cached=True, debug=False)

    X_val = val_data["X"]
    y_val = val_data["y"]

    # Perform Inference using the ensemble of trained models
    # We average the probabilities from all folds (CV-Bagging)
    avg_probs = np.zeros(len(y_val))

    for path in model_paths:
        logger.info(f"Inference with model: {os.path.basename(path)}")
        model = load_object(path)
        # Predict probability for class 1 (Success)
        probs = model.predict_proba(X_val)[:, 1]
        avg_probs += probs

    # Average predictions
    avg_probs /= len(model_paths)

    # Calculate Final Metric
    final_auc = roc_auc_score(y_val, avg_probs)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_auc}")

    # 4. Failure Analysis
    logger.info(">>> Phase 3: Failure Analysis")

    # Calculate Error Magnitude
    errors = np.abs(y_val - avg_probs)

    # Extract Metadata features for correlation analysis
    # The feature matrix X is [Primary | Aux | Meta]
    # We use the slice defined in the data dictionary to get just the metadata
    feature_slices = val_data["feature_slices"]
    meta_slice = feature_slices["meta"]
    X_meta = X_val[:, meta_slice]

    # Create a DataFrame for easy correlation calculation
    # We use the column names from Config
    df_analysis = pd.DataFrame(X_meta, columns=Config.METADATA_COLS)
    df_analysis["error_magnitude"] = errors

    # Calculate correlations
    correlations = df_analysis.corr()["error_magnitude"].drop("error_magnitude")

    logger.info("Correlation between Metadata Features and Error Magnitude:")
    print(correlations.sort_values(ascending=False))

    # 5. Submission Phase
    logger.info(">>> Phase 4: Submission")

    THRESHOLD = 0.7160806860575912

    if final_auc > THRESHOLD:
        logger.info(
            f"Validation AUC ({final_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        # Generate submission for the test set
        generate_submission(model_paths=model_paths, load_cached_data=True, debug=False)
    else:
        logger.warning(
            f"Validation AUC ({final_auc}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
