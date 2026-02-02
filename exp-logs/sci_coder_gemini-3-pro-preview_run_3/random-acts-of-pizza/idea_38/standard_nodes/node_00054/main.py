import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from library.config import SEED, METADATA_COLS
from library.utils import set_seed, get_logger
from library.trainer import Trainer

logger = get_logger("runfile")


def main():
    # 1. Setup & Configuration
    set_seed(SEED)

    # 2. Model Training
    # Initialize the Trainer which encapsulates the entire pipeline
    trainer = Trainer()

    # Execute training
    # load_cached_data=True ensures we use pre-computed features from ./working if available,
    # satisfying the requirement for a fast baseline execution.
    logger.info("Starting training pipeline...")
    model, data_dict = trainer.train(load_cached_data=True)

    # 3. Validation Assessment
    logger.info("Evaluating model on validation set...")

    # Retrieve validation targets
    y_val = data_dict["y_val"]

    # Generate predictions for the validation split
    # The StackingEnsemble internally assembles the required feature views
    val_probs = model.predict_proba(data_dict, split="val")

    # Compute Metric
    val_auc = roc_auc_score(y_val, val_probs)

    # Print the required metric string with full precision
    print(f"Final Validation Metric: {val_auc}")

    # 4. Failure Analysis
    logger.info("Performing failure analysis...")

    # Calculate absolute error for each validation sample
    errors = np.abs(y_val - val_probs)

    # Correlate errors with metadata features to find systematic weaknesses
    # Reconstruct the feature names based on FeatureFactory logic:
    # It uses METADATA_COLS from config and appends 'community_generosity_score'
    feature_names = METADATA_COLS + ["community_generosity_score"]
    X_val_meta = data_dict["X_val_metadata"]

    print("Correlation between Error and Metadata Features:")

    # Ensure dimensions match before iterating
    if X_val_meta.shape[1] == len(feature_names):
        for i, name in enumerate(feature_names):
            feat_values = X_val_meta[:, i]
            # Calculate Pearson correlation, handling potential constant columns
            if np.std(feat_values) > 1e-9:
                corr, _ = pearsonr(errors, feat_values)
                print(f"  {name}: {corr:.4f}")
            else:
                print(f"  {name}: NaN (Constant Feature)")
    else:
        # Fallback if dimensions mismatch (e.g. config changed)
        logger.warning("Feature name mismatch. Printing generic indices.")
        for i in range(X_val_meta.shape[1]):
            feat_values = X_val_meta[:, i]
            if np.std(feat_values) > 1e-9:
                corr, _ = pearsonr(errors, feat_values)
                print(f"  Feature_{i}: {corr:.4f}")

    # 5. Submission Generation
    # Check against the specific threshold required by the task
    THRESHOLD = 0.7138293787137718

    if val_auc > THRESHOLD:
        logger.info(
            f"Validation AUC ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission(data_dict)
    else:
        logger.warning(
            f"Validation AUC ({val_auc}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
