"""
Implementation of the Modality-Balanced Bagged Linear Ensemble execution script.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import PizzaDataLoader
from library.feature_engineering import TextEmbedder, TabularProcessor, FeatureFuser
from library.model_factory import ModelFactory


def main():
    # 1. Setup
    logger = setup_logger("runfile")
    set_seed(Config.SEED)

    logger.info("Initializing Modality-Balanced Bagged Linear Ensemble Pipeline...")

    # 2. Load Data
    # We load cached data if available to speed up execution
    data_loader = PizzaDataLoader()
    train_df, val_df, test_df = data_loader.load_data(load_cached_data=True)

    # 3. Feature Engineering
    # 3.1 Text Embeddings
    text_embedder = TextEmbedder()
    # These methods handle caching internally
    X_train_text = text_embedder.get_embeddings(train_df, "train")
    X_val_text = text_embedder.get_embeddings(val_df, "val")
    X_test_text = text_embedder.get_embeddings(test_df, "test")

    # 3.2 Numeric Metadata (RankGauss Transformed)
    tabular_processor = TabularProcessor()
    X_train_tab, X_val_tab, X_test_tab = tabular_processor.process_numeric_features(
        train_df, val_df, test_df
    )

    # 3.3 Targets
    y_train = train_df["requester_received_pizza"].values.astype(int)
    y_val = val_df["requester_received_pizza"].values.astype(int)

    # 4. Hyperparameter Tuning (Grid Search)
    # We search for the best combination of alpha (modality balance), C (regularization), and class_weight
    # using the fixed Train/Val split.

    best_auc = -1.0
    best_model = None
    best_params = {}

    alphas = Config.GRID_SEARCH_PARAMS["alpha"]
    Cs = Config.GRID_SEARCH_PARAMS["C"]
    class_weights = Config.GRID_SEARCH_PARAMS["class_weight"]

    logger.info(
        f"Starting Grid Search over {len(alphas)*len(Cs)*len(class_weights)} combinations..."
    )

    for alpha in alphas:
        # Fuse features with differential scaling
        # Alpha scales the metadata relative to the unit-norm text embeddings
        X_train_fused = FeatureFuser.fuse(X_train_text, X_train_tab, alpha)
        X_val_fused = FeatureFuser.fuse(X_val_text, X_val_tab, alpha)

        for C in Cs:
            for cw in class_weights:
                # Train Bagged Ensemble
                # Bagging stabilizes the high-variance potential of the fused space
                model = ModelFactory.create_bagged_ensemble(
                    C=C,
                    class_weight=cw,
                    n_estimators=Config.BAGGING_N_ESTIMATORS,
                    random_state=Config.SEED,
                )

                model.fit(X_train_fused, y_train)

                # Evaluate on Hold-out Validation Set
                val_preds = model.predict_proba(X_val_fused)[:, 1]
                auc = roc_auc_score(y_val, val_preds)

                if auc > best_auc:
                    best_auc = auc
                    best_model = model
                    best_params = {"alpha": alpha, "C": C, "class_weight": cw}

                    # Optional: Log progress for significant improvements
                    # logger.info(f"New Best AUC: {best_auc:.4f} with params {best_params}")

    # 5. Reporting
    logger.info("Grid Search Complete.")
    print(f"Final Validation Metric: {best_auc}")
    logger.info(f"Best Parameters: {best_params}")

    # 6. Failure Analysis
    logger.info("Performing Failure Analysis on Validation Set...")

    # Re-generate fused features for best alpha
    X_val_fused_best = FeatureFuser.fuse(X_val_text, X_val_tab, best_params["alpha"])
    best_val_preds = best_model.predict_proba(X_val_fused_best)[:, 1]

    # Calculate absolute error
    errors = np.abs(y_val - best_val_preds)

    # Correlate error with raw numeric features to identify weak points
    # We use the raw val_df for interpretability
    analysis_df = val_df[Config.NUMERIC_FEATURES].copy()

    # Handle NaNs in raw data for correlation calculation (simple fill)
    analysis_df = analysis_df.fillna(analysis_df.median())
    analysis_df["error"] = errors

    # Compute correlation
    correlations = (
        analysis_df.corr()["error"].drop("error").sort_values(ascending=False, key=abs)
    )

    print("Correlation between Error and Features:")
    print(correlations)

    # 7. Submission Generation
    # Threshold check
    SUBMISSION_THRESHOLD = 0.7141749705260098

    if best_auc > SUBMISSION_THRESHOLD:
        logger.info(
            f"Validation AUC ({best_auc}) exceeds threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )

        # Fuse test features
        X_test_fused = FeatureFuser.fuse(X_test_text, X_test_tab, best_params["alpha"])

        # Predict
        test_preds = best_model.predict_proba(X_test_fused)[:, 1]

        # Save
        submission_df = pd.DataFrame(
            {
                "request_id": test_df["request_id"],
                "requester_received_pizza": test_preds,
            }
        )

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        logger.warning(
            f"Validation AUC ({best_auc}) did not exceed threshold ({SUBMISSION_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
