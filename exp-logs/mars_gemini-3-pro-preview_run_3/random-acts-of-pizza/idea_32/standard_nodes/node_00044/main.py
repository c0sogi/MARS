import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from library.config import (
    TARGET_COL,
    METADATA_FEATURES,
    SEED,
)
from library.utils import (
    set_seed,
    print_header,
    print_info,
    print_metric,
    Timer,
)
from library.features import FeatureFactory
from library.ensemble import StackingPipeline


def perform_failure_analysis(X_meta_val, y_val, y_pred, feature_names):
    """
    Analyzes the correlation between model error and metadata features.
    """
    print_header("Failure Analysis")

    # Calculate absolute error
    errors = np.abs(y_val - y_pred)

    # Create a DataFrame for analysis
    analysis_df = pd.DataFrame(X_meta_val, columns=feature_names)
    analysis_df["error"] = errors

    print_info("Correlation between Error and Metadata Features:")
    correlations = (
        analysis_df.corr()["error"].drop("error").sort_values(ascending=False)
    )

    for feature, corr in correlations.items():
        print(f"  {feature}: {corr:.4f}")


def main():
    # 1. Setup
    set_seed(SEED)

    with Timer("Total Runtime"):

        # 2. Data Loading & Feature Engineering
        factory = FeatureFactory()

        # Load Raw Dataframes (for targets and IDs)
        train_df, val_df, test_df = factory.load_raw_data()

        # Extract Targets
        y_train = train_df[TARGET_COL].values
        y_val = val_df[TARGET_COL].values

        # Generate Feature Views (using cache if available)
        # Returns tuples: (train, val, test)
        X_meta_tr, X_meta_val, X_meta_te = factory.create_metadata_view(
            train_df, val_df, test_df, load_cached_data=True
        )
        X_lex_tr, X_lex_val, X_lex_te = factory.create_lexical_view(
            train_df, val_df, test_df, load_cached_data=True
        )
        X_beh_tr, X_beh_val, X_beh_te = factory.create_behavioral_view(
            train_df, val_df, test_df, load_cached_data=True
        )
        X_sem_tr, X_sem_val, X_sem_te = factory.create_semantic_view(
            train_df, val_df, test_df, load_cached_data=True
        )

        # Organize into Dictionaries for the Pipeline
        X_train_dict = {
            "metadata": X_meta_tr,
            "lexical": X_lex_tr,
            "behavioral": X_beh_tr,
            "semantic": X_sem_tr,
        }

        X_val_dict = {
            "metadata": X_meta_val,
            "lexical": X_lex_val,
            "behavioral": X_beh_val,
            "semantic": X_sem_val,
        }

        X_test_dict = {
            "metadata": X_meta_te,
            "lexical": X_lex_te,
            "behavioral": X_beh_te,
            "semantic": X_sem_te,
        }

        # 3. Pipeline Execution
        pipeline = StackingPipeline()

        # Step A: Cross-Validation (OOF Generation)
        # Trains base models on 4 folds, predicts on 5th.
        oof_preds = pipeline.run_cross_validation(X_train_dict, y_train)

        # Step B: Train Meta-Learner
        # Trains Level 2 Logistic Regression on the OOF predictions
        pipeline.train_meta_learner(oof_preds, y_train)

        # Calculate Final OOF AUC (Level 2)
        # This is the valid metric for the ensemble performance
        oof_probs = pipeline.meta_learner.predict_proba(oof_preds)[:, 1]
        val_auc = roc_auc_score(y_train, oof_probs)
        print_metric("Final OOF AUC (Validation Metric)", val_auc)

        # Step C: Retrain Final Base Models
        # Uses Validation-Guided Retraining (Train+Val for RF/LR, Train w/ Val Early Stopping for XGB)
        pipeline.retrain_final_models(X_train_dict, y_train, X_val_dict, y_val)

        # 4. Final Validation
        print_header("Validation Summary")
        print_info(
            "Using OOF AUC as the validation metric to avoid leakage from retraining on Val set."
        )
        print(f"Final Validation Metric: {val_auc}")

        # 5. Failure Analysis
        # We perform failure analysis on the OOF predictions (Training set)
        perform_failure_analysis(X_meta_tr, y_train, oof_probs, METADATA_FEATURES)

        # 6. Submission Generation
        THRESHOLD = 0.7085870249842536

        if val_auc > THRESHOLD:
            print_info(
                f"Validation metric ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
            )
            test_probs = pipeline.predict(X_test_dict)
            pipeline.generate_submission(test_df, test_probs)
        else:
            print_info(
                f"Validation metric ({val_auc}) did not meet threshold ({THRESHOLD}). Skipping submission."
            )


if __name__ == "__main__":
    main()
