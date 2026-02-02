import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, log
from library.data_loader import get_data_splits
from library.features import FeatureEngineer
from library.ensemble import StackingEnsemble


def main():
    # 1. Setup
    set_seed(Config.RANDOM_SEED)
    log("Initializing Regularized Pent-View Stacking Pipeline...")

    # 2. Load Data
    # Loads cleaned data from cache or metadata
    train_df, y_train, val_df, y_val, test_df, test_ids = get_data_splits(
        load_cached_data=True, debug_size=Config.DEBUG_SAMPLE_SIZE
    )

    # 3. Feature Engineering
    # Generates Lexical, Behavioral, Semantic, and Contextual views
    fe = FeatureEngineer()
    feature_data = fe.generate_features(
        train_df, val_df, test_df, load_cached_data=True
    )

    train_features = feature_data["train"]
    val_features = feature_data["val"]
    test_features = feature_data["test"]

    # 4. Model Training (Level 1 CV & Level 2 Training)
    # This fits the Meta-Learner on OOF predictions from the Training set
    ensemble = StackingEnsemble()
    ensemble.fit_oof(train_features, y_train)

    # 5. Hold-out Validation
    # To get a proper validation metric, we train base models on Train and predict on Val.
    # Note: We do NOT use the models from fit_oof (which are just for meta-training).
    log("Performing inference on hold-out validation set...")

    # Container for Level 1 predictions on Validation set
    val_meta_features = pd.DataFrame(
        index=np.arange(len(y_val)), columns=ensemble.base_models.keys()
    )

    for name, model in ensemble.base_models.items():
        # Retrieve specific feature view for this model
        X_train_view = ensemble._get_feature_view(name, train_features)
        X_val_view = ensemble._get_feature_view(name, val_features)

        # Clone to ensure a fresh model
        clf = clone(model)

        # Train
        if name == "SemanticBooster":
            # XGBoost with Early Stopping using Validation set
            fit_params = Config.SEMANTIC_XGB_FIT_PARAMS.copy()
            fit_params["eval_set"] = [(X_val_view, y_val)]
            clf.fit(X_train_view, y_train, **fit_params)
        else:
            # Standard fit for RF / LR
            clf.fit(X_train_view, y_train)

        # Predict
        # Note: We use the second column [:, 1] for probability of positive class
        val_meta_features[name] = clf.predict_proba(X_val_view)[:, 1]

    # Generate Final Predictions using the pre-trained Meta-Learner
    val_final_probs = ensemble.meta_model.predict_proba(val_meta_features)[:, 1]

    # Compute Metric
    val_auc = roc_auc_score(y_val, val_final_probs)
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    log("Conducting failure analysis on validation set...")

    # Calculate absolute error
    errors = np.abs(y_val - val_final_probs)

    # Prepare analysis dataframe with numerical features
    analysis_df = val_df[Config.NUMERICAL_COLS].copy()

    # Fill NaNs for correlation calculation
    analysis_df = analysis_df.fillna(analysis_df.median())
    analysis_df["error_magnitude"] = errors.values

    # Compute correlations
    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")
    sorted_corr = correlations.abs().sort_values(ascending=False)

    print("Top 5 features correlated with prediction error:")
    print(sorted_corr.head(5))

    # 7. Submission
    # Threshold defined in requirements
    SUBMISSION_THRESHOLD = 0.7085870249842536

    if val_auc > SUBMISSION_THRESHOLD:
        log(
            f"Validation AUC ({val_auc}) exceeds threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )

        # Retrain on full data (Train + Val) and predict on Test
        # This method handles the specific retraining logic (e.g. XGB uses Val for ES)
        ensemble.retrain_and_predict(
            train_features, y_train, val_features, y_val, test_features, test_ids
        )
        log("Submission generation complete.")
    else:
        log(
            f"Validation AUC ({val_auc}) did not exceed threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
