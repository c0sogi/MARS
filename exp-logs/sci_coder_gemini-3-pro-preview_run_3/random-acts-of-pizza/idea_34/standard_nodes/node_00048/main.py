import os
import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Add current directory to path to ensure imports work correctly
sys.path.append(os.getcwd())

# Import provided library modules
from library import config
from library import utils
from library import data_processing
from library import feature_engineering
from library import ensemble_pipeline
from library.models import ModelFactory


def main():
    # 1. Setup
    utils.set_seed(config.RANDOM_STATE)
    warnings.filterwarnings("ignore")

    # 2. Data Loading
    # Load and process raw data (handles text cleaning, metadata scaling, etc.)
    train_df, val_df, test_df = data_processing.load_and_process_data(
        load_cached_data=True
    )

    # 3. Feature Engineering
    # Generate Sparse (TF-IDF), Dense (Embeddings), and Metadata features
    fe = feature_engineering.FeatureEngineer()
    train_feats, val_feats, test_feats = fe.generate_features(
        train_df, val_df, test_df, load_cached_data=True
    )

    # 4. Initialize Ensemble
    ensemble = ensemble_pipeline.StackingEnsemble(train_feats, val_feats, test_feats)

    # 5. Train CV (Level 1 OOF + Level 2 Meta-Learner Training)
    # This trains the Level 2 Meta-Learner on OOF predictions from the training set.
    ensemble.train_cv()

    # 6. Validation on Hold-out Set
    # We need to evaluate the architecture on the hold-out validation set.
    # We train base learners on the Training set ONLY and predict on the Validation set.
    # Note: We do not use ensemble.train_final_models() here because that method
    # merges Train and Val for the Random Forest models, which would be data leakage for validation.

    utils.print_header("Performing Hold-out Validation")
    n_val = val_feats["metadata"].shape[0]
    n_models = len(ensemble.learners_config)
    val_level1_preds = np.zeros((n_val, n_models))

    for i, (name, factory_func, key) in enumerate(ensemble.learners_config):
        # Instantiate a fresh model
        model = factory_func()

        # Get training data
        X_train = train_feats[key]
        y_train = train_feats["y"]

        # Fit on Training Data
        # For XGBoost (Semantic Booster), we skip early stopping here to strictly adhere
        # to the hold-out principle (not using Val for model selection), or we could split Train.
        # Given the baseline nature, fitting on full Train is appropriate.
        model.fit(X_train, y_train)

        # Predict on Validation Data
        X_val = val_feats[key]
        preds = model.predict_proba(X_val)[:, 1]
        val_level1_preds[:, i] = preds

    # Predict with the Meta-Learner (which was trained in train_cv)
    # We feed it the Level 1 predictions from the validation set
    val_final_probs = ensemble.meta_learner.predict_proba(val_level1_preds)[:, 1]

    # Calculate Final Validation Metric
    y_val = val_feats["y"]
    val_auc = roc_auc_score(y_val, val_final_probs)

    # PRINT REQUIRED METRIC (Full Precision)
    print(f"Final Validation Metric: {val_auc}")

    # 7. Failure Analysis
    utils.print_header("Failure Analysis")
    # Calculate absolute error
    errors = np.abs(y_val - val_final_probs)

    # Correlate errors with Metadata features to find systematic weaknesses
    meta_cols = config.METADATA_COLS
    X_val_meta = val_feats["metadata"]

    correlations = []
    # Iterate through metadata columns
    for idx, col_name in enumerate(meta_cols):
        # Extract feature column
        # Note: X_val_meta is a dense numpy array at this point
        feat_vals = X_val_meta[:, idx]

        # Calculate Pearson correlation
        # Check for zero variance to avoid warnings
        if np.std(feat_vals) == 0:
            corr = 0.0
        else:
            corr, _ = pearsonr(errors, feat_vals)

        correlations.append((col_name, corr))

    # Sort by magnitude of correlation (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top Feature Correlations with Prediction Error:")
    for name, corr in correlations:
        print(f"  {name}: {corr:.4f}")

    # 8. Submission
    threshold = 0.7138293787137718

    if val_auc > threshold:
        utils.print_header("Generating Submission")
        # Now we execute the final retraining pipeline which maximizes data usage
        # (Retrains RFs on Train + Val, uses Val for XGB early stopping)
        ensemble.train_final_models()
        ensemble.predict()
    else:
        print(
            f"\nValidation metric {val_auc} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
