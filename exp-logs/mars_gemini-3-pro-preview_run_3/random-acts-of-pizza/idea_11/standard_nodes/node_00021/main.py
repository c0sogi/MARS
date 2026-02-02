import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.metrics import roc_auc_score

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

# Import provided library modules
from library.config import Config
from library.utils import load_data
from library.feature_extraction import FeatureManager
from library.ensemble_trainer import StackingEnsemble


def setup_environment():
    """Sets up the environment: warnings, seeds, and device configuration."""
    warnings.filterwarnings("ignore")

    # Set seeds
    Config.set_seed(Config.RANDOM_SEED)

    # Override Config for Fast Baseline & GPU
    # Reduce estimators for speed while maintaining reasonable performance
    Config.RF_PARAMS["n_estimators"] = 100
    Config.XGB_PARAMS["n_estimators"] = 500

    # Enable GPU for XGBoost if available
    if torch.cuda.is_available():
        print("GPU detected. Configuring XGBoost for GPU acceleration.")
        # Modern XGBoost uses 'device' parameter, older versions use 'tree_method'='gpu_hist'
        # We'll set both to be safe/compatible with the installed version
        Config.XGB_PARAMS["device"] = "cuda"
        Config.XGB_PARAMS["tree_method"] = "hist"
    else:
        print("No GPU detected. Using CPU.")


def perform_failure_analysis(val_df, y_val, y_pred):
    """
    Analyzes model failures by correlating error with features.
    """
    print("\nPerforming Failure Analysis on Validation Set...")

    # Calculate Error
    errors = np.abs(y_val - y_pred)

    # Select numerical columns for correlation
    numerical_cols = val_df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target and ID if present
    cols_to_exclude = [Config.TARGET_COL, Config.ID_COL, "requester_received_pizza"]
    numerical_cols = [c for c in numerical_cols if c not in cols_to_exclude]

    correlations = {}
    for col in numerical_cols:
        # Handle potential NaNs in features
        feat_values = val_df[col].fillna(val_df[col].median())
        if len(feat_values.unique()) > 1:
            corr = np.corrcoef(errors, feat_values)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for feat, corr in sorted_corr[:5]:
        print(f"  {feat}: {corr:.4f}")


def main():
    setup_environment()

    # 1. Load Data
    # We load cached data if available to save time
    train_df, val_df, test_df = load_data(load_cached_data=True, debug=False)

    # 2. Extract Features
    # FeatureManager handles caching internally
    fm = FeatureManager()
    data_dict = fm.extract_features(train_df, val_df, test_df, load_cached_data=True)

    # 3. Prepare Data for Training (Strict Hold-out Strategy)
    # The StackingEnsemble.fit method concatenates 'train' and 'val' keys.
    # To preserve the hold-out validation set, we pass empty arrays for the 'val' keys
    # in the dictionary passed to fit().

    print("Preparing data for training (hiding validation set)...")

    # Create empty placeholders matching the shapes/types
    empty_lexical = sparse.csr_matrix((0, data_dict["X_train_lexical"].shape[1]))
    empty_behavioral = sparse.csr_matrix((0, data_dict["X_train_behavioral"].shape[1]))
    empty_dense = np.zeros((0, data_dict["X_train_dense"].shape[1]))
    empty_y = np.zeros((0,))

    train_only_dict = {
        "X_train_lexical": data_dict["X_train_lexical"],
        "X_train_behavioral": data_dict["X_train_behavioral"],
        "X_train_dense": data_dict["X_train_dense"],
        "y_train": data_dict["y_train"],
        # Empty validation slots so _concat_data effectively uses only train data
        "X_val_lexical": empty_lexical,
        "X_val_behavioral": empty_behavioral,
        "X_val_dense": empty_dense,
        "y_val": empty_y,
    }

    # 4. Train Model
    ensemble = StackingEnsemble()
    ensemble.fit(train_only_dict)

    # 5. Validation Inference
    print("\nRunning inference on hold-out validation set...")

    # Map validation features to the 'test' keys expected by ensemble.predict()
    val_as_test_dict = {
        "X_test_lexical": data_dict["X_val_lexical"],
        "X_test_behavioral": data_dict["X_val_behavioral"],
        "X_test_dense": data_dict["X_val_dense"],
    }

    val_preds = ensemble.predict(val_as_test_dict)
    y_val = data_dict["y_val"]

    # 6. Evaluation
    val_auc = roc_auc_score(y_val, val_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 7. Failure Analysis
    perform_failure_analysis(val_df, y_val, val_preds)

    # 8. Submission
    threshold = 0.6913548345419015
    if val_auc > threshold:
        print(
            f"\nValidation score ({val_auc}) exceeds threshold ({threshold}). Generating submission..."
        )
        test_preds = ensemble.predict(data_dict)
        ensemble.save_submission(test_preds)
    else:
        print(
            f"\nValidation score ({val_auc}) did not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
