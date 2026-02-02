import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import roc_auc_score
import torch

from library.config import Config
from library.feature_engineering import get_features
from library.models_rf import RFModelWrapper
from library.models_mlp import MLPModelWrapper


def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run():
    # 1. Configuration for Fast Baseline
    # Reduce epochs and estimators to ensure completion within time limits
    Config.MLP_PARAMS["epochs"] = 10
    Config.RF_PARAMS["n_estimators"] = 100

    set_seeds(Config.RANDOM_STATE)

    # 2. Load Data and Features
    print("Loading features...")
    # load_cached_data=True ensures we use pre-computed features if available
    rf_data, mlp_data, labels = get_features(load_cached_data=True)
    y_val = labels["y_val"]

    # 3. Stream A: Random Forest
    print("Training Random Forest...")
    rf_model = RFModelWrapper()
    rf_model.train(rf_data, labels)

    # Generate RF Validation Predictions
    # We manually construct the validation input matrix because the wrapper's predict method
    # is designed for the test set keys.
    X_val_rf = hstack([rf_data["val_tfidf"], rf_data["val_meta"]]).tocsr()
    rf_val_preds = rf_model.model.predict_proba(X_val_rf)[:, 1]

    # 4. Stream B: MLP
    print("Training MLP...")
    mlp_model = MLPModelWrapper()
    mlp_model.train(mlp_data, labels)

    # Generate MLP Validation Predictions
    # The MLP wrapper's predict method expects 'test_*' keys.
    # We create a temporary dictionary mapping validation data to these keys.
    mlp_val_input = {
        "test_title": mlp_data["val_title"],
        "test_body": mlp_data["val_body"],
        "test_history": mlp_data["val_history"],
        "test_meta": mlp_data["val_meta"],
    }
    mlp_val_preds = mlp_model.predict(mlp_val_input)

    # 5. Ensemble Evaluation
    print("Evaluating Ensemble...")
    weights = Config.ENSEMBLE_WEIGHTS
    w_rf = weights["rf"]
    w_mlp = weights["mlp"]

    # Normalize weights
    total_weight = w_rf + w_mlp
    w_rf /= total_weight
    w_mlp /= total_weight

    # Weighted Average
    val_preds_ensemble = (w_rf * rf_val_preds) + (w_mlp * mlp_val_preds)

    # Calculate Metric
    val_auc = roc_auc_score(y_val, val_preds_ensemble)
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("Performing Failure Analysis...")
    # Calculate error magnitude
    errors = np.abs(y_val - val_preds_ensemble)

    # Define feature names for correlation analysis
    # Order must match Feature Engineering: Numeric -> Sentiment -> Ratios
    ratio_cols = ["upvote_ratio", "comment_post_ratio", "raop_activity_ratio"]
    feature_names = Config.NUMERIC_COLS + Config.SENTIMENT_COLS + ratio_cols

    # Use unscaled metadata from RF data for interpretability
    val_meta = rf_data["val_meta"]

    correlations = []
    for i, name in enumerate(feature_names):
        if i < val_meta.shape[1]:
            feat_values = val_meta[:, i]
            # Avoid correlation with constant features
            if np.std(feat_values) > 1e-9:
                corr = np.corrcoef(errors, feat_values)[0, 1]
                correlations.append((name, corr))
            else:
                correlations.append((name, 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with prediction error:")
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.10f}")

    # 7. Submission
    threshold = 0.7036289345758168

    if val_auc > threshold:
        print("Validation metric meets threshold. Generating submission...")

        # Generate Test Predictions
        rf_test_preds = rf_model.predict(rf_data)
        mlp_test_preds = mlp_model.predict(mlp_data)

        # Ensemble Test Predictions
        final_test_preds = (w_rf * rf_test_preds) + (w_mlp * mlp_test_preds)

        # Load Test IDs
        test_df = pd.read_csv(Config.TEST_PATH)

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {Config.ID_COL: test_df[Config.ID_COL], Config.TARGET_COL: final_test_preds}
        )

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {val_auc} does not exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    run()
