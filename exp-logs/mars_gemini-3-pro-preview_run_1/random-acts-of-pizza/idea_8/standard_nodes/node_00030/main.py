import sys
import os
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import roc_auc_score

# Add current directory to path to ensure imports work correctly
sys.path.append(os.getcwd())

# Import provided library modules
from library.config import Config
from library.utils import set_seed, save_submission, load_data
from library.feature_engineering import FeaturePipeline
from library.trainers import train_random_forest, train_neural_net, predict_ensemble


def run():
    # 1. Initialization
    warnings.filterwarnings("ignore")
    set_seed(Config.SEED)

    print("Starting pipeline execution...")

    # 2. Data Processing
    # Initialize pipeline and load data (using cache if available)
    pipeline = FeaturePipeline()
    data = pipeline.process_data(load_cached_data=False)

    # Extract data components for clarity
    # Stream A: Random Forest Data
    X_train_rf = data["rf"]["X_train"]
    X_val_rf = data["rf"]["X_val"]
    X_test_rf = data["rf"]["X_test"]

    # Stream B: MLP Data
    mlp_data = data["mlp"]

    # Targets and IDs
    y_train = data["y_train"]
    y_val = data["y_val"]
    test_ids = data["test_ids"]

    # 3. Model Training
    # Train Random Forest Stream
    rf_val_preds, rf_test_preds, rf_model = train_random_forest(
        X_train_rf, y_train, X_val_rf, y_val, X_test_rf
    )

    # Train Gated MLP Stream
    mlp_val_preds, mlp_test_preds, mlp_model = train_neural_net(
        mlp_data, y_train, y_val
    )

    # 4. Ensembling
    # Combine predictions with equal weights
    final_val_preds = predict_ensemble(rf_val_preds, mlp_val_preds, weights=(0.5, 0.5))
    final_test_preds = predict_ensemble(
        rf_test_preds, mlp_test_preds, weights=(0.5, 0.5)
    )

    # 5. Validation Assessment
    val_auc = roc_auc_score(y_val, final_val_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis on Validation Set...")
    # Calculate error magnitude
    errors = np.abs(y_val - final_val_preds)

    # Load original validation data to get interpretable features
    try:
        df_val = pd.read_csv(Config.VAL_DATA_PATH)

        # Select numerical columns for correlation analysis
        # We exclude ID and target columns
        exclude_cols = ["request_id", "requester_received_pizza", "source_file"]
        numeric_cols = df_val.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

        correlations = {}
        for col in numeric_cols:
            # Handle potential NaNs in features by filling with mean for correlation check
            feat_values = df_val[col].fillna(df_val[col].mean()).values
            # Ensure lengths match (just in case)
            if len(feat_values) == len(errors):
                corr = np.corrcoef(feat_values, errors)[0, 1]
                if not np.isnan(corr):
                    correlations[col] = corr

        # Sort by absolute correlation
        sorted_corr = sorted(
            correlations.items(), key=lambda x: abs(x[1]), reverse=True
        )

        print("Top 5 Features correlated with Prediction Error:")
        for name, val in sorted_corr[:5]:
            print(f"  {name}: {val:.4f}")

    except Exception as e:
        print(f"Failure analysis could not be completed: {e}")

    # 7. Submission Generation
    threshold = 0.6942941584973917
    if val_auc > threshold:
        print(
            f"\nValidation metric ({val_auc}) meets threshold ({threshold}). Generating submission..."
        )
        save_submission(
            ids=test_ids, predictions=final_test_preds, path=Config.SUBMISSION_PATH
        )
    else:
        print(
            f"\nValidation metric ({val_auc}) did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
