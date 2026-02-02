import sys
import os
import pandas as pd
import numpy as np
import warnings

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.feature_extraction import generate_dataset
from library.training_pipeline import StackedEnsemblePipeline
from library.utils import seed_everything, mae_metric

# Suppress warnings from libraries to keep output clean
warnings.filterwarnings("ignore")


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("Initializing Stacked Heterogeneous Ensemble Pipeline...")

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Configure for fast baseline execution within time limits
    # Reducing n_estimators and enabling aggressive early stopping
    Config.LGBM_PARAMS["n_estimators"] = 5000
    Config.XGB_PARAMS["n_estimators"] = 5000
    Config.EARLY_STOPPING_ROUNDS = 100

    # Ensure working directory exists for caching features
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading datasets...")
    # Load features. If cached files exist in Config.WORKING_DIR, they are used.
    # Otherwise, features are generated from raw sensor data.
    train_df = generate_dataset(
        Config.TRAIN_META_PATH, "train_features", load_cached_data=True
    )
    val_df = generate_dataset(
        Config.VAL_META_PATH, "val_features", load_cached_data=True
    )
    test_df = generate_dataset(
        Config.TEST_META_PATH, "test_features", load_cached_data=True
    )

    pipeline = StackedEnsemblePipeline()

    # ==========================================
    # 3. Validation Phase
    # ==========================================
    print("\n" + "=" * 40)
    print("STARTING VALIDATION PHASE")
    print("=" * 40)

    # Prepare data matrices
    # We train on Train Set and Evaluate on Hold-out Validation Set
    X_train, y_train, _ = pipeline.get_X_y(train_df)
    X_val, y_val, _ = pipeline.get_X_y(val_df)

    # Step 3.1: Level 0 - Cross-Validation on Training Data
    print("Running Level 0 Cross-Validation on Training Set...")
    oof_train = pipeline.run_cross_validation(X_train, y_train, n_folds=Config.N_FOLDS)

    # Step 3.2: Level 1 - Train Meta-Learner on Training OOF
    print("Training Meta-Learner (Ridge) on Training OOF...")
    meta_model = pipeline.train_meta_learner(oof_train, y_train)

    # Step 3.3: Retrain Base Models on Full Training Data
    # This is necessary to predict on the Validation Set
    print("Retraining Base Models on Training Set...")
    base_models = pipeline.retrain_base_models(X_train, y_train)

    # Step 3.4: Inference on Hold-out Validation Set
    print("Predicting on Hold-out Validation Set...")
    val_preds = pipeline.predict_ensemble(base_models, meta_model, X_val)

    # Step 3.5: Metric Calculation
    val_mae = mae_metric(y_val, val_preds)
    print(f"Final Validation Metric: {val_mae}")

    # Step 3.6: Failure Analysis
    print("\n" + "-" * 20)
    print("Failure Analysis")
    print("-" * 20)

    # Calculate absolute errors
    errors = np.abs(y_val - val_preds)
    error_series = pd.Series(errors, index=X_val.index)

    # Calculate correlation between input features and error magnitude
    # This helps identify which features are associated with high prediction errors
    correlations = X_val.corrwith(error_series).abs().sort_values(ascending=False)

    print("Top 10 Features correlated with Error Magnitude:")
    print(correlations.head(10))

    # ==========================================
    # 4. Submission Phase
    # ==========================================
    THRESHOLD = 2739761.2592384242

    if val_mae < THRESHOLD:
        print("\n" + "=" * 40)
        print("STARTING SUBMISSION PHASE")
        print("=" * 40)
        print(f"Validation MAE ({val_mae}) is below threshold ({THRESHOLD}).")
        print(
            "Proceeding to retrain on full dataset (Train + Val) and generate submission."
        )

        # Combine Train and Validation sets for maximum data utilization
        full_train_df = pd.concat([train_df, val_df], ignore_index=True)
        X_full, y_full, _ = pipeline.get_X_y(full_train_df)
        X_test, _, test_ids = pipeline.get_X_y(test_df, is_test=True)

        # Step 4.1: Level 0 - Cross-Validation on Full Data
        print("Running Level 0 Cross-Validation on Full (Train+Val) Set...")
        oof_full = pipeline.run_cross_validation(X_full, y_full, n_folds=Config.N_FOLDS)

        # Step 4.2: Level 1 - Train Meta-Learner on Full OOF
        print("Training Meta-Learner on Full OOF...")
        meta_model_full = pipeline.train_meta_learner(oof_full, y_full)

        # Step 4.3: Retrain Base Models on Full Data
        print("Retraining Base Models on Full (Train+Val) Set...")
        base_models_full = pipeline.retrain_base_models(X_full, y_full)

        # Step 4.4: Inference on Test Set
        print("Predicting on Test Set...")
        test_preds = pipeline.predict_ensemble(
            base_models_full, meta_model_full, X_test
        )

        # Step 4.5: Save Submission
        submission_df = pd.DataFrame(
            {"segment_id": test_ids, "time_to_eruption": test_preds}
        )

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print("\n" + "=" * 40)
        print("SUBMISSION ABORTED")
        print("=" * 40)
        print(f"Validation MAE ({val_mae}) did not meet the threshold ({THRESHOLD}).")


if __name__ == "__main__":
    main()
