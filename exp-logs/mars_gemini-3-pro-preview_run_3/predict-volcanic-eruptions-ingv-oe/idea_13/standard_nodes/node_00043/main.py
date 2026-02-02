import os
import sys
import numpy as np
import pandas as pd

# Import from provided library files
from library.config import SUBMISSION_PATH, SEED
from library.utils import seed_everything, calculate_mae
from library.data_processor import DatasetBuilder
from library.stacking_engine import StackingTrainer


def main():
    # 1. Setup
    seed_everything(SEED)
    print("Starting pipeline execution...")

    # 2. Data Loading
    # Initialize the dataset builder which handles caching and parallel feature extraction
    print("Initializing DatasetBuilder...")
    builder = DatasetBuilder()

    # Load Training Data
    # We use the full training set provided in metadata
    print("Loading Training Data...")
    X_train, y_train, ids_train = builder.get_train_data(load_cached_data=True)

    # Load Validation Data
    # This is the hold-out set defined in ./metadata/val.csv
    print("Loading Validation Data...")
    X_val, y_val, ids_val = builder.get_val_data(load_cached_data=True)

    print(f"Data Loaded. Train: {X_train.shape}, Val: {X_val.shape}")

    # 3. Model Training
    print("Initializing StackingTrainer...")
    trainer = StackingTrainer()

    # Step 3a: Train Base Layer (CV on Training Set)
    # This trains LGBM, XGB, and HGB using Stratified K-Fold CV on X_train.
    # It returns the Out-of-Fold (OOF) predictions for X_train.
    print("Training Base Layer (CV)...")
    oof_preds_train = trainer.train_base_layer(X_train, y_train)

    # Step 3b: Train Meta Layer
    # The Ridge Meta Learner is trained on the OOF predictions and true targets of the training set.
    print("Training Meta Layer...")
    trainer.train_meta_layer(oof_preds_train, y_train)

    # Step 3c: Retrain Base Models on Full Training Data
    # To evaluate on the validation set, we retrain the base models on the entire X_train
    # using the average best iterations found during CV.
    print("Retraining Base Models on Full Training Data...")
    trainer.retrain_full_base(X_train, y_train)

    # 4. Validation & Evaluation
    print("Performing Inference on Validation Set...")
    # Generate predictions using the full stack (Base Models -> Meta Model)
    val_preds = trainer.predict_stack(X_val)

    # Calculate Metric
    val_mae = calculate_mae(y_val, val_preds)
    # Print the metric in the exact required format
    print(f"Final Validation Metric: {val_mae}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error for each validation sample
    errors = np.abs(y_val - val_preds)

    # Calculate correlation between the error magnitude and each input feature
    n_features = X_val.shape[1]
    correlations = []

    # Use errstate to suppress warnings for constant features (div by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(n_features):
            feature_values = X_val[:, i]
            # Check for constant features to avoid NaN correlation
            if np.std(feature_values) > 1e-9:
                corr = np.corrcoef(feature_values, errors)[0, 1]
                if not np.isnan(corr):
                    correlations.append((f"f_{i}", corr))
            else:
                correlations.append((f"f_{i}", 0.0))

    # Sort features by the absolute value of their correlation with the error
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features Correlated with Error Magnitude:")
    for name, corr in correlations[:5]:
        print(f"{name}: {corr:.4f}")

    # 6. Submission Logic
    # The threshold is specified in the task description
    THRESHOLD = 2739761.2592384242

    if val_mae < THRESHOLD:
        print(
            f"\nValidation Metric ({val_mae}) meets threshold ({THRESHOLD}). Proceeding to submission."
        )

        # Optimization: Retrain base models on combined Train + Val data
        # This utilizes all available labeled data to maximize test performance.
        print("Retraining Base Models on Combined (Train + Val) Data...")
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])

        trainer.retrain_full_base(X_full, y_full)

        # Load Test Data
        print("Loading Test Data...")
        X_test, _, ids_test = builder.get_test_data(load_cached_data=True)

        # Generate Predictions for Test Set
        print("Generating Test Predictions...")
        test_preds = trainer.predict_stack(X_test)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"segment_id": ids_test, "time_to_eruption": test_preds}
        )

        # Ensure submission directory exists and save
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved successfully to {SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation Metric ({val_mae}) does NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
