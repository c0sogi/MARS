import pandas as pd
import numpy as np
import sys

# Import from the provided library files
from library.config import NUMERICAL_FEATURES, SEED
from library.utils import set_seed, compute_auc
from library.data_loader import load_and_preprocess_data
from library.feature_extractor import HybridFeaturePipeline
from library.trainer import Trainer
from library.model_dispatcher import get_logistic_regression


def main():
    # 1. Setup
    set_seed(SEED)

    # 2. Load Data
    # Uses caching to speed up subsequent runs
    print("Loading data...")
    df_train, df_val, df_test = load_and_preprocess_data(load_cached_data=True)

    # 3. Feature Extraction
    # Generates semantic embeddings (GPU) and scales numerical data
    print("Extracting features...")
    pipeline = HybridFeaturePipeline()
    # Force regeneration to apply new scaling (RobustScaler)
    X_train, y_train, X_val, y_val, X_test = pipeline.fit_transform(
        df_train, df_val, df_test, load_cached_data=False
    )

    # 4. Model Tuning (Cross-Validation)
    trainer = Trainer()
    best_C = trainer.run_cross_validation(X_train, y_train, k_folds=5)

    # 5. Validation Evaluation
    # Train a temporary model on X_train only to evaluate on hold-out X_val
    print("\nEvaluating on hold-out validation set...")
    val_model = get_logistic_regression(C=best_C)
    val_model.fit(X_train, y_train)

    # Inference on validation set
    val_preds = val_model.predict_proba(X_val)[:, 1]

    # Compute Metric
    val_auc = compute_auc(y_val, val_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("\nPerforming failure analysis...")
    # Calculate absolute error (since y is 0 or 1, this is |y - p|)
    errors = np.abs(y_val - val_preds)

    # Identify numerical columns in the concatenated feature matrix
    # The matrix is [Embeddings (N cols) | Numerical (M cols)]
    num_numerical = len(NUMERICAL_FEATURES)
    total_features = X_val.shape[1]
    embedding_dim = total_features - num_numerical

    if embedding_dim < 0:
        print("Error: Feature matrix dimensions inconsistent with configuration.")
    else:
        # Extract numerical part of X_val for correlation analysis
        X_val_num = X_val[:, embedding_dim:]

        # Create DataFrame for correlation calculation
        df_analysis = pd.DataFrame(X_val_num, columns=NUMERICAL_FEATURES)
        df_analysis["error"] = errors

        # Compute correlations
        correlations = (
            df_analysis.corr()["error"]
            .drop("error")
            .sort_values(key=abs, ascending=False)
        )

        print("Correlation between prediction error and numerical features:")
        print(correlations.head(5))

    # 7. Submission Generation
    # Threshold defined in task description
    THRESHOLD = 0.6994047619047619

    if val_auc > THRESHOLD:
        print(f"\nValidation metric {val_auc} exceeds threshold {THRESHOLD}.")
        print("Retraining on full dataset (Train + Val) and generating submission...")

        # Retrain on combined data using the best hyperparameter
        trainer.train_final_model(X_train, y_train, X_val, y_val)

        # Generate submission file
        trainer.generate_submission(X_test)
    else:
        print(f"\nValidation metric {val_auc} does not exceed threshold {THRESHOLD}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
