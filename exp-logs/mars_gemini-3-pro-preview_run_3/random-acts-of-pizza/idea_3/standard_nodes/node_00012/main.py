import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.config import Config
from library.utils import set_seed
from library.data_processor import DataProcessor
from library.feature_extraction import FeatureGenerator
from library.stacking_manager import StackingEngine


def main():
    # 1. Setup and Reproducibility
    set_seed(Config.SEED)
    print("Initializing Pizza Request Prediction Pipeline...")

    # 2. Data Loading and Processing
    # Loads stratified train/val splits and test set
    # Handles missing values and basic feature engineering (timestamps, word counts)
    dp = DataProcessor()
    train_df, val_df, test_df = dp.process_data(load_cached_data=True)

    # Extract targets
    y_train = train_df[Config.TARGET_COL].values
    y_val = val_df[Config.TARGET_COL].values

    # 3. Feature Generation (Multi-View)
    fg = FeatureGenerator()

    # View A: Lexical (Sparse TF-IDF + Metadata)
    # Used by Lexical Bagger (Random Forest)
    X_train_lex, X_val_lex, X_test_lex = fg.get_lexical_view(
        train_df, val_df, test_df, load_cached_data=True
    )

    # View B: Semantic (Dense SBERT Embeddings + Metadata)
    # Used by Semantic Bagger (RF) and Semantic Booster (XGBoost)
    X_train_sem, X_val_sem, X_test_sem = fg.get_semantic_view(
        train_df, val_df, test_df, load_cached_data=True
    )

    # 4. Stacking Ensemble Training
    engine = StackingEngine()

    # Step 4a: Cross-Validation to train Meta-Learner
    # Generates OOF predictions on the training set to train the L2 Logistic Regression
    print("\n--- Phase 1: Cross-Validation Stacking ---")
    engine.fit_cv(X_train_lex, X_train_sem, y_train)

    # Step 4b: Retrain Base Models
    # Retrains L1 models on the full training set for final inference
    print("\n--- Phase 2: Retraining Base Models ---")
    engine.retrain_base_models(X_train_lex, X_train_sem, y_train)

    # 5. Validation Inference
    print("\n--- Phase 3: Validation ---")
    # Predict on the hold-out validation set
    y_pred_val = engine.predict(X_val_lex, X_val_sem)

    # Compute Metric
    val_auc = roc_auc_score(y_val, y_pred_val)
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("\n--- Phase 4: Failure Analysis ---")
    # Calculate error magnitude
    errors = np.abs(y_val - y_pred_val)

    # Identify numerical features for correlation analysis
    # We use the processed validation dataframe which contains metadata features
    numeric_cols = val_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = [Config.ID_COL, Config.TARGET_COL]
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    correlations = []
    for col in numeric_cols:
        # Skip constant columns
        if val_df[col].nunique() <= 1:
            continue

        # Handle potential NaNs just in case, though DataProcessor handles imputation
        feature_values = val_df[col].fillna(0).values

        # Calculate Pearson correlation
        if np.std(feature_values) > 0 and np.std(errors) > 0:
            corr = np.corrcoef(feature_values, errors)[0, 1]
            if not np.isnan(corr):
                correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top features correlated with prediction error:")
    for name, corr in correlations[:5]:
        print(f"  {name}: {corr:.4f}")

    # 7. Submission Generation
    threshold = 0.6707957169850934

    if val_auc > threshold:
        print(
            f"\nValidation AUC ({val_auc}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Predict on Test Set
        y_pred_test = engine.predict(X_test_lex, X_test_sem)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {
                "request_id": test_df[Config.ID_COL],
                "requester_received_pizza": y_pred_test,
            }
        )

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation AUC ({val_auc}) did not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
