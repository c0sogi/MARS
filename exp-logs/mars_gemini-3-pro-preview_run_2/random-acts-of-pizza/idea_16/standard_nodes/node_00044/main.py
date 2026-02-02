import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, print_metric, ensure_directory
from library.data_loader import load_data
from library.feature_engineering import TextEmbedder, TabularPreprocessor
from library.model_trainer import ModelTrainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Initialization and Setup
    set_seed(Config.SEED)

    # 2. Data Loading
    # Load combined train+val data and test data using the provided loader
    # This respects the metadata/train.csv and metadata/val.csv structure by merging them
    df_train, df_test = load_data(load_from_cache=True)

    # Extract target labels
    y_full = df_train["requester_received_pizza"].values.astype(int)

    # 3. Text Embedding Generation (Global)
    # Embeddings are generated using a pre-trained model (unsupervised regarding our labels),
    # so we can compute them for the full datasets once to save time.
    embedder = TextEmbedder()
    train_emb_full = embedder.process_and_cache(
        df_train, Config.TRAIN_EMBEDDINGS_PATH, load_from_cache=True
    )
    test_emb_full = embedder.process_and_cache(
        df_test, Config.TEST_EMBEDDINGS_PATH, load_from_cache=True
    )

    # 4. 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Arrays to store predictions
    oof_preds = np.zeros(len(df_train))
    test_preds_sum = np.zeros(len(df_test))

    print(f"Starting {Config.N_FOLDS}-Fold Stratified Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, y_full)):
        # --- Data Splitting ---
        # Split DataFrames
        df_fold_train = df_train.iloc[train_idx].reset_index(drop=True)
        df_fold_val = df_train.iloc[val_idx].reset_index(drop=True)

        # Split Targets
        y_fold_train = y_full[train_idx]

        # Split Pre-computed Embeddings
        X_emb_train = train_emb_full[train_idx]
        X_emb_val = train_emb_full[val_idx]

        # --- Feature Engineering (Tabular) ---
        # Fit scaler ONLY on training fold to prevent leakage
        tab_preprocessor = TabularPreprocessor()
        X_tab_train = tab_preprocessor.fit_transform(df_fold_train)
        X_tab_val = tab_preprocessor.transform(df_fold_val)

        # --- Feature Fusion ---
        X_train_fold = np.hstack([X_emb_train, X_tab_train])
        X_val_fold = np.hstack([X_emb_val, X_tab_val])

        # --- Model Training ---
        # Optimize and train the Bagged ElasticNet Ensemble
        trainer = ModelTrainer()
        model, best_params = trainer.optimize_and_train(X_train_fold, y_fold_train)

        # --- Validation Inference ---
        val_probs = model.predict_proba(X_val_fold)[:, 1]
        oof_preds[val_idx] = val_probs

        # --- Test Inference ---
        # Transform test tabular data using the scaler fitted on this fold
        X_tab_test = tab_preprocessor.transform(df_test)
        X_test_fold = np.hstack([test_emb_full, X_tab_test])

        test_probs = model.predict_proba(X_test_fold)[:, 1]
        test_preds_sum += test_probs

        # Log fold metric
        fold_auc = roc_auc_score(y_full[val_idx], val_probs)
        # print_metric(f"Fold {fold + 1} AUC", fold_auc)

    # 5. Final Evaluation
    final_auc = roc_auc_score(y_full, oof_preds)
    print_metric("Final Validation Metric", final_auc)

    # 6. Failure Analysis
    print("Performing Failure Analysis...")
    errors = np.abs(y_full - oof_preds)

    # Analyze correlation between numeric features and prediction error
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target and non-feature columns if present
    cols_to_exclude = ["requester_received_pizza", "sample_index"]
    numeric_cols = [c for c in numeric_cols if c not in cols_to_exclude]

    correlations = {}
    for col in numeric_cols:
        # Fill NaNs with median for correlation check
        col_values = df_train[col].fillna(df_train[col].median())
        # Check if column is constant
        if col_values.std() > 0:
            corr = np.corrcoef(col_values, errors)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr

    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top Features Correlated with Prediction Error:")
    for feat, corr in sorted_corrs[:5]:
        print(f"{feat}: {corr}")

    # 7. Submission Generation
    THRESHOLD = 0.7141749705260098
    if final_auc > THRESHOLD:
        avg_test_preds = test_preds_sum / Config.N_FOLDS

        submission_df = pd.DataFrame(
            {
                "request_id": df_test["request_id"],
                "requester_received_pizza": avg_test_preds,
            }
        )

        ensure_directory(Config.SUBMISSION_PATH)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {final_auc} is below threshold {THRESHOLD}. No submission generated."
        )


if __name__ == "__main__":
    main()
