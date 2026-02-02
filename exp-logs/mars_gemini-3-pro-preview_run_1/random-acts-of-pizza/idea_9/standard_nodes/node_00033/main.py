import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, save_submission
from library.data_loader import DataLoader
from library.feature_engineering import SemanticProcessor, TabularProcessor
from library.model_rf import train_predict_rf
from library.model_mlp import train_mlp, predict_mlp

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    set_seed()

    # 2. Load Data
    # Using load_cached_data=True to utilize preprocessed parquet files if available
    df_train, df_val, df_test = DataLoader.load_raw_data(load_cached_data=True)

    # Extract Targets
    y_train = df_train[Config.TARGET_COL].values
    y_val = df_val[Config.TARGET_COL].values

    # 3. Feature Selection
    # Identify safe tabular features to prevent leakage
    safe_cols = DataLoader.filter_leakage_columns(df_train, df_test)

    # 4. Feature Engineering

    # 4a. Semantic Features (SBERT)
    # Generates embeddings for request text and subreddit history
    sem_processor = SemanticProcessor()
    (
        train_text_emb,
        train_sub_emb,
        val_text_emb,
        val_sub_emb,
        test_text_emb,
        test_sub_emb,
    ) = sem_processor.process_data(df_train, df_val, df_test, load_cached_data=True)

    # 4b. Tabular Features (TF-IDF + Metadata)
    # Generates TF-IDF vectors and engineered metadata (imputed for RF, scaled for MLP)
    tab_processor = TabularProcessor()
    (
        train_tfidf,
        train_meta_a,
        train_meta_b,
        val_tfidf,
        val_meta_a,
        val_meta_b,
        test_tfidf,
        test_meta_a,
        test_meta_b,
    ) = tab_processor.process_data(
        df_train, df_val, df_test, safe_cols, load_cached_data=True
    )

    # 5. Model Training & Prediction

    # --- Stream A: Random Forest ---
    print("\n=== Running Stream A (Random Forest) ===")
    val_probs_rf, test_probs_rf, model_rf = train_predict_rf(
        train_tfidf,
        train_meta_a,
        y_train,
        val_tfidf,
        val_meta_a,
        y_val,
        test_tfidf,
        test_meta_a,
    )

    # --- Stream B: Attention-Gated MLP ---
    print("\n=== Running Stream B (Attention-Gated MLP) ===")
    model_mlp = train_mlp(
        train_text_emb,
        train_sub_emb,
        train_meta_b,
        y_train,
        val_text_emb,
        val_sub_emb,
        val_meta_b,
        y_val,
    )

    # Inference for Stream B
    print("Stream B: Generating predictions...")
    val_probs_mlp = predict_mlp(model_mlp, val_text_emb, val_sub_emb, val_meta_b)
    test_probs_mlp = predict_mlp(model_mlp, test_text_emb, test_sub_emb, test_meta_b)

    # 6. Ensemble
    print("\n=== Ensembling ===")
    w_rf = Config.ENSEMBLE_WEIGHTS["rf"]
    w_mlp = Config.ENSEMBLE_WEIGHTS["mlp"]

    val_preds_final = (w_rf * val_probs_rf) + (w_mlp * val_probs_mlp)
    test_preds_final = (w_rf * test_probs_rf) + (w_mlp * test_probs_mlp)

    # 7. Validation Assessment
    final_auc = roc_auc_score(y_val, val_preds_final)
    print(f"Final Validation Metric: {final_auc}")

    # 8. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude
    errors = np.abs(y_val - val_preds_final)

    # Create analysis dataframe using safe numerical columns from validation set
    analysis_df = df_val[safe_cols].copy()

    # Ensure all are numeric and handle any potential NaNs for correlation calculation
    analysis_df = analysis_df.select_dtypes(include=[np.number])
    analysis_df = analysis_df.fillna(analysis_df.median())

    analysis_df["error_magnitude"] = errors

    # Compute correlation
    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")
    # Sort by absolute correlation
    correlations_abs = correlations.abs().sort_values(ascending=False)

    print("Top 10 features correlated with error magnitude:")
    print(correlations.loc[correlations_abs.index[:10]])

    # 9. Submission
    threshold = 0.6942941584973917
    if final_auc > threshold:
        print(
            f"\nValidation metric ({final_auc}) > threshold ({threshold}). Generating submission..."
        )
        ids = df_test[Config.ID_COL].values
        save_submission(ids, test_preds_final)
    else:
        print(
            f"\nValidation metric ({final_auc}) <= threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
