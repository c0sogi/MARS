import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Ensure the working directory is in the path for imports
sys.path.append(os.getcwd())

from library.config import SEED, TARGET_COL, ID_COL, N_FOLDS
from library.utils import set_seed
from library.data_loader import load_dataset
from library.feature_engineering import FeaturePipeline
from library.model_definitions import LexicalRF, BehavioralRF, SemanticXGB, MetaLearner
from library.trainer import train_stacking_ensemble


def run():
    # Set reproducible seed
    set_seed(SEED)

    print("=== Starting Validation Phase ===")

    # 1. Load Data
    # We load cached data if available for speed, but ensure we have separate train/val
    train_df, val_df, test_df = load_dataset(load_cached_data=True)

    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values

    # 2. Feature Engineering
    # Fit pipeline strictly on Train to prevent leakage during validation
    print("Fitting FeaturePipeline on Train set...")
    pipeline = FeaturePipeline()
    pipeline.fit(train_df)

    print("Transforming Train and Val sets...")
    train_views = pipeline.transform(train_df)
    val_views = pipeline.transform(val_df)

    # 3. Level 1 Cross-Validation (Stacking) on Train
    print(f"Running {N_FOLDS}-Fold CV on Train to generate OOF predictions...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Initialize OOF matrix: (n_samples, n_models)
    oof_preds = np.zeros((len(train_df), 3))

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_train)):
        # Slice data for this fold
        # Lexical (Sparse)
        X_lex_t = train_views["lexical"][train_idx]
        X_lex_v = train_views["lexical"][val_idx]

        # Behavioral (Sparse)
        X_beh_t = train_views["behavioral"][train_idx]
        X_beh_v = train_views["behavioral"][val_idx]

        # Semantic (Dense)
        X_sem_t = train_views["semantic"][train_idx]
        X_sem_v = train_views["semantic"][val_idx]

        # Targets
        y_t = y_train[train_idx]
        y_v = y_train[val_idx]

        # --- Train & Predict Level 1 Models ---

        # 1. Lexical RF
        model_lex = LexicalRF().fit(X_lex_t, y_t)
        p_lex = model_lex.predict_proba(X_lex_v)[:, 1]

        # 2. Behavioral RF
        model_beh = BehavioralRF().fit(X_beh_t, y_t)
        p_beh = model_beh.predict_proba(X_beh_v)[:, 1]

        # 3. Semantic XGB
        # We use the fold validation set for early stopping
        model_sem = SemanticXGB().fit(X_sem_t, y_t, X_val=X_sem_v, y_val=y_v)
        p_sem = model_sem.predict_proba(X_sem_v)[:, 1]

        # Store OOF predictions
        oof_preds[val_idx, 0] = p_lex
        oof_preds[val_idx, 1] = p_beh
        oof_preds[val_idx, 2] = p_sem

    # 4. Train Level 2 Meta-Learner
    print("Training Meta-Learner on OOF predictions...")
    meta_learner = MetaLearner().fit(oof_preds, y_train)

    # 5. Retrain Level 1 Models on Full Train
    print("Retraining Level 1 Models on full Train set...")
    full_lex = LexicalRF().fit(train_views["lexical"], y_train)
    full_beh = BehavioralRF().fit(train_views["behavioral"], y_train)
    full_sem = SemanticXGB().fit(train_views["semantic"], y_train)

    # 6. Generate Validation Predictions
    print("Generating predictions for Validation set...")
    val_p_lex = full_lex.predict_proba(val_views["lexical"])[:, 1]
    val_p_beh = full_beh.predict_proba(val_views["behavioral"])[:, 1]
    val_p_sem = full_sem.predict_proba(val_views["semantic"])[:, 1]

    # Stack Level 1 predictions
    X_val_level2 = np.column_stack([val_p_lex, val_p_beh, val_p_sem])

    # Final prediction via Meta-Learner
    val_final_preds = meta_learner.predict_proba(X_val_level2)[:, 1]

    # 7. Calculate Metric
    val_auc = roc_auc_score(y_val, val_final_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 8. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(y_val - val_final_preds)

    # Select numerical columns for correlation
    # We look at the original dataframe features
    numeric_cols = val_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = [
        TARGET_COL,
        ID_COL,
        "unix_timestamp_of_request",
        "unix_timestamp_of_request_utc",
    ]
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    correlations = {}
    for col in numeric_cols:
        # Skip constant columns
        if val_df[col].nunique() <= 1:
            continue

        # Fill NaNs with median for correlation calculation
        series = val_df[col].fillna(val_df[col].median())

        # Calculate correlation with error
        corr = np.corrcoef(series, errors)[0, 1]
        if not np.isnan(corr):
            correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Prediction Error:")
    for name, val in sorted_corr[:5]:
        print(f"  {name}: {val:.4f}")

    # 9. Submission
    threshold = 0.6913548345419015
    if val_auc > threshold:
        print("\n=== Metric exceeds threshold. Proceeding to Submission Phase ===")
        # We use the library trainer to handle the full retraining on Train + Val
        # We set load_cached_data=False to force a fresh processing of the combined dataset
        # to ensure no artifacts from the split processing remain.
        train_stacking_ensemble(load_cached_data=False)
    else:
        print(
            f"\nMetric {val_auc} did not exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    run()
