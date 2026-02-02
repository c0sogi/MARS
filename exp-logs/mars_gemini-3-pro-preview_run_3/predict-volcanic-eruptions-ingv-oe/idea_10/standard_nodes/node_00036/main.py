import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import joblib
import warnings
import lightgbm as lgb

# Import from provided libraries
from library.config import METADATA_DIR, SUBMISSION_PATH, RANDOM_SEED, N_FOLDS, N_JOBS
from library.data_processor import generate_feature_matrix
from library.models import get_base_models, get_meta_model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # Set global seeds for reproducibility
    np.random.seed(RANDOM_SEED)

    # Validation Threshold
    VALIDATION_THRESHOLD = 2739761.2592384242

    print("Starting Peak-Aware Stacked Kinematic Ensemble Pipeline...")

    # ==========================================
    # 1. Data Loading & Preparation
    # ==========================================
    print("Loading Metadata and Features...")

    # Load Metadata
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Generate/Load Features (using cache)
    train_features = generate_feature_matrix("train", load_cached=True)
    val_features = generate_feature_matrix("val", load_cached=True)
    test_features = generate_feature_matrix("test", load_cached=True)

    # Merge targets
    train_df = train_features.merge(
        train_meta[["segment_id", "time_to_eruption"]], on="segment_id"
    )
    val_df = val_features.merge(
        val_meta[["segment_id", "time_to_eruption"]], on="segment_id"
    )

    # Prepare Feature Matrices
    feature_cols = [
        c for c in train_df.columns if c not in ["segment_id", "time_to_eruption"]
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["time_to_eruption"]

    X_val = val_df[feature_cols]
    y_val = val_df["time_to_eruption"]

    X_test = test_features[feature_cols]

    print(
        f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}"
    )

    # ==========================================
    # 2. Phase 1: Validation (Train on Train, Eval on Val)
    # ==========================================
    print("\n=== Phase 1: Validation ===")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # Initialize structures
    base_models_proto = get_base_models(random_seed=RANDOM_SEED, n_jobs=N_JOBS)
    model_names = list(base_models_proto.keys())

    oof_train = pd.DataFrame(np.nan, index=X_train.index, columns=model_names)
    best_iterations = {name: [] for name in model_names}

    print("Level 0: Cross-Validation on Train Set...")
    for fold, (t_idx, v_idx) in enumerate(kf.split(X_train, y_train)):
        X_t, X_v = X_train.iloc[t_idx], X_train.iloc[v_idx]
        y_t, y_v = y_train.iloc[t_idx], y_train.iloc[v_idx]

        fold_models = get_base_models(random_seed=RANDOM_SEED + fold, n_jobs=N_JOBS)

        for name, model in fold_models.items():
            # Fit with early stopping to find optimal iterations
            if name == "cat":
                model.fit(
                    X_t,
                    y_t,
                    eval_set=(X_v, y_v),
                    early_stopping_rounds=100,
                    verbose=False,
                )
                best_iter = model.get_best_iteration()
                best_iterations[name].append(
                    best_iter if best_iter is not None else model.tree_count_
                )
            elif name == "lgbm":
                callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
                model.fit(X_t, y_t, eval_set=[(X_v, y_v)], callbacks=callbacks)
                best_iterations[name].append(model.best_iteration_)
            elif name == "xgb":
                model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
                best_iterations[name].append(model.best_iteration)

            # Predict OOF
            oof_train.loc[v_idx, name] = model.predict(X_v)

    # Train Meta Model on Train OOF
    print("Level 1: Training Meta Model on Train OOF...")
    meta_model_val = get_meta_model(random_seed=RANDOM_SEED)
    meta_model_val.fit(oof_train, y_train)

    # Retrain Base Models on Full Train using Average Best Iterations
    # This prevents leakage of Val labels into the training process
    print("Retraining Base Models on Full Train for Validation...")
    val_base_preds = pd.DataFrame(index=X_val.index, columns=model_names)

    base_models_retrained = get_base_models(random_seed=RANDOM_SEED, n_jobs=N_JOBS)

    for name, model in base_models_retrained.items():
        avg_iter = int(np.mean(best_iterations[name]))
        # Ensure at least 1 iteration
        avg_iter = max(1, avg_iter)

        if name == "lgbm":
            model.set_params(n_estimators=avg_iter)
            model.fit(X_train, y_train)
        elif name == "xgb":
            model.set_params(n_estimators=avg_iter, early_stopping_rounds=None)
            model.fit(X_train, y_train, verbose=False)
        elif name == "cat":
            model.set_params(iterations=avg_iter)
            model.fit(X_train, y_train, verbose=False)

        val_base_preds[name] = model.predict(X_val)

    # Meta Prediction on Val
    val_final_pred = meta_model_val.predict(val_base_preds)

    # Compute Metric
    val_mae = mean_absolute_error(y_val, val_final_pred)
    print(f"Final Validation Metric: {val_mae}")

    # ==========================================
    # 3. Phase 2: Failure Analysis
    # ==========================================
    print("\n=== Phase 2: Failure Analysis ===")
    abs_errors = np.abs(y_val - val_final_pred)

    # Correlation between features and error
    correlations = []
    # Convert to float to avoid issues
    X_val_float = X_val.astype(float)

    for col in X_val_float.columns:
        # Check for constant columns
        if X_val_float[col].std() == 0:
            continue
        corr = np.corrcoef(X_val_float[col], abs_errors)[0, 1]
        if not np.isnan(corr):
            correlations.append((col, corr))

    correlations.sort(key=lambda x: x[1], reverse=True)

    print("Top 5 Features positively correlated with Error (High value -> High Error):")
    for feat, corr in correlations[:5]:
        print(f"  {feat}: {corr:.4f}")

    print("Top 5 Features negatively correlated with Error (Low value -> High Error):")
    for feat, corr in correlations[-5:]:
        print(f"  {feat}: {corr:.4f}")

    # ==========================================
    # 4. Phase 3: Submission
    # ==========================================
    if val_mae < VALIDATION_THRESHOLD:
        print(
            f"\nValidation Metric {val_mae} is better than threshold {VALIDATION_THRESHOLD}."
        )
        print("Proceeding to generate submission on Full Data (Train + Val)...")

        # Combine Train and Val
        X_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        y_full = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

        # We perform CV on Full Data to:
        # 1. Generate OOF for Final Meta Model training
        # 2. Generate Averaged Test Predictions from Fold Models (Robustness)

        oof_full = pd.DataFrame(np.nan, index=X_full.index, columns=model_names)
        test_base_preds_accum = pd.DataFrame(
            0.0, index=X_test.index, columns=model_names
        )

        kf_full = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

        print("Running Full CV and Test Prediction...")
        for fold, (t_idx, v_idx) in enumerate(kf_full.split(X_full, y_full)):
            print(f"  Processing Fold {fold+1}/{N_FOLDS}...")
            X_t, X_v = X_full.iloc[t_idx], X_full.iloc[v_idx]
            y_t, y_v = y_full.iloc[t_idx], y_full.iloc[v_idx]

            fold_models = get_base_models(random_seed=RANDOM_SEED + fold, n_jobs=N_JOBS)

            for name, model in fold_models.items():
                # Fit with early stopping
                if name == "cat":
                    model.fit(
                        X_t,
                        y_t,
                        eval_set=(X_v, y_v),
                        early_stopping_rounds=100,
                        verbose=False,
                    )
                elif name == "lgbm":
                    callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
                    model.fit(X_t, y_t, eval_set=[(X_v, y_v)], callbacks=callbacks)
                elif name == "xgb":
                    model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)

                # OOF Prediction
                oof_full.loc[v_idx, name] = model.predict(X_v)

                # Test Prediction (Accumulate)
                test_base_preds_accum[name] += model.predict(X_test) / N_FOLDS

        # Train Final Meta Model on Full OOF
        print("Training Final Meta Model...")
        meta_model_final = get_meta_model(random_seed=RANDOM_SEED)
        meta_model_final.fit(oof_full, y_full)

        # Final Test Prediction using Stacked Ensemble
        final_test_preds = meta_model_final.predict(test_base_preds_accum)

        # Save Submission
        submission = pd.DataFrame(
            {
                "segment_id": test_features["segment_id"],
                "time_to_eruption": final_test_preds,
            }
        )

        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation Metric {val_mae} did not meet threshold. Submission skipped."
        )


if __name__ == "__main__":
    main()
