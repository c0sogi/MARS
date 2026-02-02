import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(".")

from library.config import Config
from library.utils import set_seed, load_data
from library.hybrid_ensemble import HybridStackingEnsemble

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    set_seed(Config.RANDOM_SEED)

    # Force DEBUG to False to use full dataset for best performance
    Config.DEBUG = False

    # -------------------------------------------------------------------------
    # Phase 1: Validation Run (Strict Hold-out Evaluation)
    # -------------------------------------------------------------------------
    # We disable the "Hybrid Retraining" feature temporarily.
    # This ensures that when we predict on the validation set, we are using
    # models trained ONLY on the training set (via CV), preventing data leakage.
    original_retrain_flags = Config.RETRAIN_FLAGS.copy()
    for key in Config.RETRAIN_FLAGS:
        Config.RETRAIN_FLAGS[key] = False

    print("Initializing Ensemble for Validation...")
    ensemble = HybridStackingEnsemble()

    # Fit on Training Data (Level 1 CV + Level 2 Meta on OOF)
    ensemble.fit(load_cached_data=True)

    # Load Hold-out Validation Data
    _, val_df, _ = load_data()
    y_val = val_df[Config.TARGET_COL].values

    # Generate Predictions on Validation Set
    # We manually replicate the ensemble inference logic here because the
    # provided predict() method is hardcoded for the test set.
    print("Generating predictions on hold-out validation set...")
    feats_val = ensemble.feature_pipeline.transform(
        val_df, split_name="val", load_cached_data=True
    )

    level1_val_preds = pd.DataFrame(
        index=val_df.index, columns=ensemble.registry.keys()
    )

    for name, config in ensemble.registry.items():
        feature_keys = config["feature_sets"]
        X_val = ensemble._concat_features(feats_val, feature_keys)

        # Since we disabled retraining, all models are stored in 'folds'
        # We average the predictions from all 5 folds (CV-Bagging)
        fold_preds = []
        for model in ensemble.trained_models[name]["folds"]:
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X_val)[:, 1]
            else:
                p = model.predict(X_val)
            fold_preds.append(p)
        level1_val_preds[name] = np.mean(fold_preds, axis=0)

    # Level 2 Meta-Learner Prediction
    val_preds = ensemble.meta_learner.predict_proba(level1_val_preds)[:, 1]

    # Compute and Print Metric
    val_score = roc_auc_score(y_val, val_preds)
    print(f"Final Validation Metric: {val_score}")

    # -------------------------------------------------------------------------
    # Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Failure Analysis ---")
    val_df_analysis = val_df.copy()
    val_df_analysis["prediction"] = val_preds
    val_df_analysis["error"] = np.abs(
        val_df_analysis[Config.TARGET_COL] - val_df_analysis["prediction"]
    )

    # Calculate correlation between error and numerical features
    numeric_cols = val_df_analysis.select_dtypes(include=[np.number]).columns
    correlations = []

    for col in numeric_cols:
        if col not in ["prediction", "error", Config.TARGET_COL]:
            try:
                corr = val_df_analysis["error"].corr(val_df_analysis[col])
                if not np.isnan(corr):
                    correlations.append((col, corr))
            except:
                pass

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top Feature Correlations with Prediction Error:")
    for col, corr in correlations[:5]:
        print(f"  {col}: {corr:.4f}")

    # -------------------------------------------------------------------------
    # Phase 2: Submission Run (With Hybrid Retraining)
    # -------------------------------------------------------------------------
    THRESHOLD = 0.7222984867326668

    if val_score > THRESHOLD:
        print("\nValidation score meets threshold. Proceeding to submission...")

        # Restore original configuration to enable Hybrid Retraining
        # (Stable models will be retrained on Train + Val for maximum signal)
        Config.RETRAIN_FLAGS = original_retrain_flags

        # Re-initialize the ensemble to clear state
        ensemble_final = HybridStackingEnsemble()

        # Fit with full capabilities
        ensemble_final.fit(load_cached_data=True)

        # Generate Submission
        ensemble_final.predict(load_cached_data=True)

    else:
        print(
            f"\nValidation score {val_score} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
