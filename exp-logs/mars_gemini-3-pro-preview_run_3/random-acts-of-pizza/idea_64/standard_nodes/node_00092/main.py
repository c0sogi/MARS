import os
import glob
import pandas as pd
import numpy as np
import joblib
import scipy.sparse
from sklearn.metrics import roc_auc_score

# Import provided libraries
from library.config import Config
from library.utils import set_seed, Timer
from library.data_factory import DataFactory
from library.feature_engine import FeatureGenerator
from library.model_zoo import ModelZoo
from library.training_engine import CrossValidationTrainer
from library.inference_engine import HybridPredictor

# ==============================================================================
# 1. Configuration for Fast Baseline
# ==============================================================================
# Modify Config to speed up execution within the time limit
Config.N_FOLDS = 3
Config.EARLY_STOPPING_ROUNDS = 50

# Reduce model complexity for speed
Config.LEXICAL_RF_PARAMS["n_estimators"] = 50
Config.COMMUNITY_RF_PARAMS["n_estimators"] = 50
Config.SEMANTIC_XGB_PARAMS["n_estimators"] = 100
Config.SEMANTIC_LGBM_PARAMS["n_estimators"] = 100
Config.SEMANTIC_RF_PARAMS["n_estimators"] = 50
Config.METADATA_BOOSTER_PARAMS["n_estimators"] = 100


def main():
    set_seed(Config.SEED)

    # ==========================================================================
    # 2. Training Pipeline
    # ==========================================================================
    print(">>> Starting Training Pipeline...")
    trainer = CrossValidationTrainer()
    trainer.run()

    # ==========================================================================
    # 3. Validation Assessment
    # ==========================================================================
    print("\n>>> Starting Validation Assessment...")

    # Load Validation Data
    if not os.path.exists(Config.VAL_PATH):
        raise FileNotFoundError(f"Validation file not found at {Config.VAL_PATH}")

    val_df = pd.read_parquet(Config.VAL_PATH)
    y_val = val_df[Config.TARGET_COL].values

    # Load Union Data (Train+Val) to use as the fitting basis for FeatureGenerator
    # This ensures we use the same vectorizers/scalers as training
    union_df, _ = DataFactory.load_union_dataset(load_cached_data=True)

    # Preprocess Validation Data (Replicating DataFactory logic)
    # 1. Text Concatenation
    val_df["text_combined"] = (
        val_df[Config.TEXT_COLS[0]].fillna("").astype(str)
        + " "
        + val_df[Config.TEXT_COLS[1]].fillna("").astype(str)
    )

    # 2. Behavioral Feature
    if Config.SUBREDDIT_COL in val_df.columns:
        val_df["subreddit_text"] = val_df[Config.SUBREDDIT_COL].apply(
            lambda x: (
                " ".join(x)
                if isinstance(x, list)
                else (str(x) if x is not None else "")
            )
        )
    else:
        val_df["subreddit_text"] = ""

    # 3. Imputation (Using medians from union_df)
    for col in Config.METADATA_COLS:
        if col in val_df.columns:
            val_df[col] = pd.to_numeric(val_df[col], errors="coerce")
            if col in union_df.columns:
                median_val = union_df[col].median()
                val_df[col] = val_df[col].fillna(median_val)
            else:
                val_df[col] = val_df[col].fillna(0)

    # Generate Features for Validation Set
    # We pass union_df as train and val_df as test.
    # load_cached_data=False forces regeneration (overwriting cache files with val features)
    fg_val = FeatureGenerator(union_df, val_df)

    print("Generating validation features...")
    X_tr_lex, X_val_lex = fg_val.get_lexical_features(load_cached_data=False)
    X_tr_beh, X_val_beh = fg_val.get_behavioral_features(load_cached_data=False)
    X_tr_sem, X_val_sem = fg_val.get_semantic_features(load_cached_data=False)
    X_tr_meta, X_val_meta = fg_val.get_metadata_features(load_cached_data=False)

    # Assemble Inputs
    inputs_val = {}

    def prepare_feature_set(feature_key, X_feature, X_meta):
        if feature_key in ["lexical", "behavioral"]:
            if not scipy.sparse.issparse(X_meta):
                X_meta_sparse = scipy.sparse.csr_matrix(X_meta)
            else:
                X_meta_sparse = X_meta
            return scipy.sparse.hstack([X_feature, X_meta_sparse], format="csr")
        elif feature_key == "semantic":
            return np.hstack([X_feature, X_meta])
        elif feature_key == "metadata":
            return X_meta

    raw_features_val = {
        "lexical": X_val_lex,
        "behavioral": X_val_beh,
        "semantic": X_val_sem,
        "metadata": X_val_meta,
    }

    for f_key, X_raw in raw_features_val.items():
        if f_key == "metadata":
            inputs_val[f_key] = X_raw
        else:
            inputs_val[f_key] = prepare_feature_set(f_key, X_raw, X_val_meta)

    # Level 1 Inference on Validation
    models_conf = ModelZoo.get_models_dict()
    level1_preds = pd.DataFrame()

    print("Running inference on validation set...")
    for model_name, conf in models_conf.items():
        feature_key = conf["feature_set"]
        X_input = inputs_val[feature_key]

        # We use the full models (Stable) or averaged folds (Volatile)
        # For consistency with HybridPredictor:
        if conf["type"] == "volatile":
            final_pred = np.zeros(X_input.shape[0])
            for fold in range(Config.N_FOLDS):
                model_path = os.path.join(
                    Config.WORKING_DIR, f"{model_name}_fold_{fold}.joblib"
                )
                model = joblib.load(model_path)
                final_pred += model.predict_proba(X_input)[:, 1]
            final_pred /= Config.N_FOLDS
        else:
            model_path = os.path.join(Config.WORKING_DIR, f"{model_name}_full.joblib")
            model = joblib.load(model_path)
            final_pred = model.predict_proba(X_input)[:, 1]

        level1_preds[model_name] = final_pred

    # Level 2 Meta-Learner Inference
    meta_learner = joblib.load(os.path.join(Config.WORKING_DIR, "meta_learner.joblib"))
    val_probs = meta_learner.predict_proba(level1_preds.values)[:, 1]

    # Compute Metric
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"Final Validation Metric: {val_auc}")

    # ==========================================================================
    # 4. Failure Analysis
    # ==========================================================================
    print("\n>>> Failure Analysis...")
    val_df["pred"] = val_probs
    val_df["error"] = np.abs(val_df[Config.TARGET_COL] - val_df["pred"])

    numeric_cols = val_df.select_dtypes(include=[np.number]).columns
    correlations = {}

    for col in numeric_cols:
        if col not in ["pred", "error", Config.TARGET_COL]:
            # Simple imputation for correlation calc
            series = val_df[col].fillna(0)
            if series.std() > 0:
                corr = series.corr(val_df["error"])
                correlations[col] = corr

    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Top Feature Correlations with Error:")
    for name, corr in sorted_corr[:5]:
        print(f"{name}: {corr:.4f}")

    # ==========================================================================
    # 5. Submission Generation
    # ==========================================================================
    # Clean up cache so HybridPredictor regenerates features for the actual Test set
    print("\nCleaning up validation feature cache...")
    test_cache_files = glob.glob(os.path.join(Config.WORKING_DIR, "X_test_*.np*"))
    for f in test_cache_files:
        try:
            os.remove(f)
        except OSError:
            pass

    threshold = 0.7222984867326668
    if val_auc > threshold:
        print(
            f"\nValidation metric ({val_auc}) > threshold ({threshold}). Proceeding to submission."
        )
        predictor = HybridPredictor()
        predictor.predict()
    else:
        print(
            f"\nValidation metric ({val_auc}) <= threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
