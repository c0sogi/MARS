import numpy as np
import pandas as pd
import scipy.sparse
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed, save_submission, timer
from library.data_loader import load_dataset
from library.feature_engineering import FeaturePipeline
from library.models import HeptViewEnsemble


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("Initializing Baseline Run...")

    # 2. Data Loading
    # We use limit=None because the dataset is small (~2.3k train) and we need full data for high AUC.
    with timer("Data Loading"):
        train_raw = load_dataset("train", load_cached_data=True, limit=None)
        val_raw = load_dataset("val", load_cached_data=True, limit=None)
        test_raw = load_dataset("test", load_cached_data=True, limit=None)

    y_train = train_raw["y"]
    y_val = val_raw["y"]

    # 3. Feature Engineering
    pipeline = FeaturePipeline()
    with timer("Feature Engineering"):
        pipeline.fit(train_raw)
        train_features = pipeline.transform(train_raw, "train", load_cached_data=True)
        val_features = pipeline.transform(val_raw, "val", load_cached_data=True)
        test_features = pipeline.transform(test_raw, "test", load_cached_data=True)

    # 4. Ensemble Initialization & OOF Training
    ensemble = HeptViewEnsemble()

    with timer("Level 1 OOF Training"):
        oof_preds = ensemble.train_oof(train_features, y_train)

    with timer("Level 2 Meta-Learner Training"):
        ensemble.train_meta(oof_preds, y_train)

    # 5. Validation Assessment (Manual Loop)
    # We need to assess performance on the hold-out validation set.
    # We train base models on Train and predict on Val.
    print("\nPerforming Validation Assessment...")
    val_l1_preds = pd.DataFrame(
        index=np.arange(len(y_val)), columns=ensemble.model_names
    )

    # Calculate scale_pos_weight for XGB based on training data
    n_pos = np.sum(y_train)
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    for name in ensemble.model_names:
        print(f"  Validating base learner: {name}")
        model = ensemble._get_model_instance(name)
        X_train_mod = ensemble._get_data_for_model(train_features, name)
        X_val_mod = ensemble._get_data_for_model(val_features, name)

        if isinstance(model, (RandomForestClassifier, LogisticRegression)):
            model.fit(X_train_mod, y_train)
            val_l1_preds[name] = model.predict_proba(X_val_mod)[:, 1]

        elif isinstance(model, xgb.XGBClassifier):
            model.set_params(scale_pos_weight=scale_pos_weight)
            model.set_params(early_stopping_rounds=50)
            model.fit(
                X_train_mod, y_train, eval_set=[(X_val_mod, y_val)], verbose=False
            )
            val_l1_preds[name] = model.predict_proba(X_val_mod)[:, 1]

        elif isinstance(model, lgb.LGBMClassifier):
            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ]
            model.fit(
                X_train_mod, y_train, eval_set=[(X_val_mod, y_val)], callbacks=callbacks
            )
            val_l1_preds[name] = model.predict_proba(X_val_mod)[:, 1]

    # Generate Final Validation Probabilities using the Meta-Learner
    val_final_probs = ensemble.meta_learner.predict_proba(val_l1_preds)[:, 1]

    # Compute Metric
    final_metric = roc_auc_score(y_val, val_final_probs)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(y_val - val_final_probs)

    # Create DataFrame for Metadata features (scaled)
    # Note: We use the raw metadata values for interpretation if possible,
    # but val_features['metadata'] is already scaled. We'll use the scaled values for correlation.
    # The columns correspond to Config.METADATA_COLS present in the data.
    # We need to be careful about column alignment. FeaturePipeline selects valid cols.
    # We'll assume the order matches Config.METADATA_COLS intersection with df columns.
    # For simplicity, we just use indices or try to reconstruct.
    # Let's use the scaled array directly.

    meta_array = val_features["metadata"]
    # We don't have the column names easily accessible from the pipeline object in this scope
    # without re-loading raw data logic, but we know the list from Config.
    # We will compute correlation against each column index.

    correlations = []
    for i in range(meta_array.shape[1]):
        # Pearson correlation
        if np.std(meta_array[:, i]) > 0:
            corr = np.corrcoef(errors, meta_array[:, i])[0, 1]
            correlations.append((i, corr))
        else:
            correlations.append((i, 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Metadata Features correlated with Error:")
    # Map index back to likely name if possible, otherwise print Index
    # Based on Config.METADATA_COLS, the order is preserved.
    available_cols = Config.METADATA_COLS  # Assuming all are present
    for idx, corr in correlations[:5]:
        col_name = available_cols[idx] if idx < len(available_cols) else f"Feat_{idx}"
        print(f"  {col_name}: {corr:.4f}")

    # 7. Submission
    THRESHOLD = 0.7138293787137718
    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        with timer("Final Retraining"):
            # Retrain on Train + Val (or use Val for ES)
            ensemble.train_final(train_features, y_train, val_features, y_val)

        with timer("Test Prediction"):
            test_probs = ensemble.predict(test_features)

        save_submission(test_probs, test_raw["ids"])
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
