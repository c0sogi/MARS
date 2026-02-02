import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.metrics import roc_auc_score

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

import library.config as config
from library.ensemble_pipeline import HexViewEnsemble
from library.model_definitions import ModelFactory

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # Set reproducibility
    set_seed(config.SEED)

    # Initialize Ensemble Pipeline
    ensemble = HexViewEnsemble()

    # ---------------------------------------------------------
    # 1. Train Meta-Learner via OOF (Level 1 CV -> Level 2 Train)
    # ---------------------------------------------------------
    # This step trains the meta-learner on OOF predictions from the training set.
    # It does NOT save the base learners (they are transient for OOF).
    print("Step 1: Generating OOF predictions and training Meta-Learner...")
    ensemble.train_and_predict_oof(load_cached_data=True)

    # ---------------------------------------------------------
    # 2. Validation on Hold-out Set
    # ---------------------------------------------------------
    # To get a valid metric, we must train base learners on Train and evaluate on Val.
    # We cannot use 'train_final_models' yet because that might train on Train+Val.
    print("\nStep 2: Performing Validation on Hold-out Set...")

    # Load processed data
    data = ensemble.processor.run(load_cached_data=True)
    train_data = data["train"]
    val_data = data["val"]
    y_train = train_data["y"]
    y_val = val_data["y"]

    # Load the Meta-Learner trained in Step 1
    meta_learner_path = os.path.join(ensemble.models_dir, "meta_learner.joblib")
    if not os.path.exists(meta_learner_path):
        raise FileNotFoundError("Meta-learner failed to save.")
    meta_learner = joblib.load(meta_learner_path)

    # Prepare array for Level 1 predictions on Validation set
    n_val = len(y_val)
    n_models = len(ensemble.base_learners_config)
    L1_val_preds = np.zeros((n_val, n_models))

    # Train each base learner on Train, Predict on Val
    for i, (name, factory_func, views) in enumerate(ensemble.base_learners_config):
        # Instantiate fresh model
        model = factory_func()

        # Prepare inputs
        X_train_fold = ensemble._get_model_input(train_data, views)
        X_val_fold = ensemble._get_model_input(val_data, views)

        # Fit Model
        # For boosters, we use Val for early stopping to match the 'Validation-Guided' protocol
        if "booster" in name:
            # Check if model supports eval_set (XGB/LGBM do)
            model.fit(X_train_fold, y_train, eval_set=[(X_val_fold, y_val)])
        else:
            model.fit(X_train_fold, y_train)

        # Predict on Val
        if hasattr(model, "predict_proba"):
            # Binary classification, take probability of positive class
            preds = model.predict_proba(X_val_fold)[:, 1]
        else:
            preds = model.predict(X_val_fold)

        L1_val_preds[:, i] = preds

    # Generate Final Predictions using Meta-Learner
    val_final_probs = meta_learner.predict_proba(L1_val_preds)[:, 1]

    # Compute Metric
    val_auc = roc_auc_score(y_val, val_final_probs)
    print(f"Final Validation Metric: {val_auc}")

    # ---------------------------------------------------------
    # 3. Failure Analysis
    # ---------------------------------------------------------
    print("\nStep 3: Performing Failure Analysis...")

    # Calculate absolute error
    errors = np.abs(y_val - val_final_probs)

    # Load raw validation metadata for correlation analysis
    val_meta_df = pd.read_parquet(config.VAL_METADATA_PATH)

    # Select numerical features for correlation
    numeric_cols = val_meta_df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target and ID
    ignore_cols = [config.TARGET_COL, config.ID_COL]
    numeric_cols = [c for c in numeric_cols if c not in ignore_cols]

    correlations = {}
    for col in numeric_cols:
        # Simple imputation for correlation calculation
        feat_values = val_meta_df[col].fillna(val_meta_df[col].median())

        # Skip constant columns
        if feat_values.nunique() <= 1:
            continue

        # Compute correlation
        corr = np.corrcoef(feat_values, errors)[0, 1]
        if not np.isnan(corr):
            correlations[col] = corr

    # Print top 5 correlations (positive or negative magnitude)
    print("Top correlations between Error and Features:")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corr[:5]:
        print(f"  {feat}: {corr:.6f}")

    # ---------------------------------------------------------
    # 4. Submission
    # ---------------------------------------------------------
    THRESHOLD = 0.7138293787137718

    if val_auc > THRESHOLD:
        print(
            f"\nValidation Metric ({val_auc}) > Threshold ({THRESHOLD}). Proceeding to Submission..."
        )

        # Retrain base models on Full Data (Train + Val) or as configured
        ensemble.train_final_models(load_cached_data=True)

        # Generate Test Predictions
        ensemble.predict_test(load_cached_data=True)

    else:
        print(
            f"\nValidation Metric ({val_auc}) <= Threshold ({THRESHOLD}). Submission Skipped."
        )


if __name__ == "__main__":
    main()
