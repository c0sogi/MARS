import numpy as np
import pandas as pd
import os
import sys
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Import provided library modules
from library.config import Config
from library.feature_engineering import get_all_features
from library.stacking_manager import StackingManager
from library.data_loader import load_data


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("Starting Pipeline Execution...")

    # 2. Data Loading & Feature Engineering
    # Loads cached features if available, otherwise computes them
    print("Loading features...")
    data = get_all_features(load_cached_data=True)

    # 3. Initialize Stacking Manager
    manager = StackingManager()

    # 4. Train Meta-Learner (Level 2) via Cross-Validation
    # This generates OOF predictions on the Training set and fits the Meta-Learner
    print("\n--- Phase 1: Meta-Learner Training (CV) ---")
    manager.train_cv(data)

    # 5. Validation Phase
    # Evaluate the ensemble on the hold-out Validation Set
    print("\n--- Phase 2: Hold-out Validation ---")

    y_train = data["y_train"]
    y_val = data["y_val"]

    # Storage for Level 1 predictions on Validation set
    val_preds_l1 = np.zeros((len(y_val), len(manager.l1_models)))

    print("Training Base Learners on Training Set for Validation...")
    for i, (name, model) in enumerate(manager.l1_models.items()):
        # Retrieve Training Data
        X_train_main, X_train_meta = manager.get_model_input(name, data, "train")
        # Retrieve Validation Data
        X_val_main, X_val_meta = manager.get_model_input(name, data, "val")

        # Fit Model
        # Special handling for XGBoost to use early stopping without leaking the hold-out set
        if name == "semantic_booster":
            # Split Training data internally for early stopping
            # This ensures we don't use the actual hold-out X_val for training decisions here
            X_tr_main, X_te_main, X_tr_meta, X_te_meta, y_tr, y_te = train_test_split(
                X_train_main,
                X_train_meta,
                y_train,
                test_size=0.1,
                random_state=Config.SEED,
                stratify=y_train,
            )
            eval_set = (X_te_main, X_te_meta, y_te)
            model.fit(X_tr_main, X_tr_meta, y_tr, eval_set=eval_set)
        else:
            # For RF and Linear models, fit on full training set
            model.fit(X_train_main, X_train_meta, y_train)

        # Predict on Validation Set
        val_preds_l1[:, i] = model.predict_proba(X_val_main, X_val_meta)

    # Generate Final Validation Predictions using the Meta-Learner
    # The Meta-Learner was trained in Phase 1 on OOF predictions
    val_preds_final = manager.meta_learner.predict_proba(val_preds_l1)[:, 1]

    # Compute and Print Metric
    val_auc = roc_auc_score(y_val, val_preds_final)
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("\n--- Phase 3: Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(y_val - val_preds_final)

    # Load raw validation dataframe to get interpretable feature values
    _, val_df, _ = load_data()

    # Correlate error with numerical features
    correlations = {}
    for col in Config.NUMERICAL_COLS:
        if col in val_df.columns:
            # Fill NaNs with median for correlation calculation
            feat_vals = val_df[col].fillna(val_df[col].median())

            # Ensure alignment (should be aligned by index reset in data_loader/metadata)
            if len(feat_vals) == len(errors):
                corr = np.corrcoef(feat_vals, errors)[0, 1]
                correlations[col] = corr

    # Sort and print correlations
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print("Correlation between Prediction Error and Input Features:")
    for col, corr in sorted_corr:
        print(f"  {col}: {corr:.4f}")

    # 7. Submission
    print("\n--- Phase 4: Submission ---")
    threshold = 0.7085870249842536

    if val_auc > threshold:
        print(f"Validation AUC ({val_auc}) meets threshold ({threshold}).")
        print("Retraining models on full data and generating submission...")

        # This method handles:
        # 1. Retraining RF/Linear on Train + Val
        # 2. Retraining XGBoost on Train (using Val for early stopping)
        # 3. Predicting on Test
        # 4. Using Meta-Learner for final aggregation
        final_test_preds = manager.retrain_and_predict(data)

        # Save submission
        manager.save_submission(final_test_preds)
    else:
        print(
            f"Validation AUC ({val_auc}) does not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
