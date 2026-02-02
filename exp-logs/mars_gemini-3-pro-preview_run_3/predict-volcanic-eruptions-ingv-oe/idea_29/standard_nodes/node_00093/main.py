import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error

from library.config import Config
from library.data_manager import DataManager
from library.model_engine import ModelEngine


def main():
    # 1. Setup
    Config.setup()

    # 2. Training
    # We use size=None to utilize the full dataset (Train + Val) for Cross-Validation.
    # This is necessary to achieve the high performance required by the threshold.
    print("Starting Training Pipeline...")
    engine = ModelEngine()
    engine.train_kfold_ensemble(size=None)

    # 3. Validation & Failure Analysis Preparation
    # We need to reconstruct the OOF predictions to get the exact metric and perform analysis.
    print("\nPreparing Validation Analysis...")

    # Load full data (Train + Val) exactly as ModelEngine did to ensure index alignment
    X_train_part, y_train_part = DataManager.get_train_data(size=None)
    X_val_part, y_val_part = DataManager.get_val_data(size=None)

    X = pd.concat([X_train_part, X_val_part], axis=0).reset_index(drop=True)
    y = pd.concat([y_train_part, y_val_part], axis=0).reset_index(drop=True)

    # Replicate StratifiedKFold logic used in training
    num_bins = 20
    if len(y) < num_bins * 5:
        num_bins = max(2, len(y) // 5)
    y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    oof_preds = np.zeros(len(X))

    print("Generating OOF predictions...")
    for fold, (_, val_idx) in enumerate(skf.split(X, y_bins)):
        X_fold_val = X.iloc[val_idx]

        # Load the model trained for this fold
        model_path = os.path.join(Config.CACHE_DIR, f"lgbm_model_fold_{fold}.txt")
        if not os.path.exists(model_path):
            print(f"Error: Model for fold {fold} not found.")
            continue

        model = lgb.Booster(model_file=model_path)

        # Predict
        preds = model.predict(X_fold_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = preds

    # 4. Final Metric
    final_mae = mean_absolute_error(y, oof_preds)
    print(f"Final Validation Metric: {final_mae}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    errors = np.abs(y - oof_preds)

    # Calculate correlation between features and error magnitude
    print("Calculating feature correlations with error...")
    correlations = (
        X.corrwith(pd.Series(errors, name="error")).abs().sort_values(ascending=False)
    )

    print("Top 5 features associated with high prediction error:")
    print(correlations.head(5))

    # 6. Submission Logic
    THRESHOLD = 2617304.0647319085

    if final_mae < THRESHOLD:
        print(f"\nValidation Metric {final_mae} is better than threshold {THRESHOLD}.")
        print("Proceeding to generate submission...")
        engine.predict_ensemble(size=None)
    else:
        print(f"\nValidation Metric {final_mae} did not meet threshold {THRESHOLD}.")
        print("Submission generation skipped.")


if __name__ == "__main__":
    main()
