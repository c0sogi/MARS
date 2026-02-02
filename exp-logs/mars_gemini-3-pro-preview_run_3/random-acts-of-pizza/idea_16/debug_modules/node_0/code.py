import os
import sys
import numpy as np
import pandas as pd
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.utils import set_seed, evaluate_auc
from library.data_loader import load_dataset
from library.features import FeatureExtractor
from library.ensemble import PentViewStackingEnsemble
from library.config import SEED, TARGET_COL, ID_COL


def main():
    print("=== Starting Pipeline Demonstration ===")

    # 1. Set Seed for Reproducibility
    set_seed(SEED)

    # 2. Load Data
    # We force load_cached_data=False to demonstrate the full loading/cleaning logic.
    print("\n[Step 1] Loading Data...")
    train_df, val_df, test_df = load_dataset(load_cached_data=False)

    # OPTIMIZATION: Subsample data for rapid demonstration
    DEMO_SIZE = 50
    print(f"Subsampling datasets to {DEMO_SIZE} rows for speed...")
    train_df = train_df.head(DEMO_SIZE).reset_index(drop=True)
    val_df = val_df.head(DEMO_SIZE).reset_index(drop=True)
    test_df = test_df.head(DEMO_SIZE).reset_index(drop=True)

    # Verify Data Loading
    assert len(train_df) == DEMO_SIZE
    assert TARGET_COL in train_df.columns
    print("Data loaded and subsampled successfully.")

    # 3. Feature Extraction
    print("\n[Step 2] Extracting Features...")
    fe = FeatureExtractor()

    # Fit on training data
    fe.fit(train_df)

    # Transform all splits
    # We use load_cached_data=False to force computation
    print("Transforming Train...")
    X_train = fe.transform(train_df, "demo_train", load_cached_data=False)

    print("Transforming Val...")
    X_val = fe.transform(val_df, "demo_val", load_cached_data=False)

    print("Transforming Test...")
    X_test = fe.transform(test_df, "demo_test", load_cached_data=False)

    # Verify Feature Shapes
    # Metadata should have columns corresponding to DENSE_FEATURE_COLS
    assert X_train["metadata"].shape[0] == DEMO_SIZE
    assert X_train["lexical"].shape[0] == DEMO_SIZE
    assert X_train["semantic"].shape[0] == DEMO_SIZE
    print("Feature extraction complete and verified.")

    # 4. Model Initialization & Optimization
    print("\n[Step 3] Initializing Ensemble...")
    ensemble = PentViewStackingEnsemble()

    # OPTIMIZATION: Reduce hyperparameters for speed
    print("Reducing model complexity for demonstration...")
    for name, model in ensemble.base_models.items():
        # Reduce Random Forest trees
        if hasattr(model, "n_estimators"):
            model.n_estimators = 10
        # Reduce Max Iterations for Linear Models
        if hasattr(model, "max_iter"):
            model.max_iter = 50
        # Specific handling for XGBoost
        if name == "semantic_xgb":
            model.set_params(n_estimators=10)

    # 5. Training (Level 1 OOF + Level 2 Meta)
    print("\n[Step 4] Running OOF Training (Level 1)...")
    y_train = train_df[TARGET_COL].values

    # This trains base models via CV and trains the meta-learner
    oof_preds = ensemble.fit_oof(X_train, y_train)

    # Verify OOF Output
    # Should be (N_samples, 5 base models)
    assert oof_preds.shape == (DEMO_SIZE, 5)
    print("OOF Training complete.")

    # 6. Final Retraining
    print("\n[Step 5] Retraining Base Learners on Full Train Set...")
    ensemble.fit_final(X_train, y_train)

    # 7. Evaluation on Validation Set (Optional check)
    print("\n[Step 6] Evaluating on Validation Set...")
    val_preds = ensemble.predict(X_val)
    y_val = val_df[TARGET_COL].values

    try:
        val_auc = evaluate_auc(y_val, val_preds)
        print(f"Validation AUC (on {DEMO_SIZE} samples): {val_auc:.4f}")
    except ValueError:
        print("Skipping AUC calculation (likely only one class in subsample).")

    # 8. Prediction on Test Set
    print("\n[Step 7] Generating Test Predictions...")
    test_preds = ensemble.predict(X_test)

    # Verify Predictions
    assert len(test_preds) == len(test_df)
    assert np.all((test_preds >= 0) & (test_preds <= 1))
    print("Predictions generated and verified.")

    # 9. Submission
    print("\n[Step 8] Saving Submission...")
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    submission_df = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: test_preds})

    submission_path = os.path.join(submission_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
