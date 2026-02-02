import sys
import os
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import roc_auc_score

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. Patch Configuration for Speed
# -----------------------------------------------------------------------------
# We modify the configuration parameters to run a fast demo.
# This must be done before instantiating classes that rely on these configs.
import library.config as config
import library.ensemble as ensemble_module

# Reduce Random Forest Estimators
config.RF_LEXICAL_PARAMS["n_estimators"] = 10
config.RF_BEHAVIORAL_PARAMS["n_estimators"] = 10

# Reduce XGBoost Estimators
config.XGB_CONTEXTUAL_PARAMS["n_estimators"] = 10

# Reduce CV Folds (Minimum 2 for cross-validation)
ensemble_module.NUM_FOLDS = 2

# -----------------------------------------------------------------------------
# 2. Import Library Modules
# -----------------------------------------------------------------------------
from library.utils import set_seed, Timer
from library.data_loader import load_datasets
from library.features import FeaturePipeline
from library.ensemble import TriViewStackingEnsemble


def main():
    # Set seed for reproducibility
    set_seed(42)
    print("Seed set to 42.")

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("\n--- Loading Data (Debug Mode) ---")
    # Load only 50 samples to ensure the script completes quickly
    train_df, val_df, test_df = load_datasets(load_cached_data=False, debug_size=50)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    # Verify data loaded correctly
    assert len(train_df) == 50
    assert "requester_received_pizza" in train_df.columns

    # -------------------------------------------------------------------------
    # 4. Feature Engineering
    # -------------------------------------------------------------------------
    print("\n--- Feature Engineering ---")
    pipeline = FeaturePipeline()

    # Fit pipeline on training data
    print("Fitting feature pipeline...")
    pipeline.fit(train_df)

    # Transform all splits
    print("Transforming datasets...")
    X_train = pipeline.transform(train_df, "train")
    X_val = pipeline.transform(val_df, "val")
    X_test = pipeline.transform(test_df, "test")

    # Verify Feature Dictionary Structure
    expected_keys = {"lexical", "behavioral", "dense"}
    assert expected_keys.issubset(X_train.keys()), "Missing feature views in output."

    # Verify Shapes (Dense view should have 50 rows)
    assert X_train["dense"].shape[0] == 50
    assert X_val["dense"].shape[0] == 50
    assert X_test["dense"].shape[0] == 50

    print("Feature extraction successful.")

    # -------------------------------------------------------------------------
    # 5. Ensemble Modeling
    # -------------------------------------------------------------------------
    print("\n--- Ensemble Training ---")
    model = TriViewStackingEnsemble()

    y_train = train_df["requester_received_pizza"].values
    y_val = val_df["requester_received_pizza"].values

    # Step A: Get Out-Of-Fold Predictions for Stacking
    print("Generating OOF predictions...")
    oof_preds = model.get_oof_predictions(X_train, y_train)

    # Verify OOF shape (Samples x 3 Base Models)
    assert oof_preds.shape == (50, 3)

    # Step B: Fit Meta-Learner
    print("Fitting Meta-Learner...")
    model.fit_meta_learner(oof_preds, y_train)

    # Step C: Refit Base Models on Full Train Data
    print("Refitting Base Models...")
    model.refit_base_models(X_train, y_train)

    # -------------------------------------------------------------------------
    # 6. Prediction & Validation
    # -------------------------------------------------------------------------
    print("\n--- Prediction & Evaluation ---")

    # Predict on Test Set
    test_probs = model.predict(X_test)

    # Verify Predictions
    assert len(test_probs) == 50
    assert np.all((test_probs >= 0) & (test_probs <= 1)), "Probabilities out of bounds."
    print(f"Test predictions generated. Mean probability: {np.mean(test_probs):.4f}")

    # Validate on Validation Set (Sanity Check)
    val_probs = model.predict(X_val)

    # Handle case where debug subset might have only one class
    if len(np.unique(y_val)) > 1:
        auc_score = roc_auc_score(y_val, val_probs)
        print(f"Validation AUC (on debug subset): {auc_score:.4f}")
    else:
        print("Skipping AUC calculation: Validation subset contains only one class.")

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()
