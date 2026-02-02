import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import from the provided library
from library.utils import set_seed, timer, print_header
from library.feature_engineering import DataPipeline
from library.stacking_engine import PentViewStackingEnsemble
from library.model_definitions import (
    get_lexical_rf,
    get_behavioral_rf,
    get_semantic_xgb,
    get_semantic_rf,
    get_contextual_lr,
    get_meta_learner,
)
from library.config import SUBMISSION_PATH, ID_COL, TARGET_COL


def main():
    # 1. Setup and Configuration
    set_seed(42)
    print_header("DEMONSTRATION SCRIPT START")

    # 2. Data Processing Pipeline
    # The DataPipeline handles loading, imputation, scaling, text vectorization, and embedding.
    # It automatically caches results to ./working/idea_15 to save time on subsequent runs.
    pipeline = DataPipeline()

    # process_data() returns a dictionary with 'train', 'val', 'test' keys,
    # each containing 'metadata', 'lexical', 'behavioral', 'semantic' views.
    try:
        data = pipeline.process_data(load_cached_data=True)
    except Exception as e:
        print(f"Data processing failed: {e}")
        sys.exit(1)

    # --- Verification Step: Data Structure ---
    print("\nVerifying Data Structure...")
    assert "train" in data and "val" in data and "test" in data
    assert "y" in data["train"]
    assert data["train"]["metadata"].shape[0] == data["train"]["y"].shape[0]
    print(f"Training samples: {data['train']['y'].shape[0]}")
    print(f"Validation samples: {data['val']['y'].shape[0]}")
    print(f"Test samples: {data['test']['ids'].shape[0]}")
    print("Data verification passed.")

    # 3. Instantiate and Optimize Ensemble for Speed
    # We use the PentViewStackingEnsemble class which manages 5 Level-1 models and 1 Meta-learner.
    ensemble = PentViewStackingEnsemble()

    print("\nOptimizing Ensemble Hyperparameters for Fast Demonstration...")

    # Override Level 1 models with "lightweight" versions for speed
    # In a real run, we would use the defaults defined in config.py
    ensemble.models["lexical_rf"] = get_lexical_rf(
        n_estimators=10, max_depth=5, n_jobs=-1
    )
    ensemble.models["behavioral_rf"] = get_behavioral_rf(
        n_estimators=10, max_depth=5, n_jobs=-1
    )
    ensemble.models["semantic_xgb"] = get_semantic_xgb(
        n_estimators=10, max_depth=3, n_jobs=-1
    )
    ensemble.models["semantic_rf"] = get_semantic_rf(
        n_estimators=10, max_depth=5, n_jobs=-1
    )
    ensemble.models["contextual_lr"] = get_contextual_lr(max_iter=50)

    # Override Level 2 Meta-Learner
    ensemble.meta_learner = get_meta_learner(max_iter=50)

    # Reduce CV folds for stacking from 5 to 2
    ensemble.n_folds = 2

    # --- Optimization: Subset Training Data ---
    # To strictly adhere to the "Optimize for Speed" requirement, we will train on a subset.
    subset_size = 200
    if data["train"]["y"].shape[0] > subset_size:
        print(f"Subsetting training data to first {subset_size} samples...")
        for key in ["metadata", "semantic"]:
            data["train"][key] = data["train"][key][:subset_size]
        # Sparse matrices need specific slicing
        for key in ["lexical", "behavioral"]:
            data["train"][key] = data["train"][key][:subset_size]
        data["train"]["y"] = data["train"]["y"][:subset_size]

    # 4. Train the Ensemble
    # This handles the K-Fold OOF generation, Level 1 training, Meta-learner training,
    # and final retraining of Level 1 models on the full provided training set.
    try:
        ensemble.fit(data)
    except Exception as e:
        print(f"Ensemble training failed: {e}")
        sys.exit(1)

    # 5. Validation
    print_header("Validation")
    val_preds = ensemble.predict(data["val"])

    # --- Verification Step: Predictions ---
    assert len(val_preds) == len(data["val"]["y"])
    assert np.all(
        (val_preds >= 0) & (val_preds <= 1)
    ), "Predictions must be probabilities"

    val_auc = roc_auc_score(data["val"]["y"], val_preds)
    print(f"Validation AUC (Fast Mode): {val_auc:.4f}")

    # Sanity check: AUC should be better than random (0.5) or at least valid
    # Note: With tiny data subset and weak models, performance might be noisy,
    # but we check it's a valid float.
    assert 0.0 <= val_auc <= 1.0

    # 6. Generate Submission
    print_header("Generating Submission")
    test_preds = ensemble.predict(data["test"])

    # Create submission DataFrame
    test_ids = data["test"]["ids"]
    submission_df = pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_preds})

    # Ensure output directory exists (handled by config, but good practice)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # Save
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

    # --- Verification Step: Submission File ---
    assert os.path.exists(SUBMISSION_PATH), "Submission file was not created"
    loaded_sub = pd.read_csv(SUBMISSION_PATH)
    assert loaded_sub.shape == (len(test_ids), 2), "Submission shape mismatch"
    assert list(loaded_sub.columns) == [
        ID_COL,
        TARGET_COL,
    ], "Submission columns mismatch"
    print("Submission file verified successfully.")

    print_header("DEMONSTRATION COMPLETE")


if __name__ == "__main__":
    main()
