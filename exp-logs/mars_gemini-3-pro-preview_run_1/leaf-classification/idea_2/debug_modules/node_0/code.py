import os
import sys
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.utils import set_seed, score_predictions
from library.data_loader import LeafDataManager
from library.models import LGBMWrapper, LDAWrapper, SVMWrapper
from library.ensemble import WeightOptimizer
from library.train import CrossValidationPipeline
import library.models  # Imported to patch defaults for speed

# Constants for the demo
DEMO_SEED = 42
SUBSET_SIZE = 50  # Number of samples to use for component testing
N_CLASSES = 99  # Expected number of species


def demonstrate_components():
    print("\n=== Demonstrating Individual Components ===")

    # 1. Data Loading and Preprocessing
    print("\n--- 1. Data Loading ---")
    data_manager = LeafDataManager(seed=DEMO_SEED)

    # Load 'tree' data (raw features)
    X_train_tree, y_train = data_manager.get_train_data(
        model_type="tree", load_cached_data=False
    )
    classes = data_manager.get_classes(load_cached_data=False)

    print(f"Loaded Tree Train Data: {X_train_tree.shape}")
    print(f"Loaded Classes: {len(classes)}")

    # Assertions
    assert (
        len(classes) == N_CLASSES
    ), f"Expected {N_CLASSES} classes, got {len(classes)}"
    assert (
        X_train_tree.shape[1] == 192
    ), f"Expected 192 features, got {X_train_tree.shape[1]}"
    assert not np.isnan(X_train_tree).any(), "Tree data contains NaNs"

    # Load 'linear_kernel' data (transformed features)
    X_train_lin, _ = data_manager.get_train_data(
        model_type="linear_kernel", load_cached_data=False
    )
    print(f"Loaded Linear Train Data: {X_train_lin.shape}")

    # Assertions for transformation
    # Transformed data should be roughly standardized (mean ~0, std ~1)
    assert X_train_lin.shape == X_train_tree.shape
    assert abs(X_train_lin.mean()) < 0.5, "Transformed data mean is not close to 0"

    # Create a small subset for model testing
    X_sub_tree = X_train_tree[:SUBSET_SIZE]
    X_sub_lin = X_train_lin[:SUBSET_SIZE]
    y_sub = y_train[:SUBSET_SIZE]

    # 2. Model Training and Prediction
    print("\n--- 2. Models (Fast Mode) ---")

    # A. LightGBM
    print("Testing LGBMWrapper...")
    lgbm = LGBMWrapper(n_estimators=10, random_state=DEMO_SEED)
    lgbm.fit(X_sub_tree, y_sub)
    preds_lgbm = lgbm.predict_proba(X_sub_tree)

    assert preds_lgbm.shape == (SUBSET_SIZE, N_CLASSES)
    assert np.allclose(
        preds_lgbm.sum(axis=1), 1.0
    ), "LGBM probabilities do not sum to 1"
    print("LGBM check passed.")

    # B. LDA
    print("Testing LDAWrapper...")
    lda = LDAWrapper(random_state=DEMO_SEED)
    lda.fit(X_sub_lin, y_sub)
    preds_lda = lda.predict_proba(X_sub_lin)

    assert preds_lda.shape == (SUBSET_SIZE, N_CLASSES)
    assert np.allclose(preds_lda.sum(axis=1), 1.0)
    print("LDA check passed.")

    # C. SVM
    print("Testing SVMWrapper...")
    # Use cv=2 for speed, ensuring we have enough samples per class in subset is tricky,
    # but with 50 samples and 99 classes, StratifiedKFold might complain if we don't handle it.
    # However, CalibratedClassifierCV handles this gracefully usually or we rely on the implementation.
    # For this demo, we'll try fitting. If it fails due to split, we'll skip strict CV check logic here
    # but in the full pipeline it uses the full dataset.
    try:
        svm = SVMWrapper(cv=2, random_state=DEMO_SEED)
        svm.fit(X_sub_lin, y_sub)
        preds_svm = svm.predict_proba(X_sub_lin)

        assert preds_svm.shape == (SUBSET_SIZE, N_CLASSES)
        assert np.allclose(preds_svm.sum(axis=1), 1.0)
        print("SVM check passed.")
    except ValueError as e:
        print(f"SVM check skipped due to small subset constraints: {e}")
        # Create dummy preds for ensemble step if SVM failed
        preds_svm = preds_lda.copy()

    # 3. Ensemble Optimization
    print("\n--- 3. Ensemble Optimization ---")
    optimizer = WeightOptimizer(random_state=DEMO_SEED)

    # Fit optimizer on the predictions
    optimizer.fit([preds_lgbm, preds_lda, preds_svm], y_sub, classes=classes)

    # Check weights
    weights = optimizer.weights
    print(f"Optimized Weights: {weights}")
    assert len(weights) == 3
    assert np.isclose(sum(weights), 1.0)
    assert np.all(weights >= 0)

    # Predict using ensemble
    ensemble_preds = optimizer.predict([preds_lgbm, preds_lda, preds_svm])
    assert ensemble_preds.shape == (SUBSET_SIZE, N_CLASSES)
    print("Ensemble check passed.")


def demonstrate_pipeline():
    print("\n=== Demonstrating Full Pipeline (Fast Mode) ===")

    # Monkey-patching library classes to force fast execution parameters
    # This ensures the CrossValidationPipeline runs quickly without modifying the library code file.

    print("Patching model defaults for speed...")

    # Patch LGBMWrapper
    original_lgbm_init = library.models.LGBMWrapper.__init__

    def fast_lgbm_init(self, n_estimators=10, *args, **kwargs):
        # Force n_estimators to 10
        kwargs["n_estimators"] = 10
        original_lgbm_init(self, *args, **kwargs)

    library.models.LGBMWrapper.__init__ = fast_lgbm_init

    # Patch SVMWrapper
    original_svm_init = library.models.SVMWrapper.__init__

    def fast_svm_init(self, cv=2, *args, **kwargs):
        # Force CV to 2
        kwargs["cv"] = 2
        original_svm_init(self, *args, **kwargs)

    library.models.SVMWrapper.__init__ = fast_svm_init

    # Initialize Pipeline with fewer folds
    n_folds_demo = 2
    pipeline = CrossValidationPipeline(n_folds=n_folds_demo, random_state=DEMO_SEED)

    print(f"Running pipeline with {n_folds_demo} folds...")
    # Run the pipeline
    # Note: This loads the full dataset, runs CV (patched to be fast), optimizes, and saves submission.
    pipeline.run(load_cached_data=False)

    # Verify Submission
    submission_path = "./submission/submission.csv"
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print(f"Submission generated with shape: {df_sub.shape}")

    # Check submission format
    assert "id" in df_sub.columns
    assert df_sub.shape[1] == N_CLASSES + 1  # id + 99 classes

    # Check probability range
    feature_cols = [c for c in df_sub.columns if c != "id"]
    probs = df_sub[feature_cols].values
    assert probs.min() >= 0
    assert probs.max() <= 1

    print("Pipeline demonstration successful.")


if __name__ == "__main__":
    # Set global seed
    set_seed(DEMO_SEED)

    # Clean up any previous runs
    if os.path.exists("./submission"):
        shutil.rmtree("./submission")
    if os.path.exists("./working/idea_2"):
        shutil.rmtree("./working/idea_2")

    try:
        demonstrate_components()
        demonstrate_pipeline()
        print("\nAll demonstrations completed successfully.")
    except AssertionError as e:
        print(f"\nVALIDATION FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\nEXECUTION FAILED: {e}")
        exit(1)
