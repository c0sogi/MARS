import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Ensure we can import from the library directory
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed
from library.data_loader import load_datasets
from library.text_pipeline import ProjectedTextEmbedder
from library.tabular_pipeline import InteractionMetadataProcessor
from library.model import create_bagged_ensemble
from library.training import Trainer


def main():
    print("Starting demonstration script...")

    # 1. Setup and Configuration for Speed
    print("1. Configuring environment for fast execution...")
    set_seed(42)

    # Override Config for faster demonstration
    # Reducing estimators and iterations to ensure quick run
    Config.BAGGING_N_ESTIMATORS = 5
    Config.LR_MAX_ITER = 100
    Config.LR_C = 1.0

    # We will use a separate working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"   Working directory set to: {Config.WORKING_DIR}")
    print(f"   Bagging Estimators: {Config.BAGGING_N_ESTIMATORS}")

    # 2. Verify Data Loading
    print("\n2. Verifying Data Loader...")
    # Load a small fraction to verify structure without processing everything yet
    df_train_sample, df_val_sample, df_test_sample = load_datasets(
        load_cached_data=False, subsample_frac=0.1
    )

    print(f"   Train sample shape: {df_train_sample.shape}")
    print(f"   Val sample shape: {df_val_sample.shape}")
    print(f"   Test sample shape: {df_test_sample.shape}")

    assert not df_train_sample.empty, "Train sample is empty"
    assert (
        "requester_received_pizza" in df_train_sample.columns
    ), "Target column missing in train"
    assert "request_text_edit_aware" in df_train_sample.columns, "Text column missing"

    # 3. Verify Text Pipeline
    print("\n3. Verifying Text Pipeline...")
    text_pipeline = ProjectedTextEmbedder()

    # Fit on the small sample
    print("   Fitting text pipeline on sample...")
    text_pipeline.fit(df_train_sample, load_cached_data=False)

    # Transform
    X_text_sample = text_pipeline.transform(
        df_train_sample, split="train_sample", load_cached_data=False
    )
    print(f"   Transformed text shape: {X_text_sample.shape}")

    # Assertions
    assert X_text_sample.shape[0] == len(
        df_train_sample
    ), "Text feature row count mismatch"
    assert (
        X_text_sample.shape[1] == Config.N_PCA_COMPONENTS
    ), f"Text feature dim mismatch, expected {Config.N_PCA_COMPONENTS}"
    assert np.all(np.isfinite(X_text_sample)), "Text features contain NaNs or Infs"

    # 4. Verify Tabular Pipeline
    print("\n4. Verifying Tabular Pipeline...")
    meta_pipeline = InteractionMetadataProcessor()

    # Fit on the small sample
    print("   Fitting tabular pipeline on sample...")
    meta_pipeline.fit(df_train_sample, load_cached_data=False)

    # Transform
    X_meta_sample = meta_pipeline.transform(
        df_train_sample, split="train_sample", load_cached_data=False
    )
    print(f"   Transformed meta shape: {X_meta_sample.shape}")

    # Assertions
    assert X_meta_sample.shape[0] == len(
        df_train_sample
    ), "Meta feature row count mismatch"
    # Dimension check: 9 original features.
    # PolynomialFeatures(degree=2, interaction_only=True, include_bias=False) produces:
    # 9 linear terms + (9*8)/2 = 36 interaction terms = 45 total features.
    expected_meta_dim = 45
    assert (
        X_meta_sample.shape[1] == expected_meta_dim
    ), f"Meta feature dim mismatch. Expected {expected_meta_dim}, got {X_meta_sample.shape[1]}"
    assert np.all(np.isfinite(X_meta_sample)), "Meta features contain NaNs or Infs"

    # 5. Verify Model Logic
    print("\n5. Verifying Model Logic...")
    # Combine features
    X_combined = np.hstack([X_text_sample, X_meta_sample])
    y_sample = df_train_sample["requester_received_pizza"].values

    print(f"   Combined feature shape: {X_combined.shape}")

    model = create_bagged_ensemble(
        n_estimators=Config.BAGGING_N_ESTIMATORS, max_iter=Config.LR_MAX_ITER
    )

    print("   Training model on sample...")
    model.fit(X_combined, y_sample)

    probs = model.predict_proba(X_combined)[:, 1]
    print(f"   Prediction shape: {probs.shape}")
    print(f"   Sample AUC (Training): {roc_auc_score(y_sample, probs):.4f}")

    assert len(probs) == len(y_sample), "Prediction length mismatch"
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of [0, 1] range"

    # 6. Verify Trainer (Integration Test)
    print("\n6. Verifying Trainer Class (Full Integration)...")
    # We will use the Trainer to run a quick cross-validation and then full training

    trainer = Trainer(
        load_cached_data=True
    )  # Use caching to speed up if previous steps cached anything

    # Run a quick 2-fold CV
    print("   Running 2-Fold Cross-Validation...")
    cv_score = trainer.cross_validate(n_splits=2)
    print(f"   CV Score: {cv_score:.4f}")
    assert 0.0 <= cv_score <= 1.0, "CV Score out of range"

    # 7. Generate Submission
    print("\n7. Generating Final Submission...")
    trainer.train_final(use_all_data=True)

    # 8. Verify Submission File
    print("\n8. Verifying Submission File...")
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file not found"

    df_sub = pd.read_csv(submission_path)
    print(f"   Submission shape: {df_sub.shape}")
    print(f"   Columns: {df_sub.columns.tolist()}")

    # Check against test metadata
    df_test_meta = pd.read_csv(Config.TEST_META_PATH)
    assert len(df_sub) == len(
        df_test_meta
    ), f"Submission row count {len(df_sub)} != Test set size {len(df_test_meta)}"
    assert "request_id" in df_sub.columns, "request_id column missing"
    assert (
        "requester_received_pizza" in df_sub.columns
    ), "requester_received_pizza column missing"

    # Check ID alignment
    sub_ids = sorted(df_sub["request_id"].tolist())
    meta_ids = sorted(df_test_meta["request_id"].tolist())
    assert sub_ids == meta_ids, "Request IDs in submission do not match test metadata"

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
