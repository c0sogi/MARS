import os
import sys
import numpy as np
import pandas as pd
import shutil
import joblib
from sklearn.metrics import roc_auc_score

# Import library modules
from library import config
from library import utils

# Imports moved to main() to prevent file locking during cleanup


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Cleanup
    # Ensure we are working in a clean state for the demo
    utils.set_seed(config.SEED)

    # Clean up the specific working directory for this idea to ensure fresh run
    if os.path.exists(config.WORKING_DIR):
        print(f"Cleaning working directory: {config.WORKING_DIR}")
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Import modules after cleanup to avoid holding file handles (loggers) open
    from library import data_loader
    from library import embedding_manager
    from library import pipeline_factory
    from library import trainer
    from library import inference

    print(f"Working Directory: {config.WORKING_DIR}")
    print(f"Debug Mode: True (Sample Size: {config.DEBUG_SAMPLE_SIZE})")

    # 2. Data Loading Demonstration
    print("\n[Step 1] Loading Data (Debug Mode)...")
    # We force load_cached_data=False to demonstrate raw loading logic
    train_df, val_df, test_df = data_loader.load_dataset(
        load_cached_data=False, debug_mode=True
    )

    # Assertions to verify data loading
    assert (
        len(train_df) == config.DEBUG_SAMPLE_SIZE
    ), f"Train size mismatch: {len(train_df)}"
    assert len(val_df) == config.DEBUG_SAMPLE_SIZE, f"Val size mismatch: {len(val_df)}"
    assert (
        len(test_df) == config.DEBUG_SAMPLE_SIZE
    ), f"Test size mismatch: {len(test_df)}"
    assert config.TARGET_COL in train_df.columns, "Target column missing in Train"
    assert (
        "text_concat" in train_df.columns
    ), "Text preprocessing failed (text_concat missing)"
    print("Data loading verified successfully.")

    # 3. Embedding Generation Demonstration
    print("\n[Step 2] Generating Embeddings...")
    # This will compute embeddings for the small debug subset
    embeddings = embedding_manager.get_embeddings(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Verify Embedding Shapes
    # Expected: (N_samples, Embedding_Dim)
    # MiniLM dim = 384, MPNet dim = 768
    n_train = len(train_df)

    assert embeddings["train_title_emb"].shape == (
        n_train,
        384,
    ), "Train Title Embedding shape mismatch"
    assert embeddings["train_body_emb"].shape == (
        n_train,
        384,
    ), "Train Body Embedding shape mismatch"
    assert embeddings["train_global_emb"].shape == (
        n_train,
        768,
    ), "Train Global Embedding shape mismatch"
    print("Embedding generation verified successfully.")

    # 4. Pipeline Logic Verification (Unit Test)
    print("\n[Step 3] Verifying Model Pipeline Logic...")
    # We create a synthetic batch to test the pipeline factory and custom transformers
    # Dimensions: Title(384) + Body(384) + Global(768) + Meta(10) = 1546
    meta_dim = len(config.NUMERIC_FEATURES)
    total_dim = 384 + 384 + 768 + meta_dim

    fake_X = np.random.rand(50, total_dim).astype(np.float32)
    fake_y = np.random.randint(0, 2, 50)

    pipeline = pipeline_factory.create_model_pipeline(meta_dim=meta_dim)

    # Fit pipeline
    pipeline.fit(fake_X, fake_y)

    # Predict
    preds = pipeline.predict_proba(fake_X)

    assert preds.shape == (50, 2), "Prediction shape mismatch"
    assert np.all((preds >= 0) & (preds <= 1)), "Probabilities out of bounds"
    print("Pipeline logic verified successfully.")

    # 5. Full Training Loop Demonstration
    print("\n[Step 4] Running Cross-Validation Training (Debug Mode)...")
    # This runs the trainer.py logic: loads data, builds matrices, runs CV, saves models
    trainer.run_cv_training(load_cached_data=True, debug_mode=True)

    # Verify output models
    models_dir = os.path.join(config.WORKING_DIR, "models")
    expected_models = [f"model_fold_{i}.joblib" for i in range(config.N_FOLDS)]
    for model_file in expected_models:
        path = os.path.join(models_dir, model_file)
        assert os.path.exists(path), f"Model file missing: {path}"
    print(f"All {config.N_FOLDS} fold models created successfully.")

    # 6. Inference Demonstration
    print("\n[Step 5] Running Inference and Submission Generation...")
    # This runs inference.py logic: loads test data, loads models, predicts, averages, saves csv
    inference.generate_submission(load_cached_data=True, debug_mode=True)

    # Verify submission file
    submission_path = config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file was not created"

    sub_df = pd.read_csv(submission_path)
    assert sub_df.shape == (
        config.DEBUG_SAMPLE_SIZE,
        2,
    ), f"Submission shape mismatch: {sub_df.shape}"
    assert list(sub_df.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Submission columns mismatch"
    assert (
        sub_df["requester_received_pizza"].between(0, 1).all()
    ), "Submission probabilities invalid"

    print("Submission file verified successfully.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
