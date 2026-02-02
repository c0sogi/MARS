import os
import shutil
import numpy as np
import pandas as pd
import warnings
import joblib

# Import library components
from library.utils import set_seed
from library.data_loader import load_dataset
from library.feature_extraction import TextEmbedder, MetadataExtractor
from library.knn_feature_generator import KNNFeatureAugmenter
from library.preprocessor import DataPreprocessor
from library.model_trainer import EnsembleTrainer, run_training_pipeline


def main():
    # 1. Setup Environment
    print("--- Setting up environment ---")
    set_seed(42)
    warnings.filterwarnings("ignore")

    # Define constants
    # We use 100 samples to ensure speed while satisfying k=50 requirements in the library
    DEBUG_SIZE = 100
    WORK_DIR = "./working/idea_9"
    SUBMISSION_DIR = "./submission"

    # Clean working directory to ensure we test computation logic from scratch
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)

    # 2. Test Data Loader
    print(f"\n--- Testing Data Loader (Size={DEBUG_SIZE}) ---")
    df_train, df_val, df_test = load_dataset(
        load_cached_data=False, debug_sample_size=DEBUG_SIZE
    )

    # Verify Data Loading
    assert (
        len(df_train) == DEBUG_SIZE
    ), f"Expected {DEBUG_SIZE} train samples, got {len(df_train)}"
    assert (
        len(df_val) == DEBUG_SIZE
    ), f"Expected {DEBUG_SIZE} val samples, got {len(df_val)}"
    assert (
        "requester_received_pizza" in df_train.columns
    ), "Target column missing in train"
    print("Data Loader verification passed.")

    # 3. Test Feature Extraction Components
    print("\n--- Testing Feature Extraction Components ---")

    # Test Text Embedder
    # Using a tiny subset for unit testing the class
    small_df = df_train.iloc[:10].copy()
    embedder = TextEmbedder()
    embeddings = embedder.transform(small_df)

    # Expected shape: (10, 384) for all-MiniLM-L6-v2
    assert embeddings.shape == (
        10,
        384,
    ), f"Expected embedding shape (10, 384), got {embeddings.shape}"
    print("TextEmbedder verification passed.")

    # Test Metadata Extractor
    meta_ext = MetadataExtractor()
    meta_feats = meta_ext.fit_transform(small_df)

    # Expected shape: (10, 9) based on numeric columns in library
    assert meta_feats.shape == (
        10,
        9,
    ), f"Expected metadata shape (10, 9), got {meta_feats.shape}"
    print("MetadataExtractor verification passed.")

    # 4. Test KNN Feature Generator
    print("\n--- Testing KNN Feature Generator ---")
    y_small = small_df["requester_received_pizza"].astype(int).values

    # Ensure we have at least 2 classes for KNN to work in this tiny sample
    if len(np.unique(y_small)) < 2:
        y_small[0], y_small[1] = 0, 1

    knn_aug = KNNFeatureAugmenter(k=5)  # Small k for unit test

    # Test OOF Generation
    oof_feats = knn_aug.generate_oof_features(embeddings, y_small)
    assert oof_feats.shape == (10, 1), "OOF feature shape mismatch"
    assert np.all(
        (oof_feats >= 0) & (oof_feats <= 1)
    ), "OOF probabilities out of bounds"

    # Test Fit and Transform
    knn_aug.fit(embeddings, y_small)
    trans_feats = knn_aug.transform(embeddings)
    assert trans_feats.shape == (10, 1), "Transform feature shape mismatch"
    print("KNNFeatureAugmenter verification passed.")

    # 5. Test Data Preprocessor (Integration)
    print("\n--- Testing Data Preprocessor (Integration) ---")
    # Using k=10 to be safe with DEBUG_SIZE=100
    preprocessor = DataPreprocessor(k_neighbors=10)

    X_train, y_train, X_val, y_val, X_test, test_ids = (
        preprocessor.process_and_load_data(
            load_cached_data=False, debug_sample_size=DEBUG_SIZE
        )
    )

    # Verify Assembled Shapes
    # Features = 384 (Text) + 1 (KNN) + 9 (Meta) = 394
    expected_feats = 394
    assert X_train.shape == (
        DEBUG_SIZE,
        expected_feats,
    ), f"X_train shape mismatch: {X_train.shape}"
    assert y_train.shape == (DEBUG_SIZE,), "y_train shape mismatch"
    assert X_test.shape == (DEBUG_SIZE, expected_feats), "X_test shape mismatch"
    print("Data Preprocessor verification passed.")

    # 6. Test Model Trainer
    print("\n--- Testing Model Trainer ---")
    trainer = EnsembleTrainer(random_state=42)

    # Use a single C value for speed
    c_grid = [0.1]

    best_model = trainer.tune_and_train(X_train, y_train, X_val, y_val, c_grid=c_grid)

    assert best_model is not None, "Model training failed"
    assert trainer.best_auc != -1.0, "Best AUC was not updated"

    # Test Prediction
    preds = trainer.predict(X_test)
    assert len(preds) == DEBUG_SIZE, "Prediction length mismatch"
    assert np.all((preds >= 0) & (preds <= 1)), "Predictions out of probability bounds"

    # Test Model Saving
    model_path = os.path.join(WORK_DIR, "test_model.joblib")
    trainer.save_model(model_path)
    assert os.path.exists(model_path), "Model file was not saved"
    print("Model Trainer verification passed.")

    # 7. Test Full Pipeline Execution
    print("\n--- Testing Full Pipeline Execution ---")
    # This runs the end-to-end pipeline provided in the library.
    # It will use the cache we just generated (since we are not clearing dir again),
    # verifying that the pipeline correctly picks up cached data and generates a submission.

    # Note: run_training_pipeline uses k=50 internally.
    # With DEBUG_SIZE=100, train set is 100. StratifiedKFold(5) -> train folds ~80 samples.
    # k=50 <= 80, so this is valid.

    run_training_pipeline(load_cached_data=True, debug_sample_size=DEBUG_SIZE)

    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file not found"

    sub_df = pd.read_csv(submission_path)
    assert len(sub_df) == DEBUG_SIZE, f"Submission rows {len(sub_df)} != {DEBUG_SIZE}"
    assert list(sub_df.columns) == [
        "request_id",
        "requester_received_pizza",
    ], "Submission columns mismatch"

    print("Full Pipeline verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
