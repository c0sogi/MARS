import os
import shutil
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import seed_everything
from library.feature_extraction import FeatureExtractor
from library.data_processor import DataProcessor
from library.modeling import EnsembleTrainer


def main():
    # ==========================================
    # 1. Setup and Configuration Override
    # ==========================================
    print("Initializing Demonstration...")

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Override Config parameters for a fast demonstration
    Config.N_FOLDS = 2  # Reduce folds to 2 for speed

    # We will process a small subset of data
    DEMO_LIMIT = 12

    # Create a specific directory for demo outputs to keep things clean
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_execution")
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define model output directory
    MODEL_DIR = os.path.join(Config.CACHE_DIR, "models")

    print(f"Demo Configuration:")
    print(f"  Limit: {DEMO_LIMIT} samples")
    print(f"  Folds: {Config.N_FOLDS}")
    print(f"  Cache Dir: {Config.CACHE_DIR}")

    # ==========================================
    # 2. Feature Extraction Demonstration
    # ==========================================
    print("\n--- Step 1: Feature Extraction ---")

    extractor = FeatureExtractor()

    # Extract features for a subset of the training data
    # This uses the 'train.csv' metadata but limits processing to DEMO_LIMIT images
    dino_feats, conv_feats, train_ids = extractor.extract_and_cache(
        metadata_path=Config.TRAIN_METADATA,
        dataset_name="demo_train",
        load_cached_data=False,  # Force computation for demo
        limit=DEMO_LIMIT,
    )

    # Validation
    # Expected Shapes:
    # DINO: [N, 12, 1024]
    # ConvNeXt: [N, 12, 1536]
    # IDs: [N]
    print(f"Extracted DINO features shape: {dino_feats.shape}")
    print(f"Extracted ConvNeXt features shape: {conv_feats.shape}")

    assert (
        len(train_ids) == DEMO_LIMIT
    ), f"Expected {DEMO_LIMIT} IDs, got {len(train_ids)}"
    assert dino_feats.shape == (
        DEMO_LIMIT,
        12,
        1024,
    ), f"Unexpected DINO shape: {dino_feats.shape}"
    assert conv_feats.shape == (
        DEMO_LIMIT,
        12,
        1536,
    ), f"Unexpected ConvNeXt shape: {conv_feats.shape}"

    print("Feature Extraction Validation Passed.")

    # ==========================================
    # 3. Data Processing Demonstration
    # ==========================================
    print("\n--- Step 2: Data Processing (Densification) ---")

    processor = DataProcessor()

    # Prepare densified dataset
    # This combines visual centroids (3 per image) with replicated tabular data
    # Note: We pass dataset_name="train" so it knows to look up tabular data in train.csv
    # The processor filters the metadata based on the provided 'train_ids'.
    X_train, y_train, ids_densified = processor.prepare_densified_dataset(
        dataset_name="train",
        dino_features=dino_feats,
        conv_features=conv_feats,
        ids=train_ids,
        load_cached_data=False,
    )

    # Get column indices for the pipeline
    col_indices = processor.get_column_indices()

    # Validation
    # Expected Output Size: 3 * DEMO_LIMIT
    expected_rows = 3 * DEMO_LIMIT
    # Expected Feature Dim: 1024 (DINO) + 1536 (Conv) + 192 (Tabular) = 2752
    expected_cols = 1024 + 1536 + 192

    print(f"Densified X shape: {X_train.shape}")
    print(f"Densified y shape: {y_train.shape}")

    assert X_train.shape == (
        expected_rows,
        expected_cols,
    ), f"Expected shape ({expected_rows}, {expected_cols}), got {X_train.shape}"
    assert len(y_train) == expected_rows, "Label count mismatch"
    assert len(ids_densified) == expected_rows, "ID count mismatch"

    # Verify replication logic: First 3 IDs should be identical
    assert (
        ids_densified[0] == ids_densified[1] == ids_densified[2]
    ), "Densification ID replication failed"

    print("Data Processing Validation Passed.")

    # ==========================================
    # 4. Model Training Demonstration
    # ==========================================
    print("\n--- Step 3: Ensemble Training ---")

    trainer = EnsembleTrainer(model_dir=MODEL_DIR)

    # Train the ensemble
    trainer.train(X_train, y_train, ids_densified, col_indices)

    # Validation
    # Check if model files exist
    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(MODEL_DIR, f"pipeline_fold_{fold}.pkl")
        assert os.path.exists(model_path), f"Model file missing: {model_path}"

    classes_path = os.path.join(MODEL_DIR, "classes.pkl")
    assert os.path.exists(classes_path), "Classes file missing"

    print("Training Validation Passed.")

    # ==========================================
    # 5. Inference Demonstration
    # ==========================================
    print("\n--- Step 4: Inference ---")

    # For demonstration, we will use the training subset as our "test" set
    # In a real scenario, we would use features extracted from 'test.csv'

    # Note: We reuse X_train and ids_densified here.
    # The predict function handles the aggregation of the 3 centroids back to 1 prediction per image.
    unique_ids, probs, classes = trainer.predict(X_train, ids_densified, col_indices)

    # Validation
    print(f"Predictions shape: {probs.shape}")

    assert (
        len(unique_ids) == DEMO_LIMIT
    ), f"Expected {DEMO_LIMIT} unique predictions, got {len(unique_ids)}"
    assert probs.shape[1] == len(
        classes
    ), "Probability columns do not match number of classes"

    # Check probability range
    assert np.all(probs >= 0) and np.all(
        probs <= 1.0 + 1e-6
    ), "Probabilities out of range [0, 1]"

    print("Inference Validation Passed.")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print("\n--- Step 5: Submission Generation ---")

    # Create DataFrame
    submission_df = pd.DataFrame(probs, columns=classes)
    submission_df.insert(0, "id", unique_ids)

    # Save to demo directory
    submission_path = os.path.join(Config.CACHE_DIR, "submission_demo.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to: {submission_path}")
    print("First 3 rows of submission:")
    print(submission_df.head(3))

    # Final check against sample submission format
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)
    expected_cols = set(sample_sub.columns)
    generated_cols = set(submission_df.columns)

    # Note: The generated columns depend on the classes found in the training subset (DEMO_LIMIT).
    # Since we only trained on 12 images, we likely don't have all 99 classes.
    # In a real run, the training set covers all classes.
    # We just check that 'id' is present and data types are correct.
    assert "id" in submission_df.columns
    assert submission_df["id"].dtype == sample_sub["id"].dtype or np.issubdtype(
        submission_df["id"].dtype, np.integer
    )

    print("\nDemonstration Completed Successfully.")


if __name__ == "__main__":
    main()
