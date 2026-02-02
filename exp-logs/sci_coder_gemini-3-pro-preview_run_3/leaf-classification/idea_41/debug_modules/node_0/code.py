import os
import shutil
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import seed_everything, setup_logger
from library.feature_extraction import run_feature_extraction
from library.data_processor import DataProcessor
from library.model_pipeline import train_fold, generate_submission


def main():
    # ==========================================
    # 1. Setup & Configuration Override
    # ==========================================
    print(">>> Setting up demo configuration...")
    seed_everything(42)

    # Define demo directories
    DEMO_WORKING_DIR = "./working/demo_run"
    DEMO_METADATA_DIR = os.path.join(DEMO_WORKING_DIR, "metadata")
    DEMO_MODELS_DIR = os.path.join(DEMO_WORKING_DIR, "models")
    DEMO_SUBMISSION_DIR = os.path.join(DEMO_WORKING_DIR, "submission")

    # Clean up previous demo run if exists
    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)

    os.makedirs(DEMO_METADATA_DIR, exist_ok=True)
    os.makedirs(DEMO_MODELS_DIR, exist_ok=True)
    os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)

    # Override Config paths to use our demo environment
    Config.WORKING_DIR = DEMO_WORKING_DIR
    Config.MODELS_DIR = DEMO_MODELS_DIR
    Config.SUBMISSION_DIR = DEMO_SUBMISSION_DIR

    Config.CACHE_TRAIN_DINO = os.path.join(DEMO_WORKING_DIR, "train_dino.npy")
    Config.CACHE_TRAIN_CONVNEXT = os.path.join(DEMO_WORKING_DIR, "train_convnext.npy")
    Config.CACHE_TRAIN_IDS = os.path.join(DEMO_WORKING_DIR, "train_ids.npy")
    Config.CACHE_TRAIN_LABELS = os.path.join(DEMO_WORKING_DIR, "train_labels.npy")
    Config.CACHE_TRAIN_TABULAR = os.path.join(DEMO_WORKING_DIR, "train_tab.npy")

    Config.CACHE_TEST_DINO = os.path.join(DEMO_WORKING_DIR, "test_dino.npy")
    Config.CACHE_TEST_CONVNEXT = os.path.join(DEMO_WORKING_DIR, "test_convnext.npy")
    Config.CACHE_TEST_IDS = os.path.join(DEMO_WORKING_DIR, "test_ids.npy")
    Config.CACHE_TEST_TABULAR = os.path.join(DEMO_WORKING_DIR, "test_tab.npy")

    # ==========================================
    # 2. Prepare Data Subset (Speed Optimization)
    # ==========================================
    print(">>> Preparing data subset...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Take a tiny subset (e.g., 4 samples each) to ensure execution < 1 hour
    # We combine train and val for the demo "train" set
    demo_train = pd.concat([orig_train.head(4), orig_val.head(2)], ignore_index=True)
    demo_test = orig_test.head(4)

    # We also need a demo val set for the metadata paths, though we might not strictly use it in the pipeline call
    demo_val = orig_val.iloc[2:4].reset_index(drop=True)

    # Save demo metadata
    demo_train_path = os.path.join(DEMO_METADATA_DIR, "train.csv")
    demo_val_path = os.path.join(DEMO_METADATA_DIR, "val.csv")
    demo_test_path = os.path.join(DEMO_METADATA_DIR, "test.csv")

    demo_train.to_csv(demo_train_path, index=False)
    demo_val.to_csv(demo_val_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    # Update Config to point to demo metadata
    Config.TRAIN_METADATA_PATH = demo_train_path
    Config.VAL_METADATA_PATH = demo_val_path
    Config.TEST_METADATA_PATH = demo_test_path

    print(
        f"Subset created: Train={len(demo_train)+len(demo_val)}, Test={len(demo_test)}"
    )

    # ==========================================
    # 3. Feature Extraction
    # ==========================================
    print(">>> Running Feature Extraction (Dual-Stream)...")

    # Force reload to demonstrate extraction logic
    raw_features = run_feature_extraction(load_cached_data=False)

    # Validation: Check Raw Feature Shapes
    # Expected: (N_samples, 12_views, Feature_Dim)
    n_train_total = len(demo_train) + len(demo_val)
    n_test_total = len(demo_test)

    assert raw_features["train_dino"].shape[0] == n_train_total
    assert raw_features["train_dino"].shape[1] == 12
    assert raw_features["test_convnext"].shape[0] == n_test_total
    assert raw_features["test_convnext"].shape[1] == 12

    print("Feature extraction successful. Shapes verified.")

    # ==========================================
    # 4. Data Processing (Densification)
    # ==========================================
    print(">>> Running Data Processing (Densification)...")

    processor = DataProcessor()

    # Process Train
    # Note: process_train_data saves to disk. We set load_cached_data=False to force processing.
    train_dataset = processor.process_train_data(raw_features, load_cached_data=False)

    # Process Test
    test_dataset = processor.process_test_data(raw_features, load_cached_data=False)

    # Validation: Check Densified Shapes
    # Each image produces 3 orthogonal centroids. So N -> N*3
    assert len(train_dataset) == n_train_total * 3
    assert len(test_dataset) == n_test_total * 3

    # Check that IDs are repeated correctly (A, B, C views)
    # The first 3 IDs should be identical
    assert train_dataset.ids[0] == train_dataset.ids[1] == train_dataset.ids[2]

    print(f"Data densification successful. Train samples: {len(train_dataset)}")

    # ==========================================
    # 5. Model Pipeline Training
    # ==========================================
    print(">>> Training Model Pipeline...")

    # For demonstration, we'll use the densified train_dataset as both train and validation
    # In a real scenario, we would split based on unique IDs using processor.get_stratified_folds

    pipeline, metrics = train_fold(
        dataset_train=train_dataset,
        dataset_val=train_dataset,  # Using same set for demo speed
        fold_idx=0,
        save_dir=Config.MODELS_DIR,
    )

    # Validation: Check Metrics
    assert "accuracy" in metrics
    assert "loss" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

    print(f"Training successful. Accuracy: {metrics['accuracy']:.4f}")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print(">>> Generating Submission...")

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Generate submission using the single trained model
    df_sub = generate_submission(
        models=[pipeline], test_dataset=test_dataset, output_path=submission_path
    )

    # Validation: Check Submission Format
    assert "id" in df_sub.columns
    assert len(df_sub) == n_test_total

    # Check probabilities range
    prob_cols = [c for c in df_sub.columns if c != "id"]
    probs = df_sub[prob_cols].values
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # Check that probabilities are clipped (min value should be approx 1e-15)
    assert np.isclose(probs.min(), 1e-15) or probs.min() > 1e-15

    print("Submission generated successfully.")
    print(df_sub.head())

    print("\n>>> Demo completed successfully.")


if __name__ == "__main__":
    main()
