import os
import shutil
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.dataset import PetDataset
from library.extractors import FeatureExtractor
from library.preprocessor import FeaturePreprocessor
from library.ensemble import StackingEnsemble


def main():
    print("Starting Demo Execution...")

    # ==========================================
    # 1. Setup & Configuration Override
    # ==========================================
    # We modify the Config class attributes to optimize for a quick demo run.

    # Create a separate working directory for the demo
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Limit data size for speed (e.g., 20 samples)
    N_SAMPLES = 20

    # Load original metadata
    # We assume these exist as per the problem description
    df_train_orig = pd.read_csv("./metadata/train_meta.csv")
    df_val_orig = pd.read_csv("./metadata/val_meta.csv")
    df_test_orig = pd.read_csv("./metadata/test_meta.csv")

    # Create subsets
    df_train_sub = df_train_orig.head(N_SAMPLES).copy()
    df_val_sub = df_val_orig.head(N_SAMPLES).copy()
    df_test_sub = df_test_orig.head(N_SAMPLES).copy()

    # Save subsets to demo directory and update Config paths
    Config.TRAIN_META_PATH = os.path.join(DEMO_DIR, "train_meta.csv")
    Config.VAL_META_PATH = os.path.join(DEMO_DIR, "val_meta.csv")
    Config.TEST_META_PATH = os.path.join(DEMO_DIR, "test_meta.csv")

    df_train_sub.to_csv(Config.TRAIN_META_PATH, index=False)
    df_val_sub.to_csv(Config.VAL_META_PATH, index=False)
    df_test_sub.to_csv(Config.TEST_META_PATH, index=False)

    print(f"Created subset metadata with {N_SAMPLES} samples each.")

    # Reduce Model Complexity for Demo
    # Use only one backbone to save time.
    # We select the CLIP backbone from the original list.
    Config.BACKBONES = [
        {
            "name": "openai/clip-vit-large-patch14",
            "source": "transformers",
            "type": "clip",
        },
    ]

    # Reduce Ensemble parameters
    Config.N_FOLDS = 2
    Config.BATCH_SIZE = 4
    Config.LGBM_PARAMS["n_estimators"] = 5
    Config.EXTRATREES_PARAMS["n_estimators"] = 5
    Config.EARLY_STOPPING_ROUNDS = 5

    # Seed everything for reproducibility
    seed_everything(Config.SEED)

    # ==========================================
    # 2. Dataset & DataLoader Demonstration
    # ==========================================
    print("\n=== Dataset & DataLoader ===")

    # Instantiate Datasets
    # Train uses TTA (returns stack of 2 images), Val/Test do not
    train_dataset = PetDataset(df_train_sub, mode="train", tta=True)
    val_dataset = PetDataset(df_val_sub, mode="val", tta=False)
    test_dataset = PetDataset(df_test_sub, mode="test", tta=False)

    # Verify Train Item (TTA enabled -> Stack of 2 images)
    sample_train = train_dataset[0]
    assert "image" in sample_train
    assert "metadata" in sample_train
    assert "target" in sample_train
    # Shape: (2, 3, 224, 224)
    assert sample_train["image"].shape == (
        2,
        3,
        224,
        224,
    ), f"Train TTA shape mismatch: {sample_train['image'].shape}"
    assert sample_train["metadata"].shape == (
        12,
    ), f"Metadata shape mismatch: {sample_train['metadata'].shape}"

    # Verify Test Item (No Target, No TTA)
    sample_test = test_dataset[0]
    assert "target" not in sample_test
    # Shape: (3, 224, 224)
    assert sample_test["image"].shape == (
        3,
        224,
        224,
    ), f"Test shape mismatch: {sample_test['image'].shape}"

    # Create Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    print("Dataset verification passed.")

    # ==========================================
    # 3. Feature Extraction Demonstration
    # ==========================================
    print("\n=== Feature Extraction ===")

    extractor = FeatureExtractor()

    # Extract features (force re-computation by setting load_cached_data=False)
    # This uses the reduced BACKBONES list (only CLIP)
    raw_features = extractor.extract_all(
        train_loader, val_loader, test_loader, load_cached_data=False
    )

    # Verify output structure
    backbone_name = Config.BACKBONES[0]["name"]
    assert backbone_name in raw_features
    assert "train" in raw_features[backbone_name]
    assert "val" in raw_features[backbone_name]
    assert "test" in raw_features[backbone_name]

    # Check shape: (N_SAMPLES, Feature_Dim)
    # CLIP Large usually has 1024 dim. Dual pooling (Avg+Max) -> 2048.
    feat_shape = raw_features[backbone_name]["train"].shape
    print(f"Extracted feature shape: {feat_shape}")
    assert feat_shape[0] == N_SAMPLES

    print("Feature extraction verification passed.")

    # ==========================================
    # 4. Preprocessing Demonstration
    # ==========================================
    print("\n=== Preprocessing ===")

    preprocessor = FeaturePreprocessor()

    # Run preprocessing
    # This reads the metadata from the Config paths we updated earlier
    X_train, y_train, X_val, y_val, X_test, test_ids = preprocessor.preprocess(
        raw_features, load_cached_data=False
    )

    # Verify shapes
    print(f"Preprocessed X_train shape: {X_train.shape}")
    assert len(X_train) == N_SAMPLES
    assert len(y_train) == N_SAMPLES
    assert len(X_test) == N_SAMPLES
    assert len(test_ids) == N_SAMPLES

    # Verify that X contains concatenated features (PCA + Interactions + Metadata)
    assert X_train.shape[1] > 0

    print("Preprocessing verification passed.")

    # ==========================================
    # 5. Ensemble Training & Inference
    # ==========================================
    print("\n=== Ensemble Training ===")

    ensemble = StackingEnsemble()

    # Run the full pipeline
    # This combines train+val for CV, fits final models, and predicts on test
    ensemble.run(X_train, y_train, X_val, y_val, X_test, test_ids)

    # Verify Submission
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")
    print(df_sub.head())

    assert len(df_sub) == N_SAMPLES
    assert "Id" in df_sub.columns
    assert "Pawpularity" in df_sub.columns
    assert df_sub["Pawpularity"].isnull().sum() == 0

    print("Ensemble verification passed.")
    print("\nDemo Execution Completed Successfully.")


if __name__ == "__main__":
    main()
