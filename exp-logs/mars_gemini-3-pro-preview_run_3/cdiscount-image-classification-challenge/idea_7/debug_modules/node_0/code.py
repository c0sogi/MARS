import os
import pandas as pd
import numpy as np
import torch
import timm

# Import library modules
# We import these to utilize the provided classes and functions.
import library.config
import library.dataset
import library.data_utils
from library.config import (
    TRAIN_BSON_PATH,
    TEST_BSON_PATH,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    DEVICE,
    SEED,
)
from library.feature_extractor import _process_split
from library.trainer import Trainer
from library.data_utils import seed_everything


def run_demo():
    # ==========================================
    # 1. SETUP & CONFIGURATION
    # ==========================================
    print("=== Setting up Demo Environment ===")

    # Define a separate directory for this demo to avoid conflicts with main training runs
    DEMO_DIR = "./working/demo_execution"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override working directories in library modules
    # This ensures cache files (hierarchy map, label targets) are saved in our demo dir
    library.config.WORKING_DIR = DEMO_DIR
    library.dataset.WORKING_DIR = DEMO_DIR
    library.data_utils.WORKING_DIR = DEMO_DIR

    seed_everything(SEED)

    # ==========================================
    # 2. CREATE MINI METADATA (SUBSETTING)
    # ==========================================
    print("\n=== Creating Mini Metadata Subsets ===")

    # We load the full metadata files (CSVs) and sample top N rows.
    # This simulates having a smaller dataset for quick verification.

    # Load Train Metadata
    df_train = pd.read_csv(TRAIN_META_PATH)
    mini_train = df_train.head(100).copy()  # 100 samples
    mini_train_path = os.path.join(DEMO_DIR, "mini_train.csv")
    mini_train.to_csv(mini_train_path, index=False)
    print(f"Created mini train metadata: {len(mini_train)} records")

    # Load Val Metadata
    df_val = pd.read_csv(VAL_META_PATH)
    mini_val = df_val.head(50).copy()  # 50 samples
    mini_val_path = os.path.join(DEMO_DIR, "mini_val.csv")
    mini_val.to_csv(mini_val_path, index=False)
    print(f"Created mini val metadata: {len(mini_val)} records")

    # Load Test Metadata
    df_test = pd.read_csv(TEST_META_PATH)
    mini_test = df_test.head(50).copy()  # 50 samples
    mini_test_path = os.path.join(DEMO_DIR, "mini_test.csv")
    mini_test.to_csv(mini_test_path, index=False)
    print(f"Created mini test metadata: {len(mini_test)} records")

    # ==========================================
    # 3. FEATURE EXTRACTION
    # ==========================================
    print("\n=== Extracting Features (EfficientNet-B0) ===")

    # Initialize the feature extractor model
    # We use the same model definition as in library/feature_extractor.py
    print("Loading EfficientNet-B0...")
    model = timm.create_model("tf_efficientnet_b0", pretrained=True, num_classes=0)
    model.eval()
    model.to(DEVICE)

    # Define output paths for features
    train_feat_path = os.path.join(DEMO_DIR, "train_features.npy")
    train_label_path = os.path.join(DEMO_DIR, "train_labels.npy")

    val_feat_path = os.path.join(DEMO_DIR, "val_features.npy")
    val_label_path = os.path.join(DEMO_DIR, "val_labels.npy")

    test_feat_path = os.path.join(DEMO_DIR, "test_features.npy")
    test_ids_path = os.path.join(DEMO_DIR, "test_ids.npy")

    # Extract Train
    # Note: We pass the mini metadata paths, but the original BSON paths.
    # The metadata contains offsets that point correctly into the large BSON files.
    print("Processing Mini Train...")
    _process_split(
        metadata_path=mini_train_path,
        bson_path=TRAIN_BSON_PATH,
        output_feat_path=train_feat_path,
        output_label_path=train_label_path,
        model=model,
        is_test=False,
    )

    # Extract Val
    print("Processing Mini Val...")
    _process_split(
        metadata_path=mini_val_path,
        bson_path=TRAIN_BSON_PATH,
        output_feat_path=val_feat_path,
        output_label_path=val_label_path,
        model=model,
        is_test=False,
    )

    # Extract Test
    print("Processing Mini Test...")
    _process_split(
        metadata_path=mini_test_path,
        bson_path=TEST_BSON_PATH,
        output_feat_path=test_feat_path,
        output_label_path=test_ids_path,
        model=model,
        is_test=True,
    )

    # Verification
    assert os.path.exists(train_feat_path), "Train features not saved"
    assert os.path.exists(test_feat_path), "Test features not saved"

    train_feats = np.load(train_feat_path)
    print(f"Verified Train Features Shape: {train_feats.shape}")
    assert train_feats.shape == (
        100,
        1280,
    ), f"Expected (100, 1280), got {train_feats.shape}"

    # ==========================================
    # 4. MODEL TRAINING
    # ==========================================
    print("\n=== Training Hierarchical MLP ===")

    model_save_path = os.path.join(DEMO_DIR, "demo_model.pth")

    # Initialize Trainer
    # We use a small batch size and few epochs for speed
    trainer = Trainer(
        train_features_path=train_feat_path,
        train_labels_path=train_label_path,
        val_features_path=val_feat_path,
        val_labels_path=val_label_path,
        model_save_path=model_save_path,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-4,
        device=DEVICE,
    )

    # Run Training
    # We expect the model to overfit quickly on this tiny data, which is fine for a demo.
    trainer.fit(epochs=2, patience=2)

    # Verify Model Checkpoint
    assert os.path.exists(model_save_path), "Model checkpoint not created"
    print("Model training complete and checkpoint verified.")

    # ==========================================
    # 5. INFERENCE & SUBMISSION
    # ==========================================
    print("\n=== Running Inference ===")

    submission_path = os.path.join(DEMO_DIR, "submission.csv")

    trainer.predict(
        test_features_path=test_feat_path,
        test_ids_path=test_ids_path,
        submission_path=submission_path,
    )

    # Verify Submission File
    assert os.path.exists(submission_path), "Submission file not created"

    sub_df = pd.read_csv(submission_path)
    print(f"Submission loaded. Shape: {sub_df.shape}")
    print("Head of submission:")
    print(sub_df.head())

    # Logic Checks
    assert len(sub_df) == 50, f"Expected 50 predictions, got {len(sub_df)}"
    assert list(sub_df.columns) == [
        "_id",
        "category_id",
    ], "Incorrect columns in submission"
    assert (
        sub_df["category_id"].dtype == int or sub_df["category_id"].dtype == np.int64
    ), "category_id must be integer"
    assert not sub_df.isnull().values.any(), "Submission contains null values"

    print("\n=== Demo Execution Successful ===")


if __name__ == "__main__":
    run_demo()
