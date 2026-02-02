import os
import shutil
import numpy as np
import pandas as pd
import torch
import joblib

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, setup_logger
from library.feature_extraction import extract_dataset
from library.densification import prepare_training_data, prepare_inference_data
from library.modeling import build_pipeline
from library.training import run_training
from library.inference import generate_submission


def create_demo_data():
    """
    Creates a small, valid subset of data for demonstration purposes.
    We need enough samples per class to satisfy StratifiedKFold (n_splits=2).
    """
    print("Creating demo metadata...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Select 3 classes with at least 4 samples each to ensure 2-fold CV works
    # (2 samples for train, 2 for val per fold)
    selected_classes = orig_train["species"].value_counts()
    selected_classes = selected_classes[selected_classes >= 4].index[:3]

    demo_train_rows = []
    for cls in selected_classes:
        # Take 4 samples per class
        rows = orig_train[orig_train["species"] == cls].head(4)
        demo_train_rows.append(rows)

    demo_train = pd.concat(demo_train_rows).reset_index(drop=True)

    # Select a few test samples
    demo_test = orig_test.head(5).reset_index(drop=True)

    # Save to working directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    demo_train_path = os.path.join(Config.WORKING_DIR, "demo_train.csv")
    demo_test_path = os.path.join(Config.WORKING_DIR, "demo_test.csv")

    demo_train.to_csv(demo_train_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    print(
        f"Created demo train set: {len(demo_train)} samples ({len(selected_classes)} classes)"
    )
    print(f"Created demo test set: {len(demo_test)} samples")

    return demo_train_path, demo_test_path


def update_config(demo_train_path, demo_test_path):
    """
    Updates the global Config object to use the demo data and directories.
    """
    print("Updating configuration...")

    # Use a specific subdirectory for this run
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update Metadata Paths
    Config.TRAIN_METADATA_PATH = demo_train_path
    Config.TEST_METADATA_PATH = demo_test_path

    # Update Cache Paths to point to the new working directory
    Config.CACHE_TRAIN_IMG_FEATURES = os.path.join(
        Config.WORKING_DIR, "train_img_features.npy"
    )
    Config.CACHE_TRAIN_TAB_FEATURES = os.path.join(
        Config.WORKING_DIR, "train_tab_features.npy"
    )
    Config.CACHE_TRAIN_IDS = os.path.join(Config.WORKING_DIR, "train_ids.npy")
    Config.CACHE_TRAIN_LABELS = os.path.join(Config.WORKING_DIR, "train_labels.npy")

    Config.CACHE_TEST_IMG_FEATURES = os.path.join(
        Config.WORKING_DIR, "test_img_features.npy"
    )
    Config.CACHE_TEST_TAB_FEATURES = os.path.join(
        Config.WORKING_DIR, "test_tab_features.npy"
    )
    Config.CACHE_TEST_IDS = os.path.join(Config.WORKING_DIR, "test_ids.npy")

    # Update Model Parameters for Speed
    Config.N_FOLDS = 2
    Config.DEBUG = (
        False  # We control data size via the CSVs, so we disable internal debug slicing
    )

    # Ensure submission directory exists
    Config.SUBMISSION_DIR = Config.WORKING_DIR
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission_demo.csv")


def main():
    # 0. Setup
    seed_everything(42)
    logger = setup_logger("demo_script")

    # Create demo data and update config
    # We use a temporary location for the csvs, but point Config to them
    temp_dir = "./working/demo_data"
    os.makedirs(temp_dir, exist_ok=True)
    Config.WORKING_DIR = temp_dir  # Temporarily set for create_demo_data helper

    train_csv, test_csv = create_demo_data()
    update_config(train_csv, test_csv)

    print("\n" + "=" * 40)
    print("1. DEMONSTRATING FEATURE EXTRACTION")
    print("=" * 40)

    # Extract Train Features
    # load_cached_data=False forces the extraction process to run
    print("Extracting training features (DINOv2 + ConvNeXt)...")
    img_feats, tab_feats, ids, labels = extract_dataset("train", load_cached_data=False)

    # Verify Shapes
    # Image features: (N, 12 views, 1024+1536 dim)
    expected_dim = 1024 + 1536
    print(f"Extracted Image Features Shape: {img_feats.shape}")
    print(f"Extracted Tabular Features Shape: {tab_feats.shape}")

    assert img_feats.ndim == 3, "Image features should be 3D (N, Views, Dim)"
    assert img_feats.shape[1] == 12, "Should have 12 views per image"
    assert (
        img_feats.shape[2] == expected_dim
    ), f"Feature dimension should be {expected_dim}"
    assert tab_feats.shape[1] == 192, "Tabular features should have 192 columns"
    assert len(ids) == len(labels) == len(img_feats), "Sample counts mismatch"

    print("Feature extraction verification passed.")

    print("\n" + "=" * 40)
    print("2. DEMONSTRATING DENSIFICATION")
    print("=" * 40)

    # Prepare Training Data (Convex Hull - 6x Expansion)
    print("Densifying training data (6x expansion)...")
    X_img_train, X_tab_train, y_train, ids_train = prepare_training_data(
        img_feats,
        tab_feats,
        ids,
        labels,
        cache_suffix="demo_train",
        load_cached_data=False,
    )

    n_samples = len(ids)
    print(f"Original samples: {n_samples}")
    print(f"Densified samples: {len(y_train)}")

    assert (
        len(y_train) == n_samples * 6
    ), "Training densification should increase samples by 6x"
    assert X_img_train.ndim == 2, "Densified image features should be flattened to 2D"

    # Prepare Inference Data (Canonical Centroids - 3x Expansion)
    print("Preparing inference data (3x expansion)...")
    X_img_inf, X_tab_inf, ids_inf, y_inf = prepare_inference_data(
        img_feats,
        tab_feats,
        ids,
        labels,
        cache_suffix="demo_inf",
        load_cached_data=False,
    )

    assert (
        len(ids_inf) == n_samples * 3
    ), "Inference densification should increase samples by 3x"

    print("Densification verification passed.")

    print("\n" + "=" * 40)
    print("3. DEMONSTRATING MODELING")
    print("=" * 40)

    print("Building pipeline...")
    pipeline = build_pipeline()

    # Concatenate visual and tabular for the pipeline
    X_train_full = np.hstack([X_img_train, X_tab_train])

    print(f"Fitting pipeline on shape {X_train_full.shape}...")
    pipeline.fit(X_train_full, y_train)

    print("Classes found:", pipeline.classes_)
    assert (
        len(pipeline.classes_) == 3
    ), "Should have detected 3 classes from our demo subset"

    # Test prediction
    X_inf_full = np.hstack([X_img_inf, X_tab_inf])
    preds = pipeline.predict_proba(X_inf_full)
    assert preds.shape == (len(X_inf_full), 3), "Prediction shape mismatch"

    print("Modeling verification passed.")

    print("\n" + "=" * 40)
    print("4. DEMONSTRATING FULL TRAINING LOOP")
    print("=" * 40)

    # This runs the full Stratified K-Fold process
    # It will use the 'train' features we already cached in step 1 (since paths match)
    # providing load_cache=True
    print(f"Running training with {Config.N_FOLDS} folds...")
    scores = run_training(debug=False, load_cache=True, n_folds=Config.N_FOLDS)

    print("Fold Scores (Log Loss):", scores)
    assert len(scores) == Config.N_FOLDS, "Should return a score for each fold"

    # Verify models were saved
    models_dir = os.path.join(Config.WORKING_DIR, "models")
    saved_models = os.listdir(models_dir)
    print(f"Saved files in {models_dir}: {saved_models}")
    assert "classes.pkl" in saved_models, "classes.pkl should be saved"
    assert (
        f"pipeline_fold_{Config.N_FOLDS-1}.pkl" in saved_models
    ), "Last fold model should be saved"

    print("Training loop verification passed.")

    print("\n" + "=" * 40)
    print("5. DEMONSTRATING INFERENCE & SUBMISSION")
    print("=" * 40)

    # Generate submission for the test set
    # This will trigger extraction for the test set (since we haven't done it yet)
    print("Generating submission...")
    generate_submission(load_cached_features=False, load_cached_inference=False)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission Head:")
    print(df_sub.head())

    # Verify submission shape: (N_test, N_classes + 1 for ID)
    # We used 5 test samples
    expected_cols = 3 + 1  # 3 classes + 'id'
    assert len(df_sub) == 5, f"Expected 5 rows in submission, got {len(df_sub)}"
    assert (
        len(df_sub.columns) == expected_cols
    ), f"Expected {expected_cols} columns, got {len(df_sub.columns)}"

    print("Inference verification passed.")
    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    main()
