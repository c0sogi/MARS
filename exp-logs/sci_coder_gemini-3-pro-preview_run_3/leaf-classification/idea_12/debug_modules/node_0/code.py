import os
import sys
import pandas as pd
import numpy as np
import torch

# Import from the provided library
import library.config as config
import library.utils as utils
import library.feature_extractor as fe_module
import library.data_loader as dl_module
import library.modeling as model_module


def main():
    print("Initializing Demo Script...")

    # 1. Setup and Reproducibility
    utils.seed_everything(config.SEED)

    # 2. Create a lightweight subset of data for demonstration purposes
    # We want to avoid processing the full 700+ images to keep runtime short.
    print("Creating lightweight data subsets...")

    if not os.path.exists(config.TRAIN_METADATA_PATH) or not os.path.exists(
        config.TEST_METADATA_PATH
    ):
        raise FileNotFoundError(
            "Original metadata files not found. Ensure ./metadata/train.csv exists."
        )

    full_train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    full_test_df = pd.read_csv(config.TEST_METADATA_PATH)

    # Select a subset of classes to ensure stratification works even with small data
    # We pick 3 classes and take 5 samples each for training
    selected_classes = full_train_df["species"].unique()[:3]
    demo_train_df = (
        full_train_df[full_train_df["species"].isin(selected_classes)]
        .groupby("species")
        .head(5)
        .reset_index(drop=True)
    )

    # For test, just take 5 random rows
    demo_test_df = full_test_df.head(5).reset_index(drop=True)

    # Save these demo metadata files to working directory
    demo_train_path = os.path.join(config.WORKING_DIR, "demo_train.csv")
    demo_test_path = os.path.join(config.WORKING_DIR, "demo_test.csv")

    demo_train_df.to_csv(demo_train_path, index=False)
    demo_test_df.to_csv(demo_test_path, index=False)

    print(f"Demo Train Shape: {demo_train_df.shape}")
    print(f"Demo Test Shape: {demo_test_df.shape}")

    # 3. Patch Configuration to use Demo Data and Reduced Splits
    # Since the modules import constants using 'from library.config import ...',
    # we need to patch the variables in the destination modules directly to affect their logic.

    # Patch paths in config (used by some modules)
    config.TRAIN_METADATA_PATH = demo_train_path
    config.TEST_METADATA_PATH = demo_test_path

    # Patch N_SPLITS to 2 for speed (minimum for CV)
    DEMO_N_SPLITS = 2
    config.N_SPLITS = DEMO_N_SPLITS
    dl_module.N_SPLITS = DEMO_N_SPLITS
    model_module.N_SPLITS = DEMO_N_SPLITS

    # Patch Metadata paths in data_loader module (it imports them directly)
    dl_module.TRAIN_METADATA_PATH = demo_train_path
    dl_module.TEST_METADATA_PATH = demo_test_path

    # Patch Submission Path to avoid overwriting real submission
    demo_submission_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    config.SUBMISSION_PATH = demo_submission_path
    model_module.SUBMISSION_PATH = demo_submission_path

    # 4. Demonstrate Feature Extractor
    print("\n--- Testing FeatureExtractor ---")
    feature_extractor = fe_module.FeatureExtractor()

    # We force reload (load_cached_data=False) to ensure the extraction logic runs on our new subset
    print("Extracting features for demo train set...")
    img_feats, tab_feats, ids, labels = feature_extractor.extract_features(
        demo_train_df, split_name="demo_train", load_cached_data=False
    )

    # Verification
    # Image features: (N, 4, D).
    N = len(demo_train_df)
    assert (
        img_feats.shape[0] == N
    ), f"Expected {N} image feature rows, got {img_feats.shape[0]}"
    assert (
        img_feats.shape[1] == 4
    ), f"Expected 4 views per image, got {img_feats.shape[1]}"
    assert tab_feats.shape == (
        N,
        192,
    ), f"Expected ({N}, 192) tabular features, got {tab_feats.shape}"
    assert len(ids) == N
    assert len(labels) == N
    print("Feature Extractor output shapes verified.")

    # 5. Demonstrate Data Manager
    print("\n--- Testing LeafDataManager ---")
    data_manager = dl_module.LeafDataManager(feature_extractor)

    # Setup data (loads metadata, extracts features for both train and test)
    # This will use the cache we just generated for train, and compute for test
    data_manager.setup_data(load_cached_data=True)

    # Test get_fold_data (Fold 0)
    print("Retrieving Fold 0 data...")
    X_train, y_train, X_val, y_val = data_manager.get_fold_data(
        fold_idx=0, load_cached_data=False
    )

    # Verification of Manifold Expansion (Train)
    # Train set should use Manifold Expansion (4 views per ID)
    # Val set should use Centroid Consolidation (1 view per ID)

    n_train_samples = len(y_train)
    n_val_samples = len(y_val)

    print(f"Fold 0 Train Samples: {n_train_samples}")
    print(f"Fold 0 Val Samples: {n_val_samples}")

    # Check dimensions
    # X_train: [Image(D) + Tabular(192)]
    feature_dim = X_train.shape[1]
    assert X_train.shape[0] == n_train_samples
    assert X_val.shape[0] == n_val_samples
    assert X_val.shape[1] == feature_dim

    # Assert we have data
    assert n_train_samples > 0
    assert n_val_samples > 0

    print("Data Manager fold retrieval verified.")

    # 6. Demonstrate Modeling Pipeline (End-to-End)
    print("\n--- Testing Modeling Pipeline (Run Training) ---")
    # This runs the cross-validation loop, trains the pipeline, predicts, and saves submission
    # Because we patched N_SPLITS to 2, this will be fast.
    model_module.run_training(data_manager)

    # 7. Validate Submission
    print("\n--- Validating Submission ---")
    if not os.path.exists(demo_submission_path):
        raise FileNotFoundError("Submission file was not generated.")

    df_sub = pd.read_csv(demo_submission_path)
    print(f"Submission shape: {df_sub.shape}")

    # Check rows: Should match demo_test_df count
    assert len(df_sub) == len(
        demo_test_df
    ), f"Expected {len(demo_test_df)} rows in submission, got {len(df_sub)}"

    # Check columns: id + classes
    # Note: data_manager.classes contains only the classes present in our demo subset (3 classes)
    # The submission should contain columns for these classes.
    expected_cols = ["id"] + list(data_manager.classes)
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns {list(df_sub.columns)} do not match expected classes {expected_cols}."

    # Check values are probabilities
    probs = df_sub.drop(columns=["id"]).values
    assert np.all(probs >= 0) and np.all(
        probs <= 1
    ), "Probabilities out of range [0, 1]"

    print("Submission content verified.")
    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()
