import os
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
import shutil

# Import library modules
# We import config first to override settings before other modules use them
import library.config as config
import library.transforms as lib_transforms
import library.dataset as lib_dataset
import library.model_utils as lib_model
import library.embedding_manager as lib_emb
import library.classifier as lib_clf


def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_demo():
    print("=== Starting Library Verification and Demo ===")

    # 1. Configuration Overrides for Speed and Demo
    print("\n[1] Configuring environment...")
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 10  # Process only 10 images for real data tests
    config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution
    config.CLF_CV_FOLDS = 2  # Reduce CV folds for classifier demo

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    set_seeds(config.SEED)
    print("    Debug mode enabled. Sample size set to 10.")

    # 2. Verify Transforms
    print("\n[2] Verifying Transforms...")
    # Create a dummy PIL image (RGB)
    dummy_img = Image.new("RGB", (300, 300), color="red")

    # Test Global Transform
    t_global = lib_transforms.get_global_transform()
    out_global = t_global(dummy_img)
    assert out_global.shape == (
        3,
        224,
        224,
    ), f"Global transform shape mismatch: {out_global.shape}"

    # Test Standard Transform
    t_standard = lib_transforms.get_standard_transform()
    out_standard = t_standard(dummy_img)
    assert out_standard.shape == (
        3,
        224,
        224,
    ), f"Standard transform shape mismatch: {out_standard.shape}"

    # Test Local Transform (FiveCrop)
    t_local = lib_transforms.get_local_transform()
    out_local = t_local(dummy_img)
    # Expected: (5, 3, 224, 224)
    assert out_local.shape == (
        5,
        3,
        224,
        224,
    ), f"Local transform shape mismatch: {out_local.shape}"
    print("    All transforms produced expected output shapes.")

    # 3. Verify Dataset (using real metadata in debug mode)
    print("\n[3] Verifying DogDataset...")
    # We use the train metadata provided in the environment
    ds_train = lib_dataset.DogDataset(
        config.TRAIN_METADATA_PATH, transform_type="standard", debug=True
    )

    if len(ds_train) > 0:
        img_tensor, label = ds_train[0]
        assert isinstance(img_tensor, torch.Tensor), "Dataset item 0 is not a tensor"
        assert img_tensor.shape == (
            3,
            224,
            224,
        ), f"Dataset item shape mismatch: {img_tensor.shape}"
        assert isinstance(
            label, (int, np.integer)
        ), f"Label is not integer: {type(label)}"
        print(f"    Loaded train sample successfully. Label: {label}")
    else:
        print(
            "    Warning: Dataset is empty (check input directory). Skipping item check."
        )

    # 4. Verify Model (Feature Extractor)
    print("\n[4] Verifying ConvNeXtMultiLayerExtractor...")
    model = lib_model.build_feature_extractor()

    # Create a dummy batch: (Batch=2, Channels=3, H=224, W=224)
    dummy_batch = torch.randn(2, 3, 224, 224).to(config.DEVICE)

    with torch.no_grad():
        features = model(dummy_batch)

    # Check outputs
    # ConvNeXt Large Stage 3 (features.5) -> 1024 channels
    # ConvNeXt Large Stage 4 (features.7) -> 1536 channels
    assert "stage3" in features and "stage4" in features, "Missing feature keys"
    assert features["stage3"].shape == (
        2,
        1024,
    ), f"Stage 3 shape mismatch: {features['stage3'].shape}"
    assert features["stage4"].shape == (
        2,
        1536,
    ), f"Stage 4 shape mismatch: {features['stage4'].shape}"
    print("    Model forward pass successful. Feature shapes correct.")

    # 5. Verify Embedding Manager (Pipeline)
    print("\n[5] Verifying Embedding Manager (Real Data Pipeline)...")
    # This runs the full extraction pipeline on the debug subset (10 images)
    # It handles multi-view extraction and concatenation
    # We force load_cached_data=False to ensure the code runs
    embeddings, labels = lib_emb.get_dataset_embeddings(
        "train", load_cached_data=False, debug=True
    )

    # Expected embedding dimension:
    # (Global_S4 + Global_S3) + (Standard_S4 + Standard_S3) + (Local_S4 + Local_S3)
    # (1536 + 1024) * 3 = 2560 * 3 = 7680
    expected_dim = 7680
    assert (
        embeddings.shape[1] == expected_dim
    ), f"Embedding dim mismatch: {embeddings.shape[1]} != {expected_dim}"
    assert len(embeddings) == len(labels), "Embedding and label count mismatch"
    print(
        f"    Generated embeddings for {len(embeddings)} samples with dimension {embeddings.shape[1]}."
    )

    # 6. Verify Classifier (Integration Test with Synthetic Data)
    print("\n[6] Verifying Classifier Pipeline (Synthetic Data)...")

    # To properly test the classifier logic (which reads metadata to get breed names),
    # we create a dummy metadata file with a known small set of breeds.
    dummy_meta_path = os.path.join(config.WORKING_DIR, "dummy_train_metadata.csv")
    dummy_breeds = ["breed_A", "breed_B", "breed_C"]

    # Create dummy CSV
    df_dummy = pd.DataFrame(
        {
            "id": [f"id_{i}" for i in range(30)],
            "breed": np.random.choice(dummy_breeds, 30),
            "file_path": ["dummy.jpg"] * 30,
        }
    )
    df_dummy.to_csv(dummy_meta_path, index=False)

    # Override config to point to dummy metadata
    original_meta_path = config.TRAIN_METADATA_PATH
    config.TRAIN_METADATA_PATH = dummy_meta_path

    try:
        # Generate Synthetic Embeddings
        # 30 samples, 7680 features
        n_samples = 30
        n_features = 7680
        syn_train_emb = np.random.randn(n_samples, n_features).astype(np.float32)

        # Generate labels (0, 1, 2) corresponding to sorted dummy breeds
        # Map breed names to indices
        breed_to_idx = {b: i for i, b in enumerate(sorted(dummy_breeds))}
        syn_train_lbl = df_dummy["breed"].map(breed_to_idx).values

        # Synthetic Validation Data
        syn_val_emb = np.random.randn(10, n_features).astype(np.float32)
        syn_val_lbl = np.random.randint(0, 3, 10)

        # Synthetic Test Data
        syn_test_emb = np.random.randn(5, n_features).astype(np.float32)
        syn_test_ids = [f"test_{i}" for i in range(5)]

        # Run Pipeline
        # We set load_cached_model=False to force training
        print("    Training classifier on synthetic data...")
        loss = lib_clf.train_and_predict(
            train_embeddings=syn_train_emb,
            train_labels=syn_train_lbl,
            val_embeddings=syn_val_emb,
            val_labels=syn_val_lbl,
            test_embeddings=syn_test_emb,
            test_ids=syn_test_ids,
            load_cached_model=False,
        )

        print(f"    Classifier run complete. Log Loss: {loss:.4f}")

        # Verify Submission File
        if os.path.exists(config.SUBMISSION_PATH):
            sub_df = pd.read_csv(config.SUBMISSION_PATH)
            print(f"    Submission file created at {config.SUBMISSION_PATH}")
            print(f"    Submission shape: {sub_df.shape}")

            # Check columns: id + 3 breeds
            expected_cols = ["id"] + sorted(dummy_breeds)
            assert (
                list(sub_df.columns) == expected_cols
            ), f"Submission columns mismatch. Got {list(sub_df.columns)}"
            assert len(sub_df) == 5, "Submission row count mismatch"
        else:
            raise FileNotFoundError("Submission file was not created.")

    finally:
        # Restore config and cleanup
        config.TRAIN_METADATA_PATH = original_meta_path
        if os.path.exists(dummy_meta_path):
            os.remove(dummy_meta_path)
        print("    Cleanup complete.")

    print("\n=== All Demonstrations and Verifications Passed Successfully ===")


if __name__ == "__main__":
    run_demo()
