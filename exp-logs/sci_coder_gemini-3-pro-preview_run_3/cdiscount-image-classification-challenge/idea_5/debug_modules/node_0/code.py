import os
import struct
import pandas as pd
import numpy as np
import torch
import shutil

# 1. Patch Configuration for Demo Run
# We modify the config module variables to point to the example dataset and a temporary working directory.
# This allows us to run the full pipeline on a small subset of data (100 records) quickly.
from library import config

DEMO_DIR = "./working/demo_run"
if os.path.exists(DEMO_DIR):
    shutil.rmtree(DEMO_DIR)
os.makedirs(DEMO_DIR, exist_ok=True)

print(f"Setting up demo environment in {DEMO_DIR}...")

# Override paths
config.WORKING_DIR = DEMO_DIR
config.TRAIN_BSON_PATH = os.path.join(config.INPUT_DIR, "train_example.bson")
# We reuse train_example.bson as a mock test file for demonstration
config.TEST_BSON_PATH = os.path.join(config.INPUT_DIR, "train_example.bson")

# Override Metadata paths
config.TRAIN_META_PATH = os.path.join(DEMO_DIR, "train_meta.csv")
config.VAL_META_PATH = os.path.join(DEMO_DIR, "val_meta.csv")
config.TEST_META_PATH = os.path.join(DEMO_DIR, "test_meta.csv")

# Override Feature/Label paths
config.TRAIN_FEATURES_PATH = os.path.join(DEMO_DIR, "train_features.npy")
config.TRAIN_LABELS_L1_PATH = os.path.join(DEMO_DIR, "train_labels_l1.npy")
config.TRAIN_LABELS_L2_PATH = os.path.join(DEMO_DIR, "train_labels_l2.npy")
config.TRAIN_LABELS_L3_PATH = os.path.join(DEMO_DIR, "train_labels_l3.npy")

config.VAL_FEATURES_PATH = os.path.join(DEMO_DIR, "val_features.npy")
config.VAL_LABELS_L1_PATH = os.path.join(DEMO_DIR, "val_labels_l1.npy")
config.VAL_LABELS_L2_PATH = os.path.join(DEMO_DIR, "val_labels_l2.npy")
config.VAL_LABELS_L3_PATH = os.path.join(DEMO_DIR, "val_labels_l3.npy")

config.TEST_FEATURES_PATH = os.path.join(DEMO_DIR, "test_features.npy")
config.TEST_IDS_PATH = os.path.join(DEMO_DIR, "test_ids.npy")

config.CATEGORY_ENCODER_PATH = os.path.join(DEMO_DIR, "category_encoder.pkl")
config.SUBMISSION_DIR = DEMO_DIR

# Override Training Hyperparameters for speed
config.NUM_EPOCHS = 2
config.BATCH_SIZE = 16
config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

# Import library modules after patching config
from library import utils, dataset, feature_extractor, engine, model


# =============================================================================
# HELPER: Metadata Generation for Example File
# =============================================================================
def generate_demo_metadata():
    """
    Parses train_example.bson to generate metadata CSVs for train, val, and test splits.
    """
    print("Generating metadata for train_example.bson...")
    rows = []
    file_path = config.TRAIN_BSON_PATH

    with open(file_path, "rb") as f:
        offset = 0
        while True:
            # Read size
            size_bytes = f.read(4)
            if len(size_bytes) < 4:
                break
            total_size = struct.unpack("<i", size_bytes)[0]

            # Read document body
            f.seek(offset)
            doc_data = f.read(total_size)

            # Simple manual parsing to find _id and category_id
            # We use the provided utils.BSONReader logic implicitly or just scan bytes
            # For robustness, let's use a simplified scan similar to the provided metadata script
            p = 4
            _id = None
            category_id = None

            while p < len(doc_data) - 1:
                type_byte = doc_data[p]
                p += 1
                name_end = doc_data.find(b"\x00", p)
                name = doc_data[p:name_end].decode("utf-8", errors="ignore")
                p = name_end + 1

                if name == "_id":
                    if type_byte == 0x10:  # int32
                        _id = struct.unpack("<i", doc_data[p : p + 4])[0]
                        p += 4
                    elif type_byte == 0x12:  # int64
                        _id = struct.unpack("<q", doc_data[p : p + 8])[0]
                        p += 8
                    else:
                        # Skip value
                        pass
                elif name == "category_id":
                    if type_byte == 0x10:
                        category_id = struct.unpack("<i", doc_data[p : p + 4])[0]
                        p += 4
                    elif type_byte == 0x12:
                        category_id = struct.unpack("<q", doc_data[p : p + 8])[0]
                        p += 8
                    elif type_byte == 0x01:  # double
                        val = struct.unpack("<d", doc_data[p : p + 8])[0]
                        category_id = int(val)
                        p += 8
                else:
                    # Skip value based on type (simplified for speed, assuming standard structure)
                    # For this demo, we just need _id and category_id which usually appear early.
                    # If we don't find them, we might skip parsing the rest correctly without full BSON logic.
                    # However, let's rely on the fact that we just need to record the offset/length.
                    # Since we have the total_size, we can just stop if we found both.
                    pass

                if _id is not None and category_id is not None:
                    break

            # Fallback if parsing failed (shouldn't happen on valid BSON)
            if _id is None:
                # Just increment offset and continue
                offset += total_size
                continue

            rows.append(
                {
                    "_id": _id,
                    "bson_offset": offset,
                    "bson_length": total_size,
                    "file_path": "train_example.bson",  # Relative path logic in library uses this
                    "category_id": category_id,
                }
            )

            offset += total_size

    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} records.")

    # Split into Train (80) and Val (20)
    train_df = df.iloc[:80].copy()
    val_df = df.iloc[80:].copy()

    # Test metadata (same file, but drop category_id)
    test_df = df.iloc[:20].copy()  # Use first 20 as mock test
    test_df = test_df.drop(columns=["category_id"])

    train_df.to_csv(config.TRAIN_META_PATH, index=False)
    val_df.to_csv(config.VAL_META_PATH, index=False)
    test_df.to_csv(config.TEST_META_PATH, index=False)
    print("Metadata saved.")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Set seeds
    engine.set_seed(config.SEED)

    # 1. Generate Metadata
    generate_demo_metadata()

    # 2. Demonstrate HierarchyEncoder
    print("\n=== HierarchyEncoder Demo ===")
    encoder = utils.HierarchyEncoder()
    encoder.prepare(load_cached_data=False)

    # Verify encoder loaded categories
    print(
        f"Encoder loaded: L1={encoder.num_l1}, L2={encoder.num_l2}, L3={encoder.num_l3} classes"
    )
    assert encoder.num_l3 > 0, "Encoder failed to load L3 classes"

    # Test transformation
    sample_cat_id = encoder.l3_classes[0]
    l3, l2, l1 = encoder.transform([sample_cat_id])
    print(
        f"Transform Category {sample_cat_id} -> L3_idx:{l3[0]}, L2_idx:{l2[0]}, L1_idx:{l1[0]}"
    )

    # Test inverse transformation
    inv_cat_id = encoder.inverse_transform(l3)[0]
    assert inv_cat_id == sample_cat_id, "Inverse transform mismatch"
    print("Encoder verification passed.")

    # 3. Demonstrate BSON Reading & Dataset
    print("\n=== BSONInferenceDataset Demo ===")
    # Load metadata
    train_meta = pd.read_csv(config.TRAIN_META_PATH)

    # Initialize dataset
    ds = dataset.BSONInferenceDataset(train_meta, config.TRAIN_BSON_PATH)

    # Get one item
    img_tensor, prod_id = ds[0]
    print(f"Loaded Product ID: {prod_id}")
    print(f"Image Tensor Shape: {img_tensor.shape} (N_imgs, C, H, W)")

    assert img_tensor.ndim == 4, "Image tensor should be 4D"
    assert img_tensor.shape[1] == 3, "Image should have 3 channels"
    assert img_tensor.shape[2] == config.IMG_SIZE, "Image height mismatch"

    # 4. Demonstrate Feature Extraction
    print("\n=== FeatureExtractor Demo ===")
    extractor = feature_extractor.FeatureExtractor()

    # Run extraction for all splits (Train, Val, Test)
    # This uses the mock metadata and the example BSON file
    # It will run ResNet50 inference on the images
    extractor.run_all(load_cached_data=False)

    # Verify files exist
    assert os.path.exists(config.TRAIN_FEATURES_PATH), "Train features not created"
    assert os.path.exists(config.TRAIN_LABELS_L3_PATH), "Train labels not created"
    assert os.path.exists(config.TEST_FEATURES_PATH), "Test features not created"

    # Check feature shape
    feats = np.load(config.TRAIN_FEATURES_PATH)
    print(f"Generated Train Features Shape: {feats.shape}")
    assert (
        feats.shape[1] == config.INPUT_DIM
    ), f"Feature dim mismatch. Expected {config.INPUT_DIM}"

    # 5. Demonstrate Model Training
    print("\n=== Model Training Demo ===")
    # Initialize trainer logic manually via engine.fit() to show usage
    # engine.fit() initializes model, loads data, trains, and saves best model
    best_acc = engine.fit()

    print(f"Training completed. Best Validation Accuracy: {best_acc:.4f}")
    assert os.path.exists(
        os.path.join(config.WORKING_DIR, "best_model.pth")
    ), "Best model file not found"

    # 6. Demonstrate Inference / Submission
    print("\n=== Submission Generation Demo ===")
    engine.generate_submission()

    sub_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(sub_path), "Submission file not found"

    # Validate submission format
    sub_df = pd.read_csv(sub_path)
    print("Submission Head:")
    print(sub_df.head())

    assert (
        "_id" in sub_df.columns and "category_id" in sub_df.columns
    ), "Submission columns missing"
    assert (
        len(sub_df) == 20
    ), f"Expected 20 predictions (mock test size), got {len(sub_df)}"

    print("\nDemo completed successfully!")
