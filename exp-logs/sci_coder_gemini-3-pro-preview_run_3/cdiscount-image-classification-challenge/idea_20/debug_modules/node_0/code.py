import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed, HierarchyMapper
from library.bson_io import BSONImageReader
from library.feature_extraction import FeatureExtractor, run_feature_extraction
from library.dataset import CachedFeatureDataset
from library.model import DualStreamProjectedNetwork
from library.engine import fit_model, predict_test

if __name__ == "__main__":
    # 1. Setup
    print(">>> Setting up demonstration environment...")
    set_seed(42)

    # Define working directory for this demo
    DEMO_DIR = "./working/demo_execution"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # 2. Create Mini Metadata (Subset of real data for speed)
    print(">>> Creating mini datasets for fast execution...")

    # Load original metadata
    train_meta_full = pd.read_csv(Config.TRAIN_META)
    val_meta_full = pd.read_csv(Config.VAL_META)
    test_meta_full = pd.read_csv(Config.TEST_META)

    # Create subsets (50 train, 20 val, 20 test)
    mini_train = train_meta_full.head(50).copy()
    mini_val = val_meta_full.head(20).copy()
    mini_test = test_meta_full.head(20).copy()

    # Save mini metadata
    mini_train_path = os.path.join(DEMO_DIR, "train.csv")
    mini_val_path = os.path.join(DEMO_DIR, "val.csv")
    mini_test_path = os.path.join(DEMO_DIR, "test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # 3. Monkey-Patch Config to use Mini Data and Demo Directory
    print(">>> Patching Config for demo...")
    Config.WORKING_DIR = DEMO_DIR

    # Override Input Metadata Paths
    Config.TRAIN_META = mini_train_path
    Config.VAL_META = mini_val_path
    Config.TEST_META = mini_test_path

    # Override Output Feature Paths
    Config.TRAIN_FEATURES = os.path.join(DEMO_DIR, "train_features.npy")
    Config.TRAIN_LABELS = os.path.join(DEMO_DIR, "train_labels.npy")
    Config.VAL_FEATURES = os.path.join(DEMO_DIR, "val_features.npy")
    Config.VAL_LABELS = os.path.join(DEMO_DIR, "val_labels.npy")
    Config.TEST_FEATURES = os.path.join(DEMO_DIR, "test_features.npy")
    Config.TEST_IDS = os.path.join(DEMO_DIR, "test_ids.npy")
    Config.HIERARCHY_MAPPING = os.path.join(DEMO_DIR, "hierarchy_map.parquet")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Override Training Hyperparameters for Speed
    Config.TRAIN_BATCH_SIZE = 16
    Config.EXTRACT_BATCH_SIZE = 16
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 2  # Reduce workers for small data

    Config.print_config()

    # 4. Demonstrate BSON Reading
    print("\n>>> [Demo] BSONImageReader")
    # Use the first record from our mini train set
    sample_record = mini_train.iloc[0]
    reader = BSONImageReader(Config.TRAIN_BSON)

    images = reader.read_product(
        sample_record["bson_offset"], sample_record["bson_length"]
    )
    reader.close()

    print(f"Read {len(images)} images for product ID {sample_record['_id']}")
    assert len(images) > 0, "BSON reader failed to extract images"
    assert isinstance(images[0], np.ndarray), "Image is not a numpy array"
    assert images[0].shape[2] == 3, "Image is not RGB"
    print("BSON Reader validation passed.")

    # 5. Demonstrate Feature Extraction
    print("\n>>> [Demo] FeatureExtractor (ResNet50 + EfficientNet-B0)")
    # This will run on the mini datasets defined in Config
    # We force load_cached=False to demonstrate the extraction process
    run_feature_extraction(load_cached_data=False)

    assert os.path.exists(Config.TRAIN_FEATURES), "Train features not generated"
    assert os.path.exists(Config.TEST_FEATURES), "Test features not generated"

    # Verify feature dimensions
    feat_sample = np.load(Config.TRAIN_FEATURES)
    expected_dim = Config.RESNET_DIM + Config.EFFNET_DIM
    assert (
        feat_sample.shape[1] == expected_dim
    ), f"Feature dim mismatch: got {feat_sample.shape[1]}, expected {expected_dim}"
    print(f"Feature extraction complete. Shape: {feat_sample.shape}")

    # 6. Demonstrate Hierarchy Mapper & Dataset
    print("\n>>> [Demo] HierarchyMapper & CachedFeatureDataset")

    # Initialize Dataset (Train)
    train_ds = CachedFeatureDataset(
        Config.TRAIN_FEATURES, Config.TRAIN_LABELS, is_train=True, mixup_alpha=0.2
    )

    # Check Hierarchy Mapping
    mapper = train_ds.mapper
    print(f"Mapped {mapper.num_l3} target classes.")
    assert mapper.num_l3 == 5270, "Incorrect number of L3 classes"

    # Check __getitem__ (MixUp enabled)
    item = train_ds[0]
    # Expecting 8 items tuple for MixUp: (feat, l1, l2, l3, l1_b, l2_b, l3_b, lam)
    assert len(item) == 8, f"Dataset with MixUp should return 8 items, got {len(item)}"
    print("Dataset (Train/MixUp) validation passed.")

    # Initialize Dataset (Val - No MixUp)
    val_ds = CachedFeatureDataset(
        Config.VAL_FEATURES, Config.VAL_LABELS, is_train=False
    )
    item_val = val_ds[0]
    # Expecting 4 items: (feat, l1, l2, l3)
    assert len(item_val) == 4, f"Val Dataset should return 4 items, got {len(item_val)}"
    print("Dataset (Val) validation passed.")

    # 7. Demonstrate Model
    print("\n>>> [Demo] DualStreamProjectedNetwork")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DualStreamProjectedNetwork().to(device)

    # Create dummy batch
    dummy_input = torch.randn(4, expected_dim).to(device)
    out_l1, out_l2, out_l3 = model(dummy_input)

    print(
        f"Model Output Shapes: L1={out_l1.shape}, L2={out_l2.shape}, L3={out_l3.shape}"
    )
    assert out_l1.shape == (4, Config.NUM_CLASSES_L1)
    assert out_l3.shape == (4, Config.NUM_CLASSES_L3)
    print("Model forward pass validation passed.")

    # 8. Demonstrate Training Loop
    print("\n>>> [Demo] Training Loop (1 Epoch)")

    train_loader = DataLoader(
        train_ds, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run fit_model
    best_acc = fit_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epochs=1,
        save_path=os.path.join(DEMO_DIR, "demo_model.pth"),
    )

    assert os.path.exists(
        os.path.join(DEMO_DIR, "demo_model.pth")
    ), "Model checkpoint not saved"
    print(f"Training demo complete. Best Acc: {best_acc}")

    # 9. Demonstrate Inference
    print("\n>>> [Demo] Inference on Test Set")

    test_ds = CachedFeatureDataset(Config.TEST_FEATURES, Config.TEST_IDS, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=False)

    ids, preds = predict_test(model, test_loader, device)

    print(f"Predictions generated: {len(preds)}")
    assert len(preds) == len(mini_test), "Mismatch in prediction count"

    # Convert predictions back to category_ids
    # We need the mapper from the training dataset
    pred_category_ids = train_ds.mapper.inverse_transform_targets(preds)

    # Create submission dataframe
    sub_df = pd.DataFrame({"_id": ids, "category_id": pred_category_ids})
    print("Sample Submission Head:")
    print(sub_df.head())

    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    print("\n>>> DEMONSTRATION COMPLETE SUCCESS <<<")
