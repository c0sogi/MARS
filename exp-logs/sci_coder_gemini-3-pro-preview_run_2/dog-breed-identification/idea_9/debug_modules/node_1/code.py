import os
import sys
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Subset
import shutil

# Import library modules
from library import config
from library import transforms
from library import dataset
from library import model_factory
from library import feature_engine
from library import classifier_engine


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Starting Library Usage Demonstration...")
    set_seed(config.SEED)

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[1] Configuring environment for demo...")

    # Override working directory for this demo to avoid conflicts
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Patch config to use demo directory
    config.WORKING_DIR = DEMO_DIR
    config.SUBMISSION_DIR = DEMO_DIR
    config.SUBMISSION_FILE = os.path.join(DEMO_DIR, "submission.csv")

    # Update cache paths to point to demo directory
    config.CACHE_FILES = {
        "train": {
            "features": os.path.join(DEMO_DIR, "train_emb.npy"),
            "labels": os.path.join(DEMO_DIR, "train_lbl.npy"),
            "ids": os.path.join(DEMO_DIR, "train_ids.npy"),
        },
        "val": {
            "features": os.path.join(DEMO_DIR, "val_emb.npy"),
            "labels": os.path.join(DEMO_DIR, "val_lbl.npy"),
            "ids": os.path.join(DEMO_DIR, "val_ids.npy"),
        },
        "test": {
            "features": os.path.join(DEMO_DIR, "test_emb.npy"),
            "ids": os.path.join(DEMO_DIR, "test_ids.npy"),
        },
    }

    # Optimize Classifier for Speed (Demo settings)
    config.LOGREG_PARAMS["max_iter"] = 10
    config.LOGREG_PARAMS["cv"] = 2
    config.LOGREG_PARAMS["Cs"] = 2
    config.LOGREG_PARAMS["n_jobs"] = 1

    print("    Configuration patched successfully.")

    # ==========================================
    # 2. Transforms Verification
    # ==========================================
    print("\n[2] Verifying Transforms...")

    # Get transform pipelines
    view_transforms = transforms.get_view_transforms()

    # Create dummy PIL image (RGB)
    dummy_img = Image.new("RGB", (500, 400), color="red")

    # Check Global View
    global_t = view_transforms["global"](dummy_img)
    assert global_t.shape == (
        3,
        224,
        224,
    ), f"Global view shape mismatch: {global_t.shape}"

    # Check Standard View
    standard_t = view_transforms["standard"](dummy_img)
    assert standard_t.shape == (
        3,
        224,
        224,
    ), f"Standard view shape mismatch: {standard_t.shape}"

    # Check Local View (FiveCrop -> Stack)
    local_t = view_transforms["local"](dummy_img)
    # Expecting (5, 3, 224, 224)
    assert local_t.shape == (
        5,
        3,
        224,
        224,
    ), f"Local view shape mismatch: {local_t.shape}"

    print("    All transforms produce expected output shapes.")

    # ==========================================
    # 3. Dataset Verification
    # ==========================================
    print("\n[3] Verifying Dataset...")

    # Instantiate dataset (Train split)
    # Using the provided metadata path in config
    ds = dataset.DogDataset(config.TRAIN_CSV, mode="train")

    # Fetch one item
    item = ds[0]

    # Verify keys
    expected_keys = {"id", "label", "global_view", "standard_view", "local_view"}
    assert expected_keys.issubset(
        item.keys()
    ), f"Missing keys in dataset item. Found: {item.keys()}"

    # Verify Shapes (Dataset applies TTA: Original + Flip)
    # Global: Stack of 2 -> (2, 3, 224, 224)
    assert item["global_view"].shape == (
        2,
        3,
        224,
        224,
    ), f"Global view item shape error: {item['global_view'].shape}"

    # Standard: Stack of 2 -> (2, 3, 224, 224)
    assert item["standard_view"].shape == (
        2,
        3,
        224,
        224,
    ), f"Standard view item shape error: {item['standard_view'].shape}"

    # Local: 5 crops * 2 (Orig+Flip) -> (10, 3, 224, 224)
    assert item["local_view"].shape == (
        10,
        3,
        224,
        224,
    ), f"Local view item shape error: {item['local_view'].shape}"

    # Verify Label
    assert isinstance(item["label"].item(), int), "Label should be an integer"

    print(f"    Dataset loaded successfully. Classes: {len(ds.classes)}")

    # ==========================================
    # 4. Model Factory Verification
    # ==========================================
    print("\n[4] Verifying Feature Extractor Model...")

    device = config.DEVICE
    model = model_factory.get_feature_extractor(device=device)

    # Create dummy batch: (Batch_Size=2, Channels=3, H=224, W=224)
    dummy_input = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        outputs = model(dummy_input)

    # Verify output stages and dimensions
    # Stage 3: 768 dim, Stage 4: 1536 dim
    assert "stage3" in outputs and "stage4" in outputs
    assert outputs["stage3"].shape == (
        2,
        768,
    ), f"Stage3 shape error: {outputs['stage3'].shape}"
    assert outputs["stage4"].shape == (
        2,
        1536,
    ), f"Stage4 shape error: {outputs['stage4'].shape}"

    print("    Model forward pass successful. Output dimensions verified.")

    # ==========================================
    # 5. Feature Engine Verification (Inference Logic)
    # ==========================================
    print("\n[5] Verifying Feature Engine Inference...")

    # Create a small subset loader to test run_inference quickly
    subset_indices = range(4)  # Process only 4 images
    subset_ds = Subset(ds, subset_indices)
    small_loader = DataLoader(subset_ds, batch_size=2, num_workers=0)

    # Run inference
    feats, lbls, ids = feature_engine.run_inference(small_loader, model, device)

    # Expected Feature Dimension:
    # (Global_S4 + Global_S3) + (Standard_S4 + Standard_S3) + (Local_S4 + Local_S3)
    # (1536 + 768) * 3 = 6912
    expected_dim = 6912

    assert feats.shape == (
        4,
        expected_dim,
    ), f"Feature matrix shape error: {feats.shape}"
    assert len(lbls) == 4
    assert len(ids) == 4

    print(f"    Inference successful. Feature vector shape: {feats.shape}")

    # ==========================================
    # 6. Classifier Engine Verification
    # ==========================================
    print("\n[6] Verifying Classifier Engine (Training & Prediction)...")

    # To save time, we will generate synthetic features and save them to the cache.
    # This allows us to test the classifier logic without running full inference on 7k images.

    # Dimensions
    n_train = 300  # Increased to ensure coverage of 120 classes for CV
    n_val = 20
    n_test = 10
    feat_dim = 6912
    n_classes = len(ds.classes)

    # Synthetic Data Generation
    print("    Generating synthetic cached data...")

    # Train
    X_train = np.random.rand(n_train, feat_dim).astype(np.float32)

    # Ensure every class appears at least twice for cv=2
    base_labels = np.tile(np.arange(n_classes), 2)
    if n_train > len(base_labels):
        extra = np.random.randint(0, n_classes, n_train - len(base_labels))
        y_train = np.concatenate([base_labels, extra])
    else:
        y_train = base_labels[:n_train]
    np.random.shuffle(y_train)

    ids_train = np.array([f"train_{i}" for i in range(n_train)])

    np.save(config.CACHE_FILES["train"]["features"], X_train)
    np.save(config.CACHE_FILES["train"]["labels"], y_train)
    np.save(config.CACHE_FILES["train"]["ids"], ids_train)

    # Val
    X_val = np.random.rand(n_val, feat_dim).astype(np.float32)
    y_val = np.random.randint(0, n_classes, n_val)
    ids_val = np.array([f"val_{i}" for i in range(n_val)])

    np.save(config.CACHE_FILES["val"]["features"], X_val)
    np.save(config.CACHE_FILES["val"]["labels"], y_val)
    np.save(config.CACHE_FILES["val"]["ids"], ids_val)

    # Test
    X_test = np.random.rand(n_test, feat_dim).astype(np.float32)
    ids_test = np.array([f"test_{i}" for i in range(n_test)])

    np.save(config.CACHE_FILES["test"]["features"], X_test)
    np.save(config.CACHE_FILES["test"]["ids"], ids_test)

    # Train Classifier
    # load_cached_data=True will pick up the files we just wrote
    print("    Training classifier...")
    clf, loss = classifier_engine.train_classifier(
        load_cached_data=True, save_model=True
    )

    assert clf is not None
    print(f"    Training complete. Validation Log Loss: {loss:.4f}")

    # Predict Submission
    print("    Generating submission...")
    classifier_engine.predict_submission(clf, load_cached_data=True)

    # Verify Submission File
    if not os.path.exists(config.SUBMISSION_FILE):
        raise FileNotFoundError("Submission file was not created.")

    sub_df = pd.read_csv(config.SUBMISSION_FILE)

    # Check shape: (n_test, n_classes + 1 for id)
    expected_cols = n_classes + 1
    assert sub_df.shape == (
        n_test,
        expected_cols,
    ), f"Submission shape error: {sub_df.shape}"
    assert "id" in sub_df.columns

    print(f"    Submission file verified at {config.SUBMISSION_FILE}")
    print("\nAll library components verified successfully.")


if __name__ == "__main__":
    main()
