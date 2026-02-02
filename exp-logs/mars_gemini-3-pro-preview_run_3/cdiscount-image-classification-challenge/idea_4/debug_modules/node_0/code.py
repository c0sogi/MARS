import os
import shutil
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything
from library.data_loader import BSONProductDataset, get_bson_loader, get_label_mapping
from library.model import ImageEncoder, DualStatClassifier
from library.feature_extractor import extract_and_aggregate
from library.trainer import train_model, generate_submission


def run_demo():
    # ==========================================
    # 1. SETUP & CONFIGURATION OVERRIDE
    # ==========================================
    print("=== 1. Setup & Configuration ===")
    seed_everything(42)

    # Define a demo working directory
    DEMO_DIR = os.path.join(Config.WORKING_DIR, "demo_run")
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Patch Config for speed and demo purposes
    Config.WORKING_DIR = DEMO_DIR
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = "submission.csv"  # Relative to WORKING_DIR
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 2
    Config.NUM_WORKERS = 2  # Reduce workers for small demo
    Config.DEBUG = True

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Model Save Path: {Config.MODEL_SAVE_PATH}")

    # ==========================================
    # 2. DATA LOADING VERIFICATION
    # ==========================================
    print("\n=== 2. Data Loading Verification ===")

    # Test Dataset with a small limit
    limit = 10
    dataset = BSONProductDataset(Config.TRAIN_META, limit=limit)
    print(f"Dataset initialized with limit={limit}. Length: {len(dataset)}")

    # Validate single item retrieval
    _id, img_stack, label = dataset[0]
    print(f"Item 0: ID={_id}, Label={label}, Image Stack Shape={img_stack.shape}")

    # Assertions
    assert isinstance(_id, int), "ID should be an integer"
    assert isinstance(label, int), "Label should be an integer"
    assert img_stack.dim() == 4, "Image stack should be 4D (N, C, H, W)"
    assert img_stack.shape[1] == 3, "Images should have 3 channels"
    assert (
        img_stack.shape[2] == 224 and img_stack.shape[3] == 224
    ), "Images should be 224x224"

    # Test DataLoader Collation
    loader = get_bson_loader(
        Config.TRAIN_META, batch_size=Config.BATCH_SIZE, limit=limit, num_workers=0
    )
    batch = next(iter(loader))

    print("Batch Keys:", batch.keys())
    print("Batch IDs:", batch["ids"])
    print("Batch Images Shape:", batch["images"].shape)
    print("Batch Counts:", batch["counts"])

    # Assertions for batch
    assert (
        "ids" in batch and "images" in batch and "labels" in batch and "counts" in batch
    )
    assert batch["ids"].shape[0] <= Config.BATCH_SIZE
    # Total images in batch should equal sum of counts
    assert batch["images"].shape[0] == batch["counts"].sum().item()

    print("Data Loading Logic Verified.")

    # ==========================================
    # 3. MODEL ARCHITECTURE VERIFICATION
    # ==========================================
    print("\n=== 3. Model Architecture Verification ===")

    device = Config.DEVICE

    # Test Image Encoder (ResNet Backbone)
    encoder = ImageEncoder().to(device)
    encoder.eval()

    # Create dummy batch of 2 images
    dummy_imgs = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        feats = encoder(dummy_imgs)

    print(f"Encoder Output Shape: {feats.shape}")
    assert feats.shape == (2, 2048), f"Expected (2, 2048), got {feats.shape}"

    # Test DualStatClassifier
    # Input dim is 4096 (2048 Mean + 2048 Max)
    classifier = DualStatClassifier(input_dim=4096, num_classes=Config.NUM_CLASSES).to(
        device
    )
    classifier.eval()

    dummy_agg_feats = torch.randn(2, 4096).to(device)
    with torch.no_grad():
        logits = classifier(dummy_agg_feats)

    print(f"Classifier Output Shape: {logits.shape}")
    assert logits.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Expected (2, {Config.NUM_CLASSES}), got {logits.shape}"

    print("Model Architecture Verified.")

    # ==========================================
    # 4. FEATURE EXTRACTION PIPELINE
    # ==========================================
    print("\n=== 4. Feature Extraction Pipeline ===")

    # We will extract features for a small subset of training data
    subset_size = 20
    feat_save_path = os.path.join(DEMO_DIR, "train_features.npy")
    label_save_path = os.path.join(DEMO_DIR, "train_labels.npy")
    id_save_path = os.path.join(DEMO_DIR, "train_ids.npy")

    print(f"Extracting features for {subset_size} records...")
    features, labels, ids = extract_and_aggregate(
        metadata_path=Config.TRAIN_META,
        save_path_features=feat_save_path,
        save_path_labels=label_save_path,
        save_path_ids=id_save_path,
        load_cached_data=False,  # Force extraction
        limit=subset_size,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    print(f"Extracted Features Shape: {features.shape}")
    print(f"Extracted Labels Shape: {labels.shape}")

    assert features.shape == (subset_size, 4096), "Feature shape mismatch"
    assert labels.shape == (subset_size,), "Label shape mismatch"
    assert os.path.exists(feat_save_path), "Feature file not saved"

    print("Feature Extraction Verified.")

    # ==========================================
    # 5. TRAINING LOOP DEMONSTRATION
    # ==========================================
    print("\n=== 5. Training Loop Demonstration ===")

    # Split the extracted data into train/val for demo
    split_idx = int(subset_size * 0.8)
    train_feats, val_feats = features[:split_idx], features[split_idx:]
    train_lbls, val_lbls = labels[:split_idx], labels[split_idx:]

    print(f"Train size: {len(train_feats)}, Val size: {len(val_feats)}")

    # Run training
    # This will save the model to Config.MODEL_SAVE_PATH
    train_model(train_feats, train_lbls, val_feats, val_lbls)

    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model checkpoint not found after training"
    print("Training Loop Verified.")

    # ==========================================
    # 6. INFERENCE & SUBMISSION
    # ==========================================
    print("\n=== 6. Inference & Submission ===")

    # We'll use the validation features as "test" features for this demo
    test_feats = val_feats
    test_ids = ids[split_idx:]

    print(f"Generating submission for {len(test_ids)} test items...")

    generate_submission(test_feats, test_ids)

    submission_file = os.path.join(Config.WORKING_DIR, Config.SUBMISSION_PATH)
    assert os.path.exists(submission_file), "Submission file not created"

    # Validate content
    df_sub = pd.read_csv(submission_file)
    print("Submission Head:")
    print(df_sub.head())

    assert len(df_sub) == len(test_ids), "Submission length mismatch"
    assert list(df_sub.columns) == [
        "_id",
        "category_id",
    ], "Invalid columns in submission"
    assert (
        df_sub["category_id"].dtype == int or df_sub["category_id"].dtype == np.int64
    ), "Category ID must be int"

    print("Inference Pipeline Verified.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
