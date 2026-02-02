import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import configuration and library components
from library.config import (
    TRAIN_BSON_PATH,
    TEST_BSON_PATH,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    DEVICE,
    NUM_CLASSES,
    FEATURE_DIM,
    HIDDEN_DIM,
    DROPOUT_RATE,
)
from library.utils import (
    read_bson_images_at_offset,
    save_submission,
)
from library.feature_engine import extract_dataset_features
from library.dataset import CachedFeatureDataset, get_class_weights
from library.model import AttentionClassifier, get_category_mapping
from library.trainer import run_training


def main():
    print("=== Starting Library Demo Script ===")

    # 1. Setup & Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create a temporary directory for this demo run
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR)

    print(f"Working directory: {DEMO_DIR}")

    # 2. Create Data Subsets (Optimization for Speed)
    # We use a tiny fraction of the data to demonstrate functionality quickly.
    print("\n[Step 1] Creating metadata subsets...")

    train_meta = pd.read_csv(TRAIN_META_PATH)
    val_meta = pd.read_csv(VAL_META_PATH)
    test_meta = pd.read_csv(TEST_META_PATH)

    # Select top N records
    sub_train = train_meta.head(50).copy()
    sub_val = val_meta.head(20).copy()
    sub_test = test_meta.head(20).copy()

    # Save subset metadata
    sub_train_path = os.path.join(DEMO_DIR, "train_subset.csv")
    sub_val_path = os.path.join(DEMO_DIR, "val_subset.csv")
    sub_test_path = os.path.join(DEMO_DIR, "test_subset.csv")

    sub_train.to_csv(sub_train_path, index=False)
    sub_val.to_csv(sub_val_path, index=False)
    sub_test.to_csv(sub_test_path, index=False)

    print(
        f"   Created subsets: Train={len(sub_train)}, Val={len(sub_val)}, Test={len(sub_test)}"
    )

    # 3. Verify Utility Functions
    print("\n[Step 2] Verifying BSON reading utility...")
    sample_row = sub_train.iloc[0]
    images = read_bson_images_at_offset(
        TRAIN_BSON_PATH, sample_row["bson_offset"], sample_row["bson_length"]
    )

    # Assertions to verify logic
    assert isinstance(images, list), "Output should be a list"
    assert len(images) > 0, "Should extract at least one image"
    assert isinstance(images[0], np.ndarray), "Image should be a numpy array"
    print(
        f"   Successfully extracted {len(images)} images for product ID {sample_row['_id']}"
    )

    # 4. Feature Extraction
    print("\n[Step 3] Extracting features using ResNet50...")
    # We use extract_dataset_features from library.feature_engine

    # Train Split
    train_feats, train_idx, train_labels_raw, train_ids = extract_dataset_features(
        metadata_path=sub_train_path,
        bson_path=TRAIN_BSON_PATH,
        save_dir=DEMO_DIR,
        split_name="train",
        batch_size=16,
        load_cached_data=False,
    )

    # Validation Split
    val_feats, val_idx, val_labels_raw, val_ids = extract_dataset_features(
        metadata_path=sub_val_path,
        bson_path=TRAIN_BSON_PATH,
        save_dir=DEMO_DIR,
        split_name="val",
        batch_size=16,
        load_cached_data=False,
    )

    # Test Split (No labels)
    test_feats, test_idx, _, test_ids = extract_dataset_features(
        metadata_path=sub_test_path,
        bson_path=TEST_BSON_PATH,
        save_dir=DEMO_DIR,
        split_name="test",
        batch_size=16,
        load_cached_data=False,
    )

    # Validate feature shapes
    assert train_feats.shape[1] == FEATURE_DIM, f"Feature dim should be {FEATURE_DIM}"
    assert len(train_idx) == 50, "Index length should match subset size"
    print("   Feature extraction complete.")

    # 5. Dataset Loading & Label Mapping
    print("\n[Step 4] Preparing Datasets and DataLoaders...")

    # Get global category mapping (Category ID -> 0..N index)
    cat2idx, idx2cat = get_category_mapping()

    # Transform raw category IDs to model indices
    train_y = np.array([cat2idx[l] for l in train_labels_raw])
    val_y = np.array([cat2idx[l] for l in val_labels_raw])

    # Instantiate CachedFeatureDataset
    train_ds = CachedFeatureDataset(train_feats, train_idx, train_y)
    val_ds = CachedFeatureDataset(val_feats, val_idx, val_y)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=10, shuffle=False)

    # Verify batch structure
    bags, masks, targets = next(iter(train_loader))
    assert bags.ndim == 3, "Bags should be [Batch, MaxLen, Dim]"
    assert masks.ndim == 2, "Masks should be [Batch, MaxLen]"
    print("   DataLoaders ready.")

    # 6. Model Training
    print("\n[Step 5] Initializing and Training Model...")

    # Compute class weights (using subset labels for demo purposes)
    class_weights = get_class_weights(train_y, NUM_CLASSES)
    class_weights = class_weights.to(DEVICE)

    # Initialize Architecture
    model = AttentionClassifier(
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Run Training Loop
    model_save_path = os.path.join(DEMO_DIR, "demo_model.pth")
    best_acc = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,  # Skip scheduler for short demo
        num_epochs=2,
        patience=1,
        save_path=model_save_path,
        device=DEVICE,
    )

    assert os.path.exists(model_save_path), "Model file was not saved."
    print(f"   Training successful. Best Validation Accuracy: {best_acc:.4f}")

    # 7. Inference & Submission
    print("\n[Step 6] Running Inference on Test Subset...")

    # Load best model weights
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    # Prepare Test Dataset
    test_ds = CachedFeatureDataset(test_feats, test_idx, labels=None)
    test_loader = DataLoader(test_ds, batch_size=10, shuffle=False)

    all_preds_idx = []

    with torch.no_grad():
        for bags, masks in test_loader:
            bags = bags.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(bags, masks)
            preds = torch.argmax(outputs, dim=1)
            all_preds_idx.extend(preds.cpu().numpy())

    # Map indices back to Category IDs
    final_preds = [idx2cat[p] for p in all_preds_idx]

    # Save Submission
    submission_path = os.path.join(DEMO_DIR, "submission.csv")
    save_submission(test_ids, final_preds, submission_path)

    # Verify output
    df_sub = pd.read_csv(submission_path)
    assert len(df_sub) == 20, "Submission should have 20 rows"
    assert list(df_sub.columns) == ["_id", "category_id"], "Invalid columns"

    print(f"   Submission saved to {submission_path}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
