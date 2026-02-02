import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_hierarchy_mappings, save_submission
from library.feature_extraction import extract_features_to_disk
from library.dataset import CachedFeatureDataset
from library.model import HierarchicalMLP
from library.training import train_ensemble_member


def main():
    print("=== Starting Library Usage Demo ===")

    # 1. Setup
    seed_everything(42)
    work_dir = "./working"
    os.makedirs(work_dir, exist_ok=True)

    # Override Config for the purpose of this quick demonstration
    Config.CACHE_DIR = work_dir  # Save artifacts to working dir
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # ------------------------------------------------------------------------
    # 2. Demonstrate library.utils.get_hierarchy_mappings
    # ------------------------------------------------------------------------
    print("\n[Demo] Verifying Hierarchy Mappings...")
    raw_to_l3, l3_to_raw, l3_to_l1, l3_to_l2 = get_hierarchy_mappings(
        load_cached_data=True
    )

    # Validation
    assert len(raw_to_l3) == 5270, f"Expected 5270 categories, got {len(raw_to_l3)}"
    assert len(l3_to_l1) == 5270
    assert len(l3_to_l2) == 5270
    # Check consistency: first index should map back and forth
    first_l3_idx = 0
    first_raw_id = l3_to_raw[first_l3_idx]
    assert raw_to_l3[first_raw_id] == first_l3_idx
    print("Hierarchy mappings verified successfully.")

    # ------------------------------------------------------------------------
    # 3. Demonstrate library.feature_extraction.extract_features_to_disk
    # ------------------------------------------------------------------------
    print("\n[Demo] Running Feature Extraction on Subset...")

    # Create a mini metadata file to avoid processing the whole dataset
    # We use the first 20 records from the provided train metadata
    full_train_meta = pd.read_csv(Config.TRAIN_META)
    mini_meta_df = full_train_meta.head(20).copy()
    mini_meta_path = os.path.join(work_dir, "mini_train_meta.csv")
    mini_meta_df.to_csv(mini_meta_path, index=False)

    mini_feat_path = os.path.join(work_dir, "mini_train_features.npy")
    mini_label_path = os.path.join(work_dir, "mini_train_labels.npy")

    # Run extraction
    # This reads actual images from train.bson based on offsets in mini_meta_path
    extract_features_to_disk(
        metadata_path=mini_meta_path,
        bson_path=Config.TRAIN_BSON,
        out_feat_path=mini_feat_path,
        out_label_path=mini_label_path,
        is_test=False,
    )

    # Validation
    assert os.path.exists(mini_feat_path), "Feature file not created"
    assert os.path.exists(mini_label_path), "Label file not created"

    features = np.load(mini_feat_path)
    labels = np.load(mini_label_path)

    assert features.shape == (
        20,
        2048,
    ), f"Expected shape (20, 2048), got {features.shape}"
    assert labels.shape == (20,), f"Expected shape (20,), got {labels.shape}"
    print("Feature extraction successful.")

    # ------------------------------------------------------------------------
    # 4. Demonstrate library.dataset.CachedFeatureDataset
    # ------------------------------------------------------------------------
    print("\n[Demo] Loading Dataset...")

    dataset = CachedFeatureDataset(
        features_path=mini_feat_path, labels_path=mini_label_path, is_test=False
    )

    assert len(dataset) == 20

    # Check item retrieval
    feat_tensor, targets = dataset[0]
    y1, y2, y3 = targets

    assert isinstance(feat_tensor, torch.Tensor)
    assert feat_tensor.shape == (2048,)
    assert isinstance(y3, torch.Tensor)
    print("Dataset loaded and verified.")

    # ------------------------------------------------------------------------
    # 5. Demonstrate library.model.HierarchicalMLP
    # ------------------------------------------------------------------------
    print("\n[Demo] Initializing Model...")

    model = HierarchicalMLP(
        input_dim=Config.EMBEDDING_DIM,
        num_classes_l1=Config.NUM_CLASSES_L1,
        num_classes_l2=Config.NUM_CLASSES_L2,
        num_classes_l3=Config.NUM_CLASSES_L3,
    )

    # Move to configured device
    device = torch.device(Config.DEVICE)
    model.to(device)

    # Forward pass check
    dummy_input = torch.randn(4, 2048).to(device)
    with torch.no_grad():
        l1_logits, l2_logits, l3_logits = model(dummy_input)

    assert l1_logits.shape == (4, 49)
    assert l2_logits.shape == (4, 483)
    assert l3_logits.shape == (4, 5270)
    print("Model forward pass verified.")

    # ------------------------------------------------------------------------
    # 6. Demonstrate library.training.train_ensemble_member
    # ------------------------------------------------------------------------
    print("\n[Demo] Running Training Loop...")

    # Configure for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4

    # Create loaders
    train_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Run training for one member
    # This will train for 1 epoch and save the model to Config.CACHE_DIR
    best_acc = train_ensemble_member(
        member_id=0, train_loader=train_loader, val_loader=val_loader
    )

    expected_model_path = os.path.join(Config.CACHE_DIR, "mlp_ensemble_0.pth")
    assert os.path.exists(expected_model_path), "Model checkpoint not saved"
    print(f"Training finished. Best Validation Accuracy: {best_acc}")

    # ------------------------------------------------------------------------
    # 7. Demonstrate library.utils.save_submission
    # ------------------------------------------------------------------------
    print("\n[Demo] Generating Submission File...")

    # Mock predictions
    test_ids = np.array([101, 102, 103])
    # Random Level 3 indices
    predicted_l3 = np.array([0, 100, 5269])

    submission_path = os.path.join(work_dir, "demo_submission.csv")

    save_submission(
        test_ids=test_ids,
        predicted_l3_indices=predicted_l3,
        l3_to_raw_map=l3_to_raw,
        file_path=submission_path,
    )

    assert os.path.exists(submission_path)
    df_sub = pd.read_csv(submission_path)
    assert len(df_sub) == 3
    assert list(df_sub.columns) == ["_id", "category_id"]
    print("Submission file generated successfully.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
