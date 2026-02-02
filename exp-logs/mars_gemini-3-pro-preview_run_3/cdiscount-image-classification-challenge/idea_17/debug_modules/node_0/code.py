import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import seed_everything, get_hierarchy_mappings
from library.feature_extraction import FeatureExtractor
from library.dataset import CachedFeatureDataset
from library.model import PDFCNet
from library.training import train_model


def main():
    # ==========================================
    # 1. CONFIGURATION OVERRIDE
    # ==========================================
    print("Step 1: Configuring environment for demo run...")

    # Set a specific working directory for this demo to avoid conflicts
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths to use the demo directory
    Config.WORKING_DIR = DEMO_DIR
    Config.HIERARCHY_MAPPING_PATH = os.path.join(DEMO_DIR, "hierarchy_mapping.parquet")
    Config.TRAIN_FEATURES_PATH = os.path.join(DEMO_DIR, "train_features.npy")
    Config.TRAIN_LABELS_PATH = os.path.join(DEMO_DIR, "train_labels.npy")
    Config.VAL_FEATURES_PATH = os.path.join(DEMO_DIR, "val_features.npy")
    Config.VAL_LABELS_PATH = os.path.join(DEMO_DIR, "val_labels.npy")
    Config.TEST_FEATURES_PATH = os.path.join(DEMO_DIR, "test_features.npy")
    Config.TEST_IDS_PATH = os.path.join(DEMO_DIR, "test_ids.npy")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "model_demo.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Override Config parameters for speed
    Config.DEBUG = True
    Config.DEBUG_SIZE = 64  # Process only 64 images per split
    Config.BATCH_SIZE = 16  # Small batch size
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.NUM_WORKERS = 2  # Minimal workers

    # Set seed
    seed_everything(Config.SEED)
    print("Configuration updated. Debug mode enabled.")

    # ==========================================
    # 2. HIERARCHY MAPPING
    # ==========================================
    print("\nStep 2: verifying hierarchy mappings...")
    mapping_dict, idx_to_cat = get_hierarchy_mappings(load_cached_data=False)

    # Validations
    assert len(mapping_dict) > 0, "Mapping dictionary is empty"
    assert len(idx_to_cat) > 0, "Index to Category mapping is empty"

    # Check consistency for a known category if possible, or just structure
    first_cat = next(iter(mapping_dict))
    assert "l1" in mapping_dict[first_cat]
    assert "l2" in mapping_dict[first_cat]
    assert "l3" in mapping_dict[first_cat]
    print(f"Mappings verified. Total categories: {len(mapping_dict)}")

    # ==========================================
    # 3. FEATURE EXTRACTION
    # ==========================================
    print("\nStep 3: Running feature extraction (subset)...")
    # This will use the metadata in ./metadata and the images in ./input
    # but only process Config.DEBUG_SIZE samples.
    extractor = FeatureExtractor()
    extractor.extract_features(load_cached_data=False)

    # Verify outputs
    assert os.path.exists(Config.TRAIN_FEATURES_PATH), "Train features not found"
    assert os.path.exists(Config.TEST_FEATURES_PATH), "Test features not found"

    train_feats = np.load(Config.TRAIN_FEATURES_PATH)
    print(f"Feature extraction complete. Train features shape: {train_feats.shape}")
    assert (
        train_feats.shape[1] == Config.INPUT_DIM
    ), f"Expected dim {Config.INPUT_DIM}, got {train_feats.shape[1]}"

    # ==========================================
    # 4. DATASET VERIFICATION
    # ==========================================
    print("\nStep 4: Verifying Dataset class...")
    train_ds = CachedFeatureDataset(
        Config.TRAIN_FEATURES_PATH, Config.TRAIN_LABELS_PATH, is_test=False
    )

    feat, l1, l2, l3 = train_ds[0]
    assert isinstance(feat, torch.Tensor), "Feature is not a tensor"
    assert isinstance(l1, (int, np.integer)), "Label L1 is not int"
    # Check if labels are within range
    assert 0 <= l1 < Config.NUM_CLASSES_L1
    print("Dataset item check passed.")

    # ==========================================
    # 5. MODEL VERIFICATION
    # ==========================================
    print("\nStep 5: Verifying Model architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PDFCNet().to(device)

    # Dummy forward pass
    dummy_input = torch.randn(4, Config.INPUT_DIM).to(device)
    l1_out, l2_out, l3_out = model(dummy_input)

    assert l1_out.shape == (4, Config.NUM_CLASSES_L1)
    assert l2_out.shape == (4, Config.NUM_CLASSES_L2)
    assert l3_out.shape == (4, Config.NUM_CLASSES_L3)
    print(f"Model forward pass successful. Device: {device}")

    # ==========================================
    # 6. TRAINING LOOP
    # ==========================================
    print("\nStep 6: Running training loop (1 epoch)...")
    # train_model uses the global Config which we modified
    train_model()

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    print("Training loop completed successfully.")

    # ==========================================
    # 7. INFERENCE & SUBMISSION
    # ==========================================
    print("\nStep 7: Generating submission...")

    # Load Test Dataset
    test_ds = CachedFeatureDataset(
        Config.TEST_FEATURES_PATH, id_path=Config.TEST_IDS_PATH, is_test=True
    )
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Load Model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.eval()

    all_preds = []
    all_ids = []

    # Get mapping to convert L3 index back to Category ID
    _, idx_to_cat = get_hierarchy_mappings(load_cached_data=True)

    with torch.no_grad():
        for features, pids in test_loader:
            features = features.to(device)

            # Forward
            _, _, logits3 = model(features)

            # Get predictions
            preds_idx = torch.argmax(logits3, dim=1).cpu().numpy()

            # Map to Category ID
            preds_cat = [idx_to_cat[idx] for idx in preds_idx]

            all_preds.extend(preds_cat)
            all_ids.extend(pids.numpy())

    # Create DataFrame
    submission = pd.DataFrame({"_id": all_ids, "category_id": all_preds})

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission.shape}")
    print(submission.head())

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
