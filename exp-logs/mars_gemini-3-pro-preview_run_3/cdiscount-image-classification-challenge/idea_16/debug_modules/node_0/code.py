import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, setup_logging, HierarchyMapper
from library.data import get_extraction_loader, get_training_loader, get_test_loader
from library.models import DeepFeatureCascade, DualBackboneExtractor
from library.feature_engineering import extract_features_for_split
from library.engine import train_model


def run_demo():
    # ==========================================
    # 1. SETUP & CONFIGURATION OVERRIDES
    # ==========================================
    print("\n=== 1. Setup & Configuration ===")

    # Set seeds for reproducibility
    seed_everything(42)

    # Define a demo working directory to avoid overwriting real work
    DEMO_DIR = os.path.join(Config.WORKING_DIR, "demo_execution")
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths and parameters for the demo
    Config.IDEA_DIR = DEMO_DIR
    Config.MODEL_DIR = DEMO_DIR

    # Paths for cached artifacts
    Config.TRAIN_FEATURES_PATH = os.path.join(DEMO_DIR, "train_features.npy")
    Config.TRAIN_LABELS_PATH = os.path.join(DEMO_DIR, "train_labels.npy")
    Config.VAL_FEATURES_PATH = os.path.join(
        DEMO_DIR, "val_features.npy"
    )  # Re-using train for demo speed
    Config.VAL_LABELS_PATH = os.path.join(DEMO_DIR, "val_labels.npy")
    Config.TEST_FEATURES_PATH = os.path.join(DEMO_DIR, "test_features.npy")
    Config.TEST_IDS_PATH = os.path.join(DEMO_DIR, "test_ids.npy")
    Config.HIERARCHY_MAPPING_PATH = os.path.join(DEMO_DIR, "hierarchy_map.parquet")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Hyperparameters for speed
    Config.BATCH_SIZE_EXTRACTION = 4
    Config.BATCH_SIZE_TRAIN = 8
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 2  # Reduce workers for small demo

    # Subset size for the demo
    SUBSET_SIZE = 50

    print(f"Working Directory: {DEMO_DIR}")
    print(f"Subset Size: {SUBSET_SIZE}")
    print(f"Device: {Config.DEVICE}")

    # ==========================================
    # 2. HIERARCHY MAPPING DEMO
    # ==========================================
    print("\n=== 2. Hierarchy Mapping Demo ===")

    mapper = HierarchyMapper(
        hierarchy_csv_path=Config.CATEGORY_NAMES_PATH,
        cache_path=Config.HIERARCHY_MAPPING_PATH,
    )

    # Process and create the mapping
    mapping_df = mapper.process_hierarchy(load_cached_data=False)

    # Validation
    assert os.path.exists(Config.HIERARCHY_MAPPING_PATH), "Mapping parquet not created."
    assert (
        len(mapping_df) == Config.NUM_CLASSES_L3
    ), f"Expected {Config.NUM_CLASSES_L3} classes, got {len(mapping_df)}"

    # Test get_labels with a known ID (from the first few rows of category_names.csv if possible, or random)
    # Let's pick a valid category_id from the mapping df
    sample_cat_id = mapping_df.iloc[0]["category_id"]
    labels = mapper.get_labels([sample_cat_id])

    print(f"Sample Category ID: {sample_cat_id}")
    print(
        f"Mapped Labels: L1={labels['l1'][0]}, L2={labels['l2'][0]}, L3={labels['l3'][0]}"
    )

    assert labels["l3"][0] == mapping_df.iloc[0]["label_l3"], "Label mapping mismatch."

    # Test reverse mapping
    recovered_id = mapper.get_category_id_from_label(np.array([labels["l3"][0]]))
    assert recovered_id[0] == sample_cat_id, "Reverse mapping failed."
    print("HierarchyMapper validation passed.")

    # ==========================================
    # 3. FEATURE EXTRACTION DEMO
    # ==========================================
    print("\n=== 3. Feature Extraction Demo ===")

    # Extract features for a small subset of Train
    # This tests BSON reading, Image processing, and Model forward pass
    extract_features_for_split(
        split_name="Train_Demo",
        meta_path=Config.TRAIN_META_PATH,
        bson_path=Config.TRAIN_BSON_PATH,
        feat_save_path=Config.TRAIN_FEATURES_PATH,
        target_save_path=Config.TRAIN_LABELS_PATH,
        is_test=False,
        subset_size=SUBSET_SIZE,
    )

    # Verify files exist and shapes are correct
    assert os.path.exists(Config.TRAIN_FEATURES_PATH), "Train features file missing."
    assert os.path.exists(Config.TRAIN_LABELS_PATH), "Train labels file missing."

    train_feats = np.load(Config.TRAIN_FEATURES_PATH)
    train_labels = np.load(Config.TRAIN_LABELS_PATH)

    print(f"Extracted Train Features Shape: {train_feats.shape}")
    print(f"Extracted Train Labels Shape: {train_labels.shape}")

    assert train_feats.shape == (
        SUBSET_SIZE,
        Config.DIM_INPUT,
    ), f"Expected shape ({SUBSET_SIZE}, {Config.DIM_INPUT}), got {train_feats.shape}"
    assert train_labels.shape == (
        SUBSET_SIZE,
    ), f"Expected shape ({SUBSET_SIZE},), got {train_labels.shape}"

    # For the purpose of this demo, we will copy the train features to be validation features
    # to save time running extraction again.
    shutil.copy(Config.TRAIN_FEATURES_PATH, Config.VAL_FEATURES_PATH)
    shutil.copy(Config.TRAIN_LABELS_PATH, Config.VAL_LABELS_PATH)
    print("Simulated Validation set by copying Train set.")

    # ==========================================
    # 4. TRAINING DEMO
    # ==========================================
    print("\n=== 4. Training Demo ===")

    # Create DataLoaders
    train_loader = get_training_loader(
        features_path=Config.TRAIN_FEATURES_PATH,
        labels_path=Config.TRAIN_LABELS_PATH,
        hierarchy_mapper=mapper,
        shuffle=True,
        subset_size=SUBSET_SIZE,
    )

    val_loader = get_training_loader(
        features_path=Config.VAL_FEATURES_PATH,
        labels_path=Config.VAL_LABELS_PATH,
        hierarchy_mapper=mapper,
        shuffle=False,
        subset_size=SUBSET_SIZE,
    )

    # Initialize Model
    model = DeepFeatureCascade()

    # Run Training
    # train_model handles the loop, optimizer, loss, and saving best model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name="demo_model.pth",
    )

    assert os.path.exists(
        os.path.join(Config.MODEL_DIR, "demo_model.pth")
    ), "Saved model file missing."
    print("Training loop completed successfully.")

    # ==========================================
    # 5. INFERENCE DEMO
    # ==========================================
    print("\n=== 5. Inference Demo ===")

    # 5.1 Extract Test Features (Subset)
    extract_features_for_split(
        split_name="Test_Demo",
        meta_path=Config.TEST_META_PATH,
        bson_path=Config.TEST_BSON_PATH,
        feat_save_path=Config.TEST_FEATURES_PATH,
        target_save_path=Config.TEST_IDS_PATH,
        is_test=True,
        subset_size=20,  # Smaller subset for test
    )

    # 5.2 Create Test Loader
    test_loader = get_test_loader(
        features_path=Config.TEST_FEATURES_PATH,
        ids_path=Config.TEST_IDS_PATH,
        subset_size=20,
    )

    # 5.3 Run Prediction Loop
    trained_model.eval()
    all_preds = []
    all_ids = []

    print("Running inference...")
    with torch.no_grad():
        for features, prod_ids in test_loader:
            features = features.to(Config.DEVICE)

            # Forward pass
            _, _, out_l3 = trained_model(features)

            # Get predictions (argmax of L3 logits)
            preds = torch.argmax(out_l3, dim=1).cpu().numpy()

            all_preds.append(preds)
            all_ids.append(prod_ids.numpy())

    all_preds = np.concatenate(all_preds)
    all_ids = np.concatenate(all_ids)

    print(f"Predictions shape: {all_preds.shape}")

    # 5.4 Map back to Category IDs
    predicted_category_ids = mapper.get_category_id_from_label(all_preds)

    # 5.5 Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"_id": all_ids, "category_id": predicted_category_ids}
    )

    print("Sample Submission:")
    print(submission_df.head())

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
