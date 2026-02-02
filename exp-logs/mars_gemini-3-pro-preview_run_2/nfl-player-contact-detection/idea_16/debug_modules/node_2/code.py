import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data_processing import DataProcessor
from library.dataset import ContactDataset
from library.model import DCN
from library.train_eval import train_model


def main():
    # 1. Setup & Configuration Patching
    print("--- Setting up Demo Environment ---")
    seed_everything(Config.SEED)

    # Define demo directories
    DEMO_DIR = "./working/demo_run"
    DEMO_DATA_DIR = os.path.join(DEMO_DIR, "data")
    DEMO_SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DATA_DIR, exist_ok=True)
    os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)

    # Patch Config to use demo directories and reduced parameters
    Config.WORKING_DIR = os.path.join(DEMO_DIR, "working")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    Config.SUBMISSION_DIR = DEMO_SUBMISSION_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_SUBMISSION_DIR, "submission.csv")

    Config.CACHE_TRAIN_FEATURES = os.path.join(
        Config.WORKING_DIR, "train_features.parquet"
    )
    Config.CACHE_VAL_FEATURES = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.CACHE_TEST_FEATURES = os.path.join(
        Config.WORKING_DIR, "test_features.parquet"
    )

    Config.SCALER_PATH = os.path.join(Config.WORKING_DIR, "scaler.joblib")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.THRESHOLD_PATH = os.path.join(Config.WORKING_DIR, "best_threshold.npy")

    # Speed up training for demo
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 128
    Config.EARLY_STOPPING_PATIENCE = 1

    # 2. Create Data Subsets
    print("--- Creating Data Subsets ---")

    # Load original metadata
    orig_train = pd.read_csv(Config.TRAIN_LABELS_PATH)
    orig_val = pd.read_csv(Config.VAL_LABELS_PATH)
    orig_test = pd.read_csv(Config.TEST_META_PATH)

    # Sample data (ensure we have both classes in train)
    # Stratified sample for train to ensure contact events exist
    n_train = 2000
    n_pos = int(n_train * 0.1)  # Force 10% positive for demo stability
    n_neg = n_train - n_pos

    train_pos = orig_train[orig_train["contact"] == 1].sample(
        min(n_pos, len(orig_train[orig_train["contact"] == 1])),
        random_state=Config.SEED,
    )
    train_neg = orig_train[orig_train["contact"] == 0].sample(
        n_neg, random_state=Config.SEED
    )
    demo_train = (
        pd.concat([train_pos, train_neg])
        .sample(frac=1, random_state=Config.SEED)
        .reset_index(drop=True)
    )

    demo_val = orig_val.sample(500, random_state=Config.SEED).reset_index(drop=True)
    demo_test = orig_test.sample(500, random_state=Config.SEED).reset_index(drop=True)

    # Save subsets
    demo_train_path = os.path.join(DEMO_DATA_DIR, "train.csv")
    demo_val_path = os.path.join(DEMO_DATA_DIR, "val.csv")
    demo_test_path = os.path.join(DEMO_DATA_DIR, "test.csv")

    demo_train.to_csv(demo_train_path, index=False)
    demo_val.to_csv(demo_val_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    # Patch Config paths to point to subsets
    Config.TRAIN_LABELS_PATH = demo_train_path
    Config.VAL_LABELS_PATH = demo_val_path
    Config.TEST_META_PATH = demo_test_path

    print(
        f"Subset sizes -> Train: {len(demo_train)}, Val: {len(demo_val)}, Test: {len(demo_test)}"
    )

    # 3. Data Processing
    print("\n--- Running Data Processing ---")
    processor = DataProcessor()

    # Force load_cached=False to demonstrate processing logic
    X_train, y_train, X_val, y_val = processor.get_train_val_datasets(load_cached=False)

    # Verification
    print(f"Features shape: {X_train.shape}")
    assert X_train.shape[0] == len(
        demo_train
    ), f"Mismatch in train samples: {X_train.shape[0]} vs {len(demo_train)}"
    assert (
        X_train.shape[1] == Config.INPUT_DIM
    ), f"Mismatch in feature dim: {X_train.shape[1]} vs {Config.INPUT_DIM}"
    assert not np.isnan(X_train).any(), "NaNs found in training features"

    # 4. Dataset & DataLoader
    print("\n--- Initializing Datasets & Loaders ---")
    train_dataset = ContactDataset(X_train, y_train)
    val_dataset = ContactDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 0 for simple demo debugging
        drop_last=True,  # Drop last to ensure batch norm stability if batch is small
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # 5. Model Initialization & Verification
    print("\n--- Initializing Model ---")
    device = torch.device(Config.DEVICE)
    model = DCN().to(device)

    # Dummy forward pass
    dummy_input = torch.randn(Config.BATCH_SIZE, Config.INPUT_DIM).to(device)
    with torch.no_grad():
        dummy_out = model(dummy_input)

    assert dummy_out.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch: {dummy_out.shape}"
    print("Model architecture verified.")

    # 6. Training Loop
    print("\n--- Starting Training ---")
    # train_model handles the loop, saving, and threshold optimization
    trained_model = train_model(train_loader, val_loader)

    assert os.path.exists(Config.MODEL_PATH), "Best model file was not saved."
    assert os.path.exists(Config.THRESHOLD_PATH), "Threshold file was not saved."

    # 7. Inference & Submission
    print("\n--- Running Inference ---")
    X_test, test_ids = processor.get_test_dataset(load_cached=False)

    test_dataset = ContactDataset(X_test)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Load best threshold
    best_threshold = np.load(Config.THRESHOLD_PATH)[0]
    print(f"Using optimized threshold: {best_threshold:.4f}")

    trained_model.eval()
    all_preds = []

    with torch.no_grad():
        for features in test_loader:
            features = features.to(device)
            logits = trained_model(features)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= best_threshold).astype(int)
            all_preds.append(preds)

    all_preds = np.concatenate(all_preds).flatten()

    # Create Submission
    submission = pd.DataFrame({"contact_id": test_ids, "contact": all_preds})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Final Verification
    assert len(submission) == len(demo_test), "Submission length mismatch"
    assert submission["contact"].isin([0, 1]).all(), "Invalid values in prediction"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
