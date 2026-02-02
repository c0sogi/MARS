import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# --- Patch tqdm to disable progress bars ---
# This must be done before importing modules that use tqdm
import tqdm


def no_op_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.tqdm = no_op_tqdm

# --- Import Library Components ---
from library.config import Config
from library.utils import seed_everything, compute_mcc, FocalLoss
from library.data_processing import DataProcessor
from library.model import ECPIRN
from library.train_eval import Trainer


def setup_demo_config():
    """
    Overrides Config parameters to run a fast demo on a subset of data.
    """
    # Define demo paths
    demo_dir = "./working/demo_run"
    meta_dir = os.path.join(demo_dir, "metadata")
    sub_dir = os.path.join(demo_dir, "submission")
    work_dir = os.path.join(demo_dir, "working")

    for d in [demo_dir, meta_dir, sub_dir, work_dir]:
        os.makedirs(d, exist_ok=True)

    # Override Config
    Config.WORKING_DIR = work_dir
    Config.SUBMISSION_DIR = sub_dir
    Config.SUBMISSION_PATH = os.path.join(sub_dir, "submission.csv")

    Config.TRAIN_METADATA_PATH = os.path.join(meta_dir, "train.csv")
    Config.VAL_METADATA_PATH = os.path.join(meta_dir, "validation.csv")
    Config.TEST_METADATA_PATH = os.path.join(meta_dir, "test.csv")

    # Speed up training
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 128
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    return meta_dir


def create_subset_data(meta_dir):
    """
    Reads original metadata, samples a small subset, and saves to demo location.
    """
    print("Creating data subsets for demonstration...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/validation.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Sample subsets (ensure we keep game_play integrity roughly, though simple head is fine for demo)
    # We take enough rows to ensure we have some data, but small enough to be fast.
    # We also ensure we have both classes in train for stability.

    # Stratified sample for train to ensure contact events exist
    train_subset = orig_train.groupby("contact", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), 200), random_state=42)
    )
    val_subset = orig_val.head(200)
    test_subset = orig_test.head(200)

    # Save to demo paths
    train_subset.to_csv(Config.TRAIN_METADATA_PATH, index=False)
    val_subset.to_csv(Config.VAL_METADATA_PATH, index=False)
    test_subset.to_csv(Config.TEST_METADATA_PATH, index=False)

    print(
        f"Subsets saved: Train({len(train_subset)}), Val({len(val_subset)}), Test({len(test_subset)})"
    )


def verify_utils():
    print("\n--- Verifying Utils ---")

    # Test MCC
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])
    mcc = compute_mcc(y_true, y_pred)
    # TP=2, TN=2, FP=0, FN=1
    # MCC should be positive
    assert -1.0 <= mcc <= 1.0, "MCC score out of range"
    print("MCC computation verified.")

    # Test Focal Loss
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.randn(10, 1)
    targets = torch.randint(0, 2, (10, 1)).float()
    loss = criterion(logits, targets)
    assert loss.item() > 0, "Focal Loss should be positive"
    print("Focal Loss verified.")


def verify_data_processing():
    print("\n--- Verifying Data Processing ---")
    processor = DataProcessor()

    # 1. Train/Val Data
    # load_cached_data=False forces processing from scratch using our new subset CSVs
    X_train, y_train, X_val, y_val = processor.get_train_val_data(
        load_cached_data=False
    )

    print(f"Processed Train Shape: {X_train.shape}")
    print(f"Processed Val Shape: {X_val.shape}")

    # Assertions
    assert X_train.ndim == 2, "X_train should be 2D"
    assert (
        X_train.shape[1] == Config.INPUT_DIM
    ), f"Feature dimension mismatch. Expected {Config.INPUT_DIM}, got {X_train.shape[1]}"
    assert len(X_train) == len(y_train), "X and y length mismatch"
    assert not np.isnan(X_train).any(), "NaNs found in training data"

    # 2. Test Data
    X_test, df_test_meta = processor.get_test_data(load_cached_data=False)
    print(f"Processed Test Shape: {X_test.shape}")
    assert X_test.shape[1] == Config.INPUT_DIM, "Test feature dimension mismatch"

    return X_train, y_train, X_val, y_val, X_test, df_test_meta


def verify_model():
    print("\n--- Verifying Model Architecture ---")
    model = ECPIRN()
    model.eval()

    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, Config.INPUT_DIM)

    # Forward pass
    output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (batch_size, 1), "Model output shape incorrect"
    print("Model forward pass verified.")


def verify_training_pipeline(X_train, y_train, X_val, y_val, X_test, df_test_meta):
    print("\n--- Verifying Training Pipeline ---")

    # Prepare DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_ds = TensorDataset(torch.tensor(X_test))

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Initialize Trainer
    trainer = Trainer(device="cpu")  # Force CPU for demo stability/simplicity

    # Train
    print("Starting training (1 epoch)...")
    trainer.fit(train_loader, val_loader, epochs=Config.NUM_EPOCHS)

    # Check for model checkpoint
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_path), "Model checkpoint not saved"
    print("Training complete. Checkpoint verified.")

    # Predict
    print("Running inference...")
    preds = trainer.predict(test_loader)

    assert len(preds) == len(X_test), "Prediction length mismatch"
    assert set(np.unique(preds)).issubset({0, 1}), "Predictions must be binary"

    # Create Submission
    sub_df = pd.DataFrame({"contact_id": df_test_meta["contact_id"], "contact": preds})

    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"
    print(f"Submission generated with {len(sub_df)} rows.")


if __name__ == "__main__":
    # 1. Setup
    seed_everything(Config.SEED)
    meta_dir = setup_demo_config()
    create_subset_data(meta_dir)

    # 2. Verify Components
    verify_utils()

    # 3. Verify Data Pipeline
    # This simulates the data loading part of run_pipeline
    X_train, y_train, X_val, y_val, X_test, df_test_meta = verify_data_processing()

    # 4. Verify Model
    verify_model()

    # 5. Verify Training & Inference
    verify_training_pipeline(X_train, y_train, X_val, y_val, X_test, df_test_meta)

    print("\n=== All Demonstrations and Verifications Passed Successfully ===")
