import os
import shutil
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.feature_engineering import FeatureEngineer
from library.dataset import GestureDataset
from library.model import CascadedNet
from library.losses import CombinedLoss
from library.trainer import Trainer
from library.inference import generate_test_predictions
from library.utils import decode_predictions


def run_demo():
    # ==========================================
    # 1. Setup & Configuration Override
    # ==========================================
    print("=== Setting up Demonstration Environment ===")

    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    # Define a temporary working directory for this demo
    DEMO_DIR = os.path.join("working", "demo_execution")
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config parameters for speed
    Config.WORKING_DIR = DEMO_DIR
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 2
    Config.EARLY_STOPPING_PATIENCE = 2
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "demo_model.pth")
    Config.SUBMISSION_FILE_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Override Cache Paths
    Config.TRAIN_CACHE_PATH = os.path.join(DEMO_DIR, "train_feat.npz")
    Config.VAL_CACHE_PATH = os.path.join(DEMO_DIR, "val_feat.npz")
    Config.TEST_CACHE_PATH = os.path.join(DEMO_DIR, "test_feat.npz")

    print(f"Working directory set to: {DEMO_DIR}")

    # ==========================================
    # 2. Data Preparation (Mini-Subset)
    # ==========================================
    print("\n=== Creating Mini-Datasets ===")

    # Load original metadata
    df_train_full = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val_full = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test_full = pd.read_csv(Config.TEST_METADATA_PATH)

    # Take a small subset (e.g., 5 samples)
    subset_size = 5
    df_train_mini = df_train_full.head(subset_size)
    df_val_mini = df_val_full.head(subset_size)
    df_test_mini = df_test_full.head(subset_size)

    # Save mini metadata
    mini_train_path = os.path.join(DEMO_DIR, "mini_train.csv")
    mini_val_path = os.path.join(DEMO_DIR, "mini_val.csv")
    mini_test_path = os.path.join(DEMO_DIR, "mini_test.csv")

    df_train_mini.to_csv(mini_train_path, index=False)
    df_val_mini.to_csv(mini_val_path, index=False)
    df_test_mini.to_csv(mini_test_path, index=False)

    # Update Config to point to mini metadata
    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path
    Config.TEST_METADATA_PATH = mini_test_path

    print(f"Created mini datasets with {subset_size} samples each.")

    # ==========================================
    # 3. Feature Engineering Demo
    # ==========================================
    print("\n=== Running Feature Engineering ===")

    fe = FeatureEngineer()

    # Process Training Data
    print("Processing training subset...")
    train_data = fe.process_dataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        cache_path=Config.TRAIN_CACHE_PATH,
        load_cached_data=False,  # Force processing
    )

    # Validation of Feature Engineering Output
    assert "features" in train_data
    assert "labels" in train_data
    assert train_data["features"].shape[1] == Config.INPUT_DIM
    print(f"Feature shape verified: {train_data['features'].shape}")
    print(f"Labels shape verified: {train_data['labels'].shape}")

    # Process Validation Data
    print("Processing validation subset...")
    val_data = fe.process_dataset(
        metadata_path=Config.VAL_METADATA_PATH,
        cache_path=Config.VAL_CACHE_PATH,
        load_cached_data=False,
    )

    # ==========================================
    # 4. Dataset & DataLoader Demo
    # ==========================================
    print("\n=== Initializing Datasets and Loaders ===")

    # Train Dataset (Windowed)
    train_dataset = GestureDataset(train_data, is_train=True)
    # Val Dataset (Full Sequence)
    val_dataset = GestureDataset(val_data, is_train=False)

    # Check Dataset Lengths
    print(f"Training Windows: {len(train_dataset)}")
    print(f"Validation Sequences: {len(val_dataset)}")

    # Create Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Validate Train Loader Batch
    if len(train_loader) > 0:
        feat_batch, label_batch = next(iter(train_loader))
        # Expected: (Batch, Window, InputDim)
        assert feat_batch.dim() == 3
        assert feat_batch.shape[0] == Config.BATCH_SIZE
        assert feat_batch.shape[1] == Config.WINDOW_SIZE
        assert feat_batch.shape[2] == Config.INPUT_DIM
        # Expected: (Batch, Window)
        assert label_batch.dim() == 2
        assert label_batch.shape[0] == Config.BATCH_SIZE
        assert label_batch.shape[1] == Config.WINDOW_SIZE
        print("Train Loader batch shapes verified.")
    else:
        print(
            "Warning: Train loader is empty (sequences might be shorter than window size)."
        )

    # ==========================================
    # 5. Model Demo
    # ==========================================
    print("\n=== Initializing Model ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CascadedNet().to(device)

    # Test Forward Pass with Dummy Data
    dummy_input = torch.randn(2, Config.WINDOW_SIZE, Config.INPUT_DIM).to(device)
    s1_logits, s2_logits = model(dummy_input)

    # Check Output Shapes: (Batch, Time, NumClasses)
    assert s1_logits.shape == (2, Config.WINDOW_SIZE, Config.NUM_CLASSES)
    assert s2_logits.shape == (2, Config.WINDOW_SIZE, Config.NUM_CLASSES)
    print("Model forward pass successful. Output shapes verified.")

    # ==========================================
    # 6. Loss Function Demo
    # ==========================================
    print("\n=== Testing Loss Function ===")

    criterion = CombinedLoss().to(device)
    dummy_targets = torch.randint(0, Config.NUM_CLASSES, (2, Config.WINDOW_SIZE)).to(
        device
    )

    loss_dict = criterion(s1_logits, s2_logits, dummy_targets)

    assert "loss" in loss_dict
    assert "ce1" in loss_dict
    assert "ce2" in loss_dict
    assert "smooth" in loss_dict
    # Ensure loss is a scalar tensor
    assert loss_dict["loss"].dim() == 0
    print(f"Loss calculation successful. Total Loss: {loss_dict['loss'].item():.4f}")

    # ==========================================
    # 7. Training Loop Demo
    # ==========================================
    print("\n=== Running Training Loop ===")

    trainer = Trainer(train_loader, val_loader, device=device)

    # Run training
    # This will train for Config.NUM_EPOCHS (set to 2)
    trainer.fit()

    # Verify model checkpoint creation
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Model checkpoint successfully saved to {Config.MODEL_SAVE_PATH}")
    else:
        raise FileNotFoundError("Model checkpoint was not created!")

    # ==========================================
    # 8. Inference Demo
    # ==========================================
    print("\n=== Running Inference ===")

    # Generate predictions on the mini test set
    # Note: We pass load_cached_data=False to ensure it processes our new mini subset
    predictions = generate_test_predictions(load_cached_data=False, device=device)

    # Verify predictions structure
    assert isinstance(predictions, dict)
    assert len(predictions) > 0

    # Check submission file
    if os.path.exists(Config.SUBMISSION_FILE_PATH):
        print(
            f"Submission file successfully generated at {Config.SUBMISSION_FILE_PATH}"
        )

        # Read and verify content format
        with open(Config.SUBMISSION_FILE_PATH, "r") as f:
            lines = f.readlines()
            if len(lines) > 0:
                first_line = lines[0].strip().split(",")
                # Format: SessionID, Label1, Label2...
                print(f"Sample submission line: {lines[0].strip()}")
                assert len(first_line) >= 1  # At least SessionID
    else:
        raise FileNotFoundError("Submission file was not created!")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
