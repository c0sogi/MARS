import os
import shutil
import pandas as pd
import torch
import numpy as np

# Import library modules
from library.config import Config
from library.utils import (
    set_seed,
    compute_levenshtein,
    decode_predictions,
    smooth_predictions,
)
from library.data import load_data, GestureDataset, collate_fn
from library.model import SG_CRCN
from library.loss import TotalLoss
from library.train import Trainer


def main():
    # -------------------------------------------------------------------------
    # 1. Setup Environment for Demo
    # -------------------------------------------------------------------------
    print(">>> Setting up demo environment...")
    DEMO_DIR = "./working/demo_task"
    DEMO_META_DIR = os.path.join(DEMO_DIR, "metadata")
    DEMO_CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    DEMO_CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    DEMO_SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

    # Clean up previous run if exists
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)

    os.makedirs(DEMO_META_DIR)
    os.makedirs(DEMO_CACHE_DIR)
    os.makedirs(DEMO_CHECKPOINT_DIR)
    os.makedirs(DEMO_SUBMISSION_DIR)

    # -------------------------------------------------------------------------
    # 2. Create Subset Metadata (Speed Optimization)
    # -------------------------------------------------------------------------
    # We read the original metadata and take a small sample to verify the pipeline
    orig_train_csv = os.path.join("./metadata", "train.csv")
    orig_val_csv = os.path.join("./metadata", "val.csv")
    orig_test_csv = os.path.join("./metadata", "test.csv")

    # Read and sample (ensure we have enough for a batch)
    df_train = pd.read_csv(orig_train_csv).head(21)  # 21 samples
    df_val = pd.read_csv(orig_val_csv).head(11)  # 11 samples
    df_test = pd.read_csv(orig_test_csv).head(6)  # 6 samples

    # Save to demo metadata dir
    df_train.to_csv(os.path.join(DEMO_META_DIR, "train.csv"), index=False)
    df_val.to_csv(os.path.join(DEMO_META_DIR, "val.csv"), index=False)
    df_test.to_csv(os.path.join(DEMO_META_DIR, "test.csv"), index=False)

    print(f"Created subset metadata in {DEMO_META_DIR}")

    # -------------------------------------------------------------------------
    # 3. Override Config
    # -------------------------------------------------------------------------
    print(">>> Overriding Config for Demo...")
    Config.METADATA_DIR = DEMO_META_DIR
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_CACHE_FILE = os.path.join(DEMO_CACHE_DIR, "train_data.npz")
    Config.VAL_CACHE_FILE = os.path.join(DEMO_CACHE_DIR, "val_data.npz")
    Config.TEST_CACHE_FILE = os.path.join(DEMO_CACHE_DIR, "test_data.npz")
    Config.MODEL_CHECKPOINT = os.path.join(DEMO_CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(DEMO_SUBMISSION_DIR, "submission.csv")

    # Hyperparameters for speed
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 2
    Config.LSTM_LAYERS = 1  # Reduce model size for speed
    Config.NUM_TCN_LAYERS = 4  # Reduce TCN depth
    Config.HIDDEN_DIM = 64  # Reduce width
    Config.DROPOUT = 0.1

    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 4. Verify Utility Functions
    # -------------------------------------------------------------------------
    print(">>> Verifying Utility Functions...")
    # Levenshtein
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist = compute_levenshtein(seq1, seq2)
    assert dist == 0, f"Levenshtein distance should be 0, got {dist}"

    seq1 = [1, 2, 3]
    seq2 = [1, 2]
    dist = compute_levenshtein(seq1, seq2)
    assert dist == 1, f"Levenshtein distance should be 1, got {dist}"

    # Decode Predictions
    # 0 is background.
    raw_preds = [0, 0, 1, 1, 1, 0, 2, 2, 0, 0, 3]
    decoded = decode_predictions(raw_preds, background_class=0)
    assert decoded == [1, 2, 3], f"Decoding failed. Got {decoded}"

    # Smooth Predictions
    noisy_preds = np.array([1, 1, 2, 1, 1])  # Median window 3 should fix the 2 -> 1
    smoothed = smooth_predictions(noisy_preds, window_size=3)
    # Center element: window [1, 2, 1] -> median 1
    assert smoothed[2] == 1, f"Smoothing failed. Got {smoothed}"
    print("Utilities verified.")

    # -------------------------------------------------------------------------
    # 5. Verify Data Loading & Processing
    # -------------------------------------------------------------------------
    print(">>> Verifying Data Loading...")
    # This will generate cache files in the demo dir
    train_data = load_data("train")
    assert "positions" in train_data
    assert len(train_data["ids"]) == 21

    # Create Dataset
    ds = GestureDataset(train_data, augment=False)
    item = ds[0]
    # Check item keys
    assert "features" in item
    assert "labels" in item
    assert "boundaries" in item

    # Check feature dimensions: (Time, 85)
    # 12 joints * 3 pos + 12 joints * 3 vel + 13 mfcc = 36 + 36 + 13 = 85
    feat_dim = item["features"].shape[1]
    assert feat_dim == 85, f"Expected feature dim 85, got {feat_dim}"
    print("Data loading verified.")

    # -------------------------------------------------------------------------
    # 6. Verify Model & Loss
    # -------------------------------------------------------------------------
    print(">>> Verifying Model & Loss...")
    # Create a batch
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn)
    batch = next(iter(loader))

    features = batch["features"]  # (B, T, 85)
    mask = batch["mask"]  # (B, T)

    # Initialize model with the reduced hyperparameters set in Config
    model = SG_CRCN()

    # Forward pass
    outputs = model(features, mask)

    # Check outputs
    assert "stage3_cls" in outputs
    assert "stage3_bnd" in outputs

    # Check shape: (B, T, NumClasses)
    b, t, c = outputs["stage3_cls"].shape
    assert b == Config.BATCH_SIZE
    assert c == Config.NUM_CLASSES  # 21

    # Loss
    criterion = TotalLoss()
    loss, metrics = criterion(outputs, batch)

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0
    print("Model & Loss verified.")

    # -------------------------------------------------------------------------
    # 7. Verify Training Loop
    # -------------------------------------------------------------------------
    print(">>> Verifying Training Loop...")
    # Force CPU for demo stability/simplicity, or let it use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(device=device)

    # Load datasets (will use the cached subset data we generated)
    trainer.load_datasets(batch_size=Config.BATCH_SIZE)

    # Run fit (Config.NUM_EPOCHS is set to 2)
    trainer.fit()

    # Check if model saved
    assert os.path.exists(
        Config.MODEL_CHECKPOINT
    ), "Model checkpoint not found after training."
    print("Training loop verified.")

    print("\n>>> All demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    main()
