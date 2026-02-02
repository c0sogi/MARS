import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.architecture import DiGUT
from library.data_factory import preprocess_data, ManufacturingDataset
from library.trainer import train_one_epoch, evaluate, apply_swap_noise

# -----------------------------------------------------------------------------
# 1. Setup and Helper Functions
# -----------------------------------------------------------------------------


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_dummy_data(output_dir, num_rows=100, is_test=False):
    """Generates a dummy dataframe matching the competition schema."""
    os.makedirs(output_dir, exist_ok=True)

    # Generate numerical features f_00 to f_30
    data = {f"f_{i:02d}": np.random.randn(num_rows) for i in range(31)}

    # Generate sequence feature f_27 (random strings of length 10)
    chars = list("ABCDEFGHIJ")
    data["f_27"] = ["".join(np.random.choice(chars, 10)) for _ in range(num_rows)]

    # ID and Source Path
    start_id = 0 if not is_test else 1000
    data["id"] = np.arange(start_id, start_id + num_rows)
    data["source_path"] = "test.csv" if is_test else "train.csv"

    # Target (only for train/val)
    if not is_test:
        data["target"] = np.random.randint(0, 2, num_rows)

    df = pd.DataFrame(data)
    return df


# -----------------------------------------------------------------------------
# 2. Main Execution Block
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting DiGUT Library Demonstration...")

    # Set seed for reproducibility
    set_seed(42)

    # Define paths for demo
    base_dir = "./working/demo"
    metadata_dir = os.path.join(base_dir, "metadata")
    cache_dir = os.path.join(base_dir, "cache")
    submission_dir = os.path.join(base_dir, "submission")

    # Clean up previous runs if any
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(metadata_dir)

    # -------------------------------------------------------------------------
    # Step 1: Generate Mock Data
    # -------------------------------------------------------------------------
    print("\n[1] Generating mock data...")

    train_df = generate_dummy_data(metadata_dir, num_rows=50, is_test=False)
    val_df = generate_dummy_data(metadata_dir, num_rows=20, is_test=False)
    test_df = generate_dummy_data(metadata_dir, num_rows=20, is_test=True)

    train_path = os.path.join(metadata_dir, "train.csv")
    val_path = os.path.join(metadata_dir, "val.csv")
    test_path = os.path.join(metadata_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Mock data created at {metadata_dir}")

    # -------------------------------------------------------------------------
    # Step 2: Configure
    # -------------------------------------------------------------------------
    print("\n[2] Configuring pipeline...")

    # Create a custom config for the demo
    demo_config = Config(
        METADATA_DIR=metadata_dir,
        WORKING_DIR=base_dir,
        SUBMISSION_DIR=submission_dir,
        TRAIN_PATH=train_path,
        VAL_PATH=val_path,
        TEST_PATH=test_path,
        CACHE_DIR=cache_dir,
        # Reduce model size for speed
        HIDDEN_DIM=32,
        NUM_LAYERS=2,
        NUM_HEADS=4,
        FORWARD_DIM=64,
        # Reduce training parameters
        BATCH_SIZE=8,
        EPOCHS=1,
        NUM_WORKERS=0,  # Avoid multiprocessing overhead in demo
    )

    demo_config.display()

    # -------------------------------------------------------------------------
    # Step 3: Data Preprocessing
    # -------------------------------------------------------------------------
    print("\n[3] Running Data Preprocessing...")

    (
        X_num_train,
        X_seq_train,
        y_train,
        X_num_val,
        X_seq_val,
        y_val,
        X_num_test,
        X_seq_test,
        ids_test,
        meta,
    ) = preprocess_data(demo_config, load_cached_data=False)

    # Verify shapes
    print("Verifying data shapes...")
    assert X_num_train.shape == (
        50,
        32,
    ), f"Expected (50, 32), got {X_num_train.shape}"  # 31 features + unique_chars
    assert X_seq_train.shape == (50, 10), f"Expected (50, 10), got {X_seq_train.shape}"
    assert y_train.shape == (50,), f"Expected (50,), got {y_train.shape}"
    assert meta["num_numerical_features"] == 32
    assert meta["sequence_length"] == 10
    print("Data shapes verified.")

    # -------------------------------------------------------------------------
    # Step 4: Dataset and DataLoader
    # -------------------------------------------------------------------------
    print("\n[4] Testing Dataset and DataLoader...")

    # Instantiate dataset
    train_dataset = ManufacturingDataset(
        X_num_train, X_seq_train, y_train, is_train=False, config=demo_config
    )

    # Check __getitem__
    x_n, x_s, y, mask = train_dataset[0]
    assert isinstance(x_n, torch.FloatTensor)
    assert isinstance(x_s, torch.LongTensor)
    assert x_n.shape == (32,)
    assert x_s.shape == (10,)

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=demo_config.BATCH_SIZE, shuffle=True, drop_last=True
    )

    # Fetch one batch
    batch_x_n, batch_x_s, batch_y, batch_mask = next(iter(train_loader))
    assert batch_x_n.shape == (8, 32)
    assert batch_x_s.shape == (8, 10)
    print("Dataset and DataLoader verified.")

    # -------------------------------------------------------------------------
    # Step 5: Model Initialization and Forward Pass
    # -------------------------------------------------------------------------
    print("\n[5] Testing Model Architecture...")

    device = demo_config.DEVICE
    model = DiGUT(
        num_numerical_features=meta["num_numerical_features"],
        vocab_size=meta["vocab_size"],
        sequence_length=meta["sequence_length"],
        config=demo_config,
    ).to(device)

    # Move batch to device
    batch_x_n = batch_x_n.to(device)
    batch_x_s = batch_x_s.to(device)

    # Forward pass
    target_logits, disc_logits = model(batch_x_n, batch_x_s)

    # Check output shapes
    # target_logits: (B, 1)
    assert target_logits.shape == (8, 1)
    # disc_logits: (B, Num_Feats + Seq_Len, 1) -> (8, 32 + 10, 1)
    expected_tokens = 32 + 10
    assert disc_logits.shape == (8, expected_tokens, 1)

    print("Model forward pass verified.")

    # -------------------------------------------------------------------------
    # Step 6: Testing Swap Noise Function
    # -------------------------------------------------------------------------
    print("\n[6] Testing Swap Noise Logic...")

    x_num_corr, x_seq_corr, mask_combined = apply_swap_noise(
        batch_x_n, batch_x_s, swap_prob=0.5
    )

    assert x_num_corr.shape == batch_x_n.shape
    assert x_seq_corr.shape == batch_x_s.shape
    assert mask_combined.shape == (8, expected_tokens)

    # Ensure some corruption happened (probability is high)
    # Check if mask has any 1s
    assert mask_combined.sum() > 0, "Swap noise mask should contain some 1s with p=0.5"
    print("Swap noise logic verified.")

    # -------------------------------------------------------------------------
    # Step 7: Training Loop Verification
    # -------------------------------------------------------------------------
    print("\n[7] Testing Training Loop...")

    optimizer = optim.AdamW(model.parameters(), lr=demo_config.LEARNING_RATE)
    scheduler = None  # Skip scheduler for simple demo

    # Run one epoch
    loss = train_one_epoch(
        model, train_loader, optimizer, scheduler, device, demo_config
    )

    assert not np.isnan(loss), "Training loss is NaN"
    assert loss > 0, "Training loss should be positive"
    print(f"Train Loop successful. Loss: {loss:.4f}")

    # -------------------------------------------------------------------------
    # Step 8: Evaluation Verification
    # -------------------------------------------------------------------------
    print("\n[8] Testing Evaluation...")

    val_loader = DataLoader(
        ManufacturingDataset(
            X_num_val, X_seq_val, y_val, is_train=False, config=demo_config
        ),
        batch_size=demo_config.BATCH_SIZE,
        shuffle=False,
    )

    auc, preds = evaluate(model, val_loader, device)

    assert 0.0 <= auc <= 1.0, f"AUC {auc} is out of bounds"
    assert len(preds) == 20, f"Expected 20 predictions, got {len(preds)}"
    print(f"Evaluation successful. AUC: {auc:.4f}")

    # -------------------------------------------------------------------------
    # Step 9: Submission Generation
    # -------------------------------------------------------------------------
    print("\n[9] Generating Submission...")

    test_loader = DataLoader(
        ManufacturingDataset(
            X_num_test, X_seq_test, None, is_train=False, config=demo_config
        ),
        batch_size=demo_config.BATCH_SIZE,
        shuffle=False,
    )

    _, test_preds = evaluate(model, test_loader, device)

    submission = pd.DataFrame({"id": ids_test, "target": test_preds.flatten()})

    assert len(submission) == 20
    assert list(submission.columns) == ["id", "target"]

    sub_path = demo_config.SUBMISSION_PATH
    submission.to_csv(sub_path, index=False)

    assert os.path.exists(sub_path)
    print(f"Submission generated at {sub_path}")

    print("\nAll verification steps completed successfully.")
