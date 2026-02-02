import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.dataset import RNADataset
from library.model import DeepStabilizedBiGRU
from library.engine import train_fn, eval_fn


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration Overrides
    set_seed(Config.SEED)

    # Override Config for rapid demonstration
    print("Configuring for fast demonstration...")
    Config.DEBUG = True  # Use small data subset (50 samples)
    Config.NUM_EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update cache paths to avoid conflicts with existing full runs
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_cache.npz")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_cache.npz")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_cache.npz")

    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading
    print("\n=== Loading Datasets ===")
    # Initialize Datasets
    train_dataset = RNADataset(split="train", debug=Config.DEBUG)
    val_dataset = RNADataset(split="val", debug=Config.DEBUG)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Verify Data Shapes
    sample = train_dataset[0]
    seq_shape = sample["sequence"].shape
    pair_shape = sample["pair_indices"].shape
    target_shape = sample["targets"].shape

    print(f"Sample Sequence Shape: {seq_shape} (Expected: 107, 14)")
    print(f"Sample Targets Shape: {target_shape} (Expected: 68, 5)")

    assert seq_shape == (107, 14), f"Incorrect sequence shape: {seq_shape}"
    assert pair_shape == (107,), f"Incorrect pair indices shape: {pair_shape}"
    assert target_shape == (68, 5), f"Incorrect target shape: {target_shape}"

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for simple script execution
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # 3. Model Initialization
    print("\n=== Initializing Model ===")
    model = DeepStabilizedBiGRU().to(device)

    # Dummy Forward Pass Verification
    dummy_batch = next(iter(train_loader))
    dummy_seq = dummy_batch["sequence"].to(device)
    dummy_pair = dummy_batch["pair_indices"].to(device)

    with torch.no_grad():
        dummy_out = model(dummy_seq, dummy_pair)

    print(f"Model Output Shape: {dummy_out.shape}")
    # Model outputs predictions for the full sequence length (107)
    assert dummy_out.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), f"Model output shape mismatch. Expected ({Config.BATCH_SIZE}, 107, 5), got {dummy_out.shape}"

    # 4. Training Loop
    print("\n=== Starting Training Loop ===")
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.ETA_MIN
    )

    best_score = float("inf")

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")

        # Train
        train_loss = train_fn(model, train_loader, optimizer, device, scheduler)
        print(f"  Train Loss (MCRMSE): {train_loss:.4f}")

        # Validate
        val_score = eval_fn(model, val_loader, device)
        print(f"  Val Score (MCRMSE on scored cols): {val_score:.4f}")

        # Save Best
        if val_score < best_score:
            best_score = val_score
            save_path = os.path.join(Config.WORKING_DIR, "demo_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model to {save_path}")

    # 5. Inference and Submission
    print("\n=== Generating Submission ===")

    # Load Test Data
    test_dataset = RNADataset(split="test", debug=Config.DEBUG)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Load Best Model
    model.load_state_dict(
        torch.load(os.path.join(Config.WORKING_DIR, "demo_model.pth"))
    )
    model.eval()

    ids_list = []
    preds_list = []

    print("Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            sequence = batch["sequence"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            batch_ids = batch["id"]

            # Forward pass
            outputs = model(sequence, pair_indices)  # (B, 107, 5)
            outputs = outputs.cpu().numpy()

            ids_list.extend(batch_ids)
            preds_list.append(outputs)

    # Concatenate all predictions
    all_preds = np.concatenate(preds_list, axis=0)  # (N_samples, 107, 5)

    # Prepare Submission DataFrame
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_rows = []

    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # (107, 5)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos].tolist()

            row_dict = {"id_seqpos": row_id}
            for col_name, val in zip(target_cols, row_values):
                row_dict[col_name] = val

            submission_rows.append(row_dict)

    submission_df = pd.DataFrame(submission_rows)

    # Verify Submission Structure
    print(f"Submission DataFrame Shape: {submission_df.shape}")
    expected_rows = len(test_dataset) * Config.SEQ_LEN
    assert (
        len(submission_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(submission_df)}"

    # Save Submission
    sub_path = os.path.join(Config.WORKING_DIR, "submission_demo.csv")
    submission_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
