import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import shutil

# Import from provided libraries
from library.config import (
    INPUT_DIM,
    NUM_CLASSES,
    seed_everything,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
)
from library.data_loader import GestureDataset, collate_fn
from library.model import GestureGRU
from library.trainer import Trainer
from library.inference import Predictor
from library.utils import calculate_levenshtein_distance, rle_collapse

if __name__ == "__main__":
    # 1. Setup
    print(">>> Setting up demonstration environment...")
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a specific directory for this demo run
    demo_dir = os.path.join(WORKING_DIR, "demo_run")
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Update paths for demo to avoid messing with main config paths implicitly
    demo_train_meta = os.path.join(demo_dir, "train_subset.csv")
    demo_val_meta = os.path.join(demo_dir, "val_subset.csv")
    demo_test_meta = os.path.join(demo_dir, "test_subset.csv")

    # 2. Create Data Subsets for Speed
    print(">>> Creating data subsets for fast execution...")

    def create_subset(src_path, dst_path, n=5):
        if os.path.exists(src_path):
            df = pd.read_csv(src_path)
            # Sample n rows or take all if less than n
            subset = df.head(n)
            subset.to_csv(dst_path, index=False)
            print(f"    Created subset {dst_path} with {len(subset)} samples.")
        else:
            # Create dummy empty file if source doesn't exist (should not happen based on prompt)
            pd.DataFrame(columns=["sample_id"]).to_csv(dst_path, index=False)
            print(f"    Warning: Source {src_path} not found. Created empty subset.")

    create_subset(TRAIN_METADATA_PATH, demo_train_meta, n=8)  # Enough for one batch
    create_subset(VAL_METADATA_PATH, demo_val_meta, n=4)
    create_subset(TEST_METADATA_PATH, demo_test_meta, n=4)

    # 3. Data Loading Demonstration
    print("\n>>> Initializing Datasets and DataLoaders...")

    # Initialize Datasets (this will trigger feature extraction/caching for the subset)
    # We force load_cached_data=False to demonstrate the processing logic once,
    # but since we made a new directory, it would process anyway.
    train_dataset = GestureDataset(
        demo_train_meta, load_cached_data=False, mode="train"
    )
    val_dataset = GestureDataset(demo_val_meta, load_cached_data=False, mode="val")

    batch_size = 4
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    print(f"    Train Dataset Size: {len(train_dataset)}")
    print(f"    Val Dataset Size: {len(val_dataset)}")

    # Verify DataLoader output
    features, targets, lengths, ids = next(iter(train_loader))
    print(
        f"    Batch Shapes -> Features: {features.shape}, Targets: {targets.shape}, Lengths: {lengths.shape}"
    )

    # Assertions for shapes
    # Features: (Batch, Time, InputDim)
    assert features.dim() == 3
    assert features.shape[2] == INPUT_DIM
    # Targets: (Batch, Time)
    assert targets.dim() == 2
    assert features.shape[0] == targets.shape[0] == batch_size

    # 4. Model Demonstration
    print("\n>>> Initializing Model...")
    model = GestureGRU()
    model.to(device)

    # Verify Forward Pass
    features = features.to(device)
    lengths = lengths.to(device)
    logits = model(features, lengths)

    print(f"    Output Logits Shape: {logits.shape}")
    assert logits.shape == (batch_size, features.shape[1], NUM_CLASSES)

    # 5. Training Loop Demonstration
    print("\n>>> Starting Training Demonstration...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader, optimizer, device)

    # Override checkpoint path to use our demo directory
    trainer.checkpoint_dir = demo_dir
    trainer.checkpoint_path = os.path.join(demo_dir, "best_model_demo.pth")

    # Run for 2 epochs
    trainer.fit(epochs=2)

    # Verify Checkpoint creation
    assert os.path.exists(trainer.checkpoint_path), "Checkpoint file was not created!"
    print("    Training complete and checkpoint verified.")

    # 6. Inference Demonstration
    print("\n>>> Starting Inference Demonstration...")

    # Initialize Predictor with the trained model
    predictor = Predictor(model_path=trainer.checkpoint_path, device=device)

    # Run inference using the wrapper method
    submission_path = os.path.join(demo_dir, "submission_demo.csv")
    predictions = predictor.run_inference(
        test_metadata_path=demo_test_meta,
        output_filename=submission_path,
        batch_size=2,
        load_cached_data=False,  # Force process test data
    )

    # Verify Submission
    assert os.path.exists(submission_path), "Submission file was not created!"

    # Check submission content
    sub_df = pd.read_csv(submission_path, header=None)
    # Format is SessionID,labels... so 2 columns if read generically, or just lines
    # We expect at least the number of rows as in our test subset
    print(f"    Submission generated with {len(sub_df)} rows.")
    assert len(sub_df) == len(pd.read_csv(demo_test_meta))

    # 7. Utility Verification
    print("\n>>> Verifying Utilities...")

    # Test Levenshtein
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist_eq = calculate_levenshtein_distance(seq1, seq2)
    assert (
        dist_eq == 0
    ), f"Levenshtein distance for identical sequences should be 0, got {dist_eq}"

    seq3 = [1, 2, 3]
    seq4 = [1, 2, 4]
    dist_diff = calculate_levenshtein_distance(seq3, seq4)
    assert (
        dist_diff == 1
    ), f"Levenshtein distance for 1 substitution should be 1, got {dist_diff}"

    # Test RLE Collapse
    # 0 is background. Sequence: 0, 1, 1, 1, 0, 2, 2, 0, 1
    # Expected: [1, 2, 1]
    raw_seq = [0, 1, 1, 1, 0, 2, 2, 0, 1]
    collapsed = rle_collapse(raw_seq, remove_background=True, background_class=0)
    assert collapsed == [
        1,
        2,
        1,
    ], f"RLE Collapse failed. Expected [1, 2, 1], got {collapsed}"

    print("    Utilities verified.")

    print("\n>>> Demonstration Completed Successfully!")
