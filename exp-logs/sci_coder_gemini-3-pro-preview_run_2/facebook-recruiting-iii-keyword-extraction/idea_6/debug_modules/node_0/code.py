import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Creates a temporary directory and prepares subsampled datasets for the demo.
    """
    print("Setting up demo environment...")

    # Define paths
    base_dir = "./working/demo_run"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    # Define metadata paths
    meta_train_path = "./metadata/train.csv"
    meta_val_path = "./metadata/validation.csv"
    meta_test_path = "./metadata/test.csv"

    # Subsample size
    N_SAMPLES = 2000

    # Load and save subsamples
    print(f"Subsampling {N_SAMPLES} rows from metadata...")

    df_train = pd.read_csv(meta_train_path, nrows=N_SAMPLES)
    df_val = pd.read_csv(meta_val_path, nrows=N_SAMPLES)
    df_test = pd.read_csv(meta_test_path, nrows=N_SAMPLES)

    # Save to working dir
    train_dest = os.path.join(base_dir, "train.csv")
    val_dest = os.path.join(base_dir, "validation.csv")
    test_dest = os.path.join(base_dir, "test.csv")

    df_train.to_csv(train_dest, index=False)
    df_val.to_csv(val_dest, index=False)
    df_test.to_csv(test_dest, index=False)

    print("Subsampled data saved.")
    return base_dir, train_dest, val_dest, test_dest


def monkey_patch_library(base_dir, train_path, val_path, test_path):
    """
    Updates constants in library modules to use demo settings.
    """
    print("Monkey-patching library configurations...")

    import library.config
    import library.data_processor
    import library.dataset
    import library.model
    import library.trainer
    import library.inference

    # New settings
    DEMO_OUTPUT_DIR = os.path.join(base_dir, "output")
    DEMO_SUBMISSION_PATH = os.path.join(base_dir, "submission.csv")
    DEMO_VOCAB_WIDE = 1000
    DEMO_VOCAB_DEEP = 1000
    DEMO_NUM_TAGS = 100
    DEMO_BATCH_SIZE = 32
    DEMO_EPOCHS = 1

    os.makedirs(DEMO_OUTPUT_DIR, exist_ok=True)

    # Patch library.config (though modules have already imported from it)
    library.config.OUTPUT_DIR = DEMO_OUTPUT_DIR
    library.config.TRAIN_PATH = train_path
    library.config.VAL_PATH = val_path
    library.config.TEST_PATH = test_path
    library.config.SUBMISSION_PATH = DEMO_SUBMISSION_PATH
    library.config.VOCAB_SIZE_WIDE = DEMO_VOCAB_WIDE
    library.config.VOCAB_SIZE_DEEP = DEMO_VOCAB_DEEP
    library.config.NUM_TAGS = DEMO_NUM_TAGS
    library.config.BATCH_SIZE = DEMO_BATCH_SIZE
    library.config.EPOCHS = DEMO_EPOCHS

    # Patch library.data_processor
    library.data_processor.OUTPUT_DIR = DEMO_OUTPUT_DIR
    library.data_processor.TRAIN_PATH = train_path
    library.data_processor.VAL_PATH = val_path
    library.data_processor.TEST_PATH = test_path
    library.data_processor.VOCAB_SIZE_WIDE = DEMO_VOCAB_WIDE
    library.data_processor.VOCAB_SIZE_DEEP = DEMO_VOCAB_DEEP
    library.data_processor.NUM_TAGS = DEMO_NUM_TAGS

    # Patch library.dataset
    library.dataset.NUM_TAGS = DEMO_NUM_TAGS
    library.dataset.BATCH_SIZE = DEMO_BATCH_SIZE

    # Patch library.model
    library.model.VOCAB_SIZE_WIDE = DEMO_VOCAB_WIDE
    library.model.VOCAB_SIZE_DEEP = DEMO_VOCAB_DEEP
    library.model.NUM_TAGS = DEMO_NUM_TAGS

    # Patch library.trainer
    library.trainer.OUTPUT_DIR = DEMO_OUTPUT_DIR
    library.trainer.SUBMISSION_PATH = DEMO_SUBMISSION_PATH
    library.trainer.EPOCHS = DEMO_EPOCHS
    library.trainer.BATCH_SIZE = DEMO_BATCH_SIZE
    library.trainer.NUM_TAGS = DEMO_NUM_TAGS

    # Patch library.inference
    library.inference.SUBMISSION_PATH = DEMO_SUBMISSION_PATH

    return DEMO_OUTPUT_DIR, DEMO_SUBMISSION_PATH


def run_demo():
    # 1. Setup Data
    base_dir, train_path, val_path, test_path = setup_demo_environment()

    # 2. Patch Libraries
    output_dir, submission_path = monkey_patch_library(
        base_dir, train_path, val_path, test_path
    )

    # Import functions after patching (or re-import if needed, but here simple import works as we patched attributes)
    from library.dataset import get_dataloaders
    from library.model import WideAndDeepModel, FocalLoss
    from library.trainer import train_one_epoch, evaluate, set_seed
    from library.inference import find_best_threshold, generate_submission
    from library.config import DEVICE

    set_seed(42)

    # 3. Data Processing
    print("\n=== Data Processing ===")
    # Force reprocessing by setting load_cached_data=False
    train_loader, val_loader, test_loader, test_ids, preprocessor = get_dataloaders(
        load_cached_data=False
    )

    # Validation: Check DataLoader output
    batch = next(iter(train_loader))
    print("Checking batch keys:", batch.keys())
    assert "deep_seq" in batch
    assert "wide_indices" in batch
    assert "target" in batch

    # Check shapes
    # deep_seq: (batch_size, max_len)
    assert batch["deep_seq"].shape[0] == 32  # Batch size
    # target: (batch_size, num_tags)
    assert batch["target"].shape[1] == 100  # Num tags

    print("Data processing successful. Batch shapes verified.")

    # 4. Model Initialization
    print("\n=== Model Initialization ===")
    model = WideAndDeepModel().to(DEVICE)
    print(model)

    # Validation: Forward pass with dummy batch
    with torch.no_grad():
        probs = model(
            batch["deep_seq"].to(DEVICE),
            batch["wide_indices"].to(DEVICE),
            batch["wide_values"].to(DEVICE),
            batch["wide_offsets"].to(DEVICE),
        )

    assert probs.shape == (32, 100)
    assert probs.min() >= 0 and probs.max() <= 1
    print("Model forward pass successful.")

    # 5. Training Loop (1 Epoch)
    print("\n=== Training Demo ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = FocalLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, scaler, DEVICE
    )
    print(f"Train Loss: {train_loss:.4f}")
    assert train_loss > 0, "Training loss should be positive"

    # 6. Evaluation
    print("\n=== Evaluation Demo ===")
    val_loss, val_probs, val_targets = evaluate(model, val_loader, criterion, DEVICE)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Probs Shape: {val_probs.shape}")

    assert val_probs.shape[0] == 2000, "Validation predictions count mismatch"
    assert val_probs.shape[1] == 100, "Validation predictions class count mismatch"

    # 7. Inference & Submission
    print("\n=== Inference Demo ===")

    # Find threshold
    best_thresh, best_score = find_best_threshold(val_probs, val_targets)
    print(f"Best Threshold: {best_thresh:.4f}, Best F1: {best_score:.4f}")

    # Generate submission
    submission_df = generate_submission(
        model, test_loader, test_ids, preprocessor, best_thresh, DEVICE
    )

    # Validation: Check submission file
    assert os.path.exists(submission_path), "Submission file was not created"

    loaded_sub = pd.read_csv(submission_path)
    print("Submission Head:")
    print(loaded_sub.head())

    assert "Id" in loaded_sub.columns
    assert "Tags" in loaded_sub.columns
    assert len(loaded_sub) == 2000, "Submission row count mismatch"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
