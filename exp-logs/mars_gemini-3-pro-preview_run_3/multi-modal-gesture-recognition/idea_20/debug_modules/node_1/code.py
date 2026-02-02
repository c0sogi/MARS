import os
import sys
import torch
import numpy as np
import shutil

# ==========================================
# 1. Configuration Overrides
# ==========================================
# Import config first to modify settings before other modules load them
import library.config as config

# Override settings for a fast demonstration
config.NUM_EPOCHS = 1
config.BATCH_SIZE = 4
config.DEBUG_SUBSET_SIZE = 10  # Only process 10 samples
config.CACHE_DIR = os.path.join(config.WORKING_DIR, "demo_cache")

# Ensure cache directory exists
os.makedirs(config.CACHE_DIR, exist_ok=True)

# ==========================================
# 2. Library Imports
# ==========================================
# Import modules after config modification
from library.utils import (
    setup_logger,
    decode_predictions,
    compute_levenshtein,
    generate_submission_file,
)
from library.dataset import GestureDataset
from library.modules import LGKRN
from library.loss import CascadedSmoothLoss
from library.engine import train_one_epoch, evaluate
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    DEVICE,
    INPUT_DIM,
    NUM_CLASSES,
    WINDOW_SIZE,
    SEED,
)


def main():
    print("=== Gesture Recognition Library Demo ===")

    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Setup Logger
    log_path = os.path.join(config.WORKING_DIR, "demo.log")
    logger = setup_logger(log_path)
    logger.info("Logger initialized.")

    # ==========================================
    # 3. Data Loading & Verification
    # ==========================================
    print("\n[Step 1] Initializing Datasets...")

    # Initialize Train Dataset (with augmentation, force no cache for demo)
    train_dataset = GestureDataset(
        metadata_path=TRAIN_METADATA_PATH,
        split_name="train_demo",
        load_cache=False,
        augment=True,
        debug_size=config.DEBUG_SUBSET_SIZE,
    )

    # Initialize Val Dataset (no augmentation)
    val_dataset = GestureDataset(
        metadata_path=VAL_METADATA_PATH,
        split_name="val_demo",
        load_cache=False,
        augment=False,
        debug_size=config.DEBUG_SUBSET_SIZE,
    )

    print(f"Train Dataset size (windows): {len(train_dataset)}")
    print(f"Val Dataset size (windows): {len(val_dataset)}")

    # Verify Data Shapes if data is available
    if len(train_dataset) > 0:
        x, y, idx = train_dataset[0]
        # x: (WindowSize, InputDim), y: (WindowSize,)
        assert x.shape == (
            WINDOW_SIZE,
            INPUT_DIM,
        ), f"Input shape mismatch. Expected {(WINDOW_SIZE, INPUT_DIM)}, got {x.shape}"
        assert y.shape == (
            WINDOW_SIZE,
        ), f"Target shape mismatch. Expected {(WINDOW_SIZE,)}, got {y.shape}"
        print("Dataset shapes verified.")

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )

    # ==========================================
    # 4. Model Initialization & Forward Pass
    # ==========================================
    print("\n[Step 2] Initializing Model...")
    model = LGKRN().to(DEVICE)
    print(f"Model moved to {DEVICE}.")

    # Dummy Forward Pass
    dummy_input = torch.randn(2, WINDOW_SIZE, INPUT_DIM).to(DEVICE)
    logits_1, logits_2, logits_3 = model(dummy_input)

    expected_shape = (2, WINDOW_SIZE, NUM_CLASSES)
    assert logits_1.shape == expected_shape, f"Stage 1 shape error: {logits_1.shape}"
    assert logits_2.shape == expected_shape, f"Stage 2 shape error: {logits_2.shape}"
    assert logits_3.shape == expected_shape, f"Stage 3 shape error: {logits_3.shape}"
    print("Model forward pass verified.")

    # ==========================================
    # 5. Loss Function Verification
    # ==========================================
    print("\n[Step 3] Initializing Loss Function...")
    criterion = CascadedSmoothLoss()

    # Dummy Loss Calculation
    dummy_targets = torch.randint(0, NUM_CLASSES, (2, WINDOW_SIZE)).to(DEVICE)
    loss = criterion([logits_1, logits_2, logits_3], dummy_targets)

    assert torch.isfinite(loss), "Loss is non-finite (NaN or Inf)."
    print(f"Loss computation successful. Value: {loss.item():.4f}")

    # ==========================================
    # 6. Training Loop (1 Epoch)
    # ==========================================
    print("\n[Step 4] Running Training Loop...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if len(train_loader) > 0:
        avg_train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        print(f"Training Epoch Completed. Avg Loss: {avg_train_loss:.4f}")
    else:
        print("Skipping training (dataset empty).")

    # ==========================================
    # 7. Evaluation Loop
    # ==========================================
    print("\n[Step 5] Running Evaluation...")
    if len(val_loader) > 0:
        val_loss, val_acc, val_lev = evaluate(
            model, val_loader, val_dataset, criterion, DEVICE
        )
        print(f"Validation Results:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Frame Accuracy: {val_acc:.4f}")
        print(f"  Levenshtein Error: {val_lev:.4f}")
    else:
        print("Skipping evaluation (dataset empty).")

    # ==========================================
    # 8. Utilities Verification
    # ==========================================
    print("\n[Step 6] Verifying Utilities...")

    # Test decode_predictions
    # Sequence: 0(BG), 1, 1, 0, 2, 2, 2, 0 -> [1, 2]
    raw_preds = np.array([0, 1, 1, 0, 2, 2, 2, 0])
    decoded = decode_predictions(raw_preds)
    assert decoded == [1, 2], f"Decoder error. Expected [1, 2], got {decoded}"
    print("decode_predictions verified.")

    # Test compute_levenshtein
    # Pred: [1, 2], Target: [1, 3] -> Edit Dist = 1 (Sub 2->3)
    # Norm = 1 / len(Target) = 1/2 = 0.5
    score = compute_levenshtein([[1, 2]], [[1, 3]])
    assert abs(score - 0.5) < 1e-6, f"Levenshtein error. Expected 0.5, got {score}"
    print("compute_levenshtein verified.")

    # Test generate_submission_file
    sample_ids = ["SampleTest01", "SampleTest02"]
    predictions = [[1, 2], [3]]
    out_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")

    generate_submission_file(predictions, sample_ids, out_path)
    assert os.path.exists(out_path), "Submission file not created."

    with open(out_path, "r") as f:
        lines = [l.strip() for l in f.readlines()]
        assert len(lines) == 2
        assert lines[0] == "SampleTest01,1,2"
        assert lines[1] == "SampleTest02,3"
    print("generate_submission_file verified.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
