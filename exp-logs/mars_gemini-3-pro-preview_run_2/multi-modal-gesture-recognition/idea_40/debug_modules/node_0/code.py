import os
import torch
import numpy as np
import shutil
import time

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data_loader import get_dataloaders
from library.model import GMG_CRGN
from library.losses import DeepSupervisionLoss
from library.train import train_one_epoch, validate
from library.inference import predict_all, post_process_sequence


def run_demo():
    print("=== Starting GMG-CRGN Pipeline Demo ===")

    # 1. Patch Configuration for Speed
    # We modify the Config class attributes directly to run a lightweight version
    print("Configuring for fast execution...")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG_SUBSET_SIZE = 10  # Use only 10 samples
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    # Reduce model complexity for demo speed
    Config.MSTCN_STAGES = 1  # Only 1 refinement stage
    Config.MSTCN_LAYERS = 2  # Only 2 layers per stage
    Config.HIDDEN_DIM = 32  # Smaller hidden dim
    Config.MSTCN_CHANNELS = 32  # Smaller channel dim

    # Ensure reproducibility
    set_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("\n--- Testing Data Loading ---")
    # Force cache regeneration to test parsing logic by ensuring cache dir is clean or just relying on overwrite
    # Since we can't delete pre-existing cache easily without knowing if it affects others, we rely on the unique WORK_DIR in Config
    # Config.WORK_DIR is ./working/idea_40, which is likely clean.

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=True,
        load_cached_data=True,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    features, targets, lengths, mask, ids = batch

    # Verify Shapes
    # Features: (B, T, InputDim) -> InputDim should be 118
    assert features.dim() == 3, f"Expected 3D features, got {features.shape}"
    assert (
        features.size(2) == Config.INPUT_DIM
    ), f"Expected input dim {Config.INPUT_DIM}, got {features.size(2)}"
    # Targets: (B, T)
    assert targets.dim() == 2, f"Expected 2D targets, got {targets.shape}"
    # Mask: (B, T)
    assert mask.shape == targets.shape, "Mask shape mismatch"

    print("Data Batch Verification Successful:")
    print(f"  Features: {features.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  Lengths: {lengths}")

    # 3. Model Initialization
    print("\n--- Testing Model Initialization ---")
    model = GMG_CRGN().to(device)

    # Move batch to device
    features = features.to(device)
    mask = mask.to(device)
    targets = targets.to(device)
    lengths = lengths.to(device)

    # 4. Forward Pass
    print("\n--- Testing Forward Pass ---")
    outputs = model(features, mask)

    # Output should be a list of tensors (one per stage)
    assert isinstance(outputs, list), "Model output should be a list"
    assert (
        len(outputs) == Config.MSTCN_STAGES + 1
    ), f"Expected {Config.MSTCN_STAGES + 1} outputs (1 initial + {Config.MSTCN_STAGES} stages)"

    final_output = outputs[-1]
    # Shape: (B, T, NumClasses + 1)
    expected_channels = Config.NUM_CLASSES + 1
    assert final_output.shape == (
        features.size(0),
        features.size(1),
        expected_channels,
    ), f"Output shape mismatch. Expected (B, T, {expected_channels}), got {final_output.shape}"

    print(f"Forward pass successful. Output shape: {final_output.shape}")

    # 5. Loss Computation
    print("\n--- Testing Loss Computation ---")
    criterion = DeepSupervisionLoss().to(device)

    loss, metrics = criterion(outputs, targets, lengths)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    assert "loss_cls_1" in metrics, "Metrics dictionary missing classification loss"

    print(f"Loss calculation successful. Total Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")

    # 6. Training Loop (One Epoch)
    print("\n--- Testing Training Loop ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)

    avg_loss, avg_metrics = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    print(f"Epoch complete. Avg Loss: {avg_loss:.4f}")

    # 7. Validation
    print("\n--- Testing Validation ---")
    val_loss, val_metrics, val_acc = validate(model, val_loader, criterion, device)
    print(f"Validation complete. Acc: {val_acc:.4f}")

    # 8. Inference & Post-processing
    print("\n--- Testing Inference & Post-processing ---")
    # We'll use the validation loader for inference demo
    results = predict_all(model, val_loader, device)

    assert len(results) > 0, "Inference returned no results"

    sample_id, raw_preds = results[0]
    print(f"Sample ID: {sample_id}")
    print(f"Raw Preds Shape: {raw_preds.shape}")

    # Test Post-processing
    # Create a dummy prediction sequence to ensure post-processing logic works deterministically
    # Sequence: 0 0 1 1 1 1 1 0 0 2 2 2 2 2 0 0 (Background -> Gesture 1 -> Background -> Gesture 2 -> BG)
    dummy_preds = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 0, 0])
    processed_gestures = post_process_sequence(dummy_preds, kernel_size=3)

    print(f"Dummy Raw: {dummy_preds}")
    print(f"Processed: {processed_gestures}")

    assert processed_gestures == [
        1,
        2,
    ], f"Post-processing failed. Expected [1, 2], got {processed_gestures}"

    # Process actual result
    actual_gestures = post_process_sequence(raw_preds, kernel_size=7)
    print(f"Actual Sample Processed Gestures: {actual_gestures}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
