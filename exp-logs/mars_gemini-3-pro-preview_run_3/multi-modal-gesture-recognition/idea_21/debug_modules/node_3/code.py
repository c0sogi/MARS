import os
import shutil
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import the provided library modules
from library import config, utils, model, loss, dataset, engine


def run_demo():
    print("=== Starting Library Demonstration ===\n")

    # ==========================================
    # 1. Configuration Override for Demo
    # ==========================================
    print("1. Configuring environment for demo...")

    # Override config parameters to ensure fast execution
    config.NUM_EPOCHS = 1
    config.BATCH_SIZE = 2
    config.DEBUG = True
    config.DEBUG_SUBSET_SIZE = 5  # Only use 5 samples
    config.WORKING_DIR = "./working/demo_execution"
    config.CACHE_DIR = os.path.join(config.WORKING_DIR, "cache")
    config.SUBMISSION_DIR = config.WORKING_DIR
    config.BEST_MODEL_PATH = os.path.join(config.WORKING_DIR, "best_model.pth")
    config.SUBMISSION_FILE = os.path.join(config.WORKING_DIR, "submission.csv")

    # Clean and recreate working directory
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Set seed
    config.set_seed(42)
    device = config.get_device()
    print(f"   Working Directory: {config.WORKING_DIR}")
    print(f"   Device: {device}")
    print("   Configuration updated.\n")

    # ==========================================
    # 2. Demonstrate Utilities (library.utils)
    # ==========================================
    print("2. Testing Utilities...")

    # A. Test RLE Encoding
    # Logic: Collapse consecutive duplicates, remove background (0)
    raw_preds = [0, 0, 1, 1, 1, 0, 2, 2, 0, 3, 0]
    expected_rle = [1, 2, 3]
    rle_result = utils.rle_encode(raw_preds)
    print(f"   RLE Input: {raw_preds}")
    print(f"   RLE Output: {rle_result}")
    assert (
        rle_result == expected_rle
    ), f"RLE failed. Expected {expected_rle}, got {rle_result}"
    print("   [Pass] RLE Encoding logic verified.")

    # B. Test Skeleton Parsing & Kinematics
    # We need a real file path. Let's grab one from metadata.
    train_meta = pd.read_csv(config.TRAIN_METADATA_PATH)
    sample_row = train_meta.iloc[0]
    mat_path = os.path.join(config.INPUT_DIR, sample_row["data_path"])

    if os.path.exists(mat_path):
        # Test Parsing
        skeleton = utils.safe_parse_skeleton(mat_path)
        if skeleton is not None:
            # Expected shape: (Frames, 20, 3)
            assert len(skeleton.shape) == 3, "Skeleton should be 3D array"
            assert (
                skeleton.shape[1] == config.NUM_JOINTS
            ), f"Expected {config.NUM_JOINTS} joints"
            assert skeleton.shape[2] == 3, "Expected 3 spatial coordinates"
            print(f"   [Pass] Skeleton parsed. Shape: {skeleton.shape}")

            # Test Kinematics
            # Expected shape: (Frames, InputDim) where InputDim includes Pos+Vel+Acc
            # Note: compute_kinematics flattens joints
            kinematics = utils.compute_kinematics(skeleton)
            # Calculate expected dim: 20 joints * 3 coords * 3 (pos+vel+acc) = 180
            # (Audio MFCC is added later in dataset, not here)
            expected_k_dim = config.NUM_JOINTS * 3 * 3
            assert (
                kinematics.shape[1] == expected_k_dim
            ), f"Kinematics dim mismatch. Got {kinematics.shape[1]}"
            print(f"   [Pass] Kinematics computed. Shape: {kinematics.shape}")

            # Test Augmentation
            aug_skel = utils.augment_skeleton(skeleton)
            assert aug_skel.shape == skeleton.shape, "Augmentation changed shape"
            assert not np.array_equal(
                aug_skel, skeleton
            ), "Augmentation did not modify data"
            print("   [Pass] Augmentation verified.")
    else:
        print(f"   [Skip] Could not find {mat_path} for testing utils.")

    print("   Utilities demonstration complete.\n")

    # ==========================================
    # 3. Demonstrate Dataset (library.dataset)
    # ==========================================
    print("3. Testing Dataset Loading...")

    # Initialize dataset in debug mode (loads small subset)
    train_dataset = dataset.GestureDataset(
        split="train", load_cached_data=False, augment=True, debug=True
    )

    assert len(train_dataset) > 0, "Dataset is empty"
    print(f"   Loaded {len(train_dataset)} windows from debug subset.")

    # Fetch one sample
    features, labels, mask = train_dataset[0]

    # Verify shapes
    # Features: (WindowSize, InputDim) -> InputDim = Kinematics(180) + MFCC(13) = 193
    assert (
        features.shape[0] == config.WINDOW_SIZE
    ), f"Feature time dim mismatch: {features.shape[0]}"
    assert (
        features.shape[1] == config.INPUT_DIM
    ), f"Feature channel dim mismatch: {features.shape[1]}"
    assert labels.shape[0] == config.WINDOW_SIZE, "Label time dim mismatch"
    assert mask.shape[0] == config.WINDOW_SIZE, "Mask time dim mismatch"

    print(
        f"   Sample shapes verified: Features {tuple(features.shape)}, Labels {tuple(labels.shape)}"
    )
    print("   Dataset demonstration complete.\n")

    # ==========================================
    # 4. Demonstrate Model Architecture (library.model)
    # ==========================================
    print("4. Testing Model Architecture...")

    net = model.RHKRN().to(device)

    # Create dummy input batch: (Batch, Time, InputDim)
    # Note: Model expects (Batch, Time, InputDim) for GRU, but internally handles perms
    dummy_input = torch.randn(2, config.WINDOW_SIZE, config.INPUT_DIM).to(device)

    # Forward pass
    outputs = net(dummy_input)

    # Expect list of 3 outputs (Stage 1, 2, 3)
    assert isinstance(outputs, list), "Model output should be a list"
    assert len(outputs) == 3, f"Expected 3 stages, got {len(outputs)}"

    # Check shape of last stage: (Batch, NumClasses, Time)
    final_logits = outputs[-1]
    assert final_logits.shape == (
        2,
        config.NUM_CLASSES,
        config.WINDOW_SIZE,
    ), f"Output shape mismatch. Got {final_logits.shape}"

    print("   [Pass] Model forward pass successful.")
    print(f"   Output shapes: {[o.shape for o in outputs]}")
    print("   Model demonstration complete.\n")

    # ==========================================
    # 5. Demonstrate Loss Function (library.loss)
    # ==========================================
    print("5. Testing Loss Function...")

    criterion = loss.CascadedLoss().to(device)

    # Create dummy targets: (Batch, Time)
    dummy_targets = torch.randint(0, config.NUM_CLASSES, (2, config.WINDOW_SIZE)).to(
        device
    )

    # Compute loss
    loss_val, metrics = criterion(outputs, dummy_targets)

    assert isinstance(loss_val, torch.Tensor), "Loss should be a tensor"
    assert loss_val.item() > 0, "Loss should be positive"
    assert "total_loss" in metrics, "Metrics dict missing total_loss"

    print(f"   [Pass] Loss calculated: {loss_val.item():.4f}")
    print(f"   Metrics: {metrics}")
    print("   Loss demonstration complete.\n")

    # ==========================================
    # 6. Demonstrate Engine / Training Loop (library.engine)
    # ==========================================
    print("6. Testing Training & Inference Engine...")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Initialize Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    # A. Train One Epoch
    print("   Running training step...")
    avg_loss, avg_metrics = engine.train_one_epoch(
        net, train_loader, criterion, optimizer, device
    )
    print(f"   [Pass] Training epoch finished. Avg Loss: {avg_loss:.4f}")

    # B. Evaluation
    print("   Running evaluation step...")
    # Reuse train dataset as val for demo purposes
    val_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    val_loss, val_metrics = engine.evaluate(net, val_loader, criterion, device)
    print(f"   [Pass] Evaluation finished. Val Acc: {val_metrics['accuracy']:.4f}")

    # C. Inference on Single Sequence
    print("   Running inference on a single sequence...")
    # Grab raw data from dataset
    raw_skel = train_dataset.skeletons[0]
    raw_audio = train_dataset.audios[0]

    preds = engine.infer_sequence(net, raw_skel, raw_audio, device)
    print(f"   [Pass] Inference successful. Predicted gestures: {preds}")

    # D. Generate Submission
    print("   Generating submission file...")
    # Use the same dataset as 'test' for demo
    engine.generate_submission(net, train_dataset, device, config.SUBMISSION_FILE)

    assert os.path.exists(config.SUBMISSION_FILE), "Submission file was not created"
    with open(config.SUBMISSION_FILE, "r") as f:
        lines = f.readlines()
        print(f"   [Pass] Submission file created with {len(lines)} lines.")
        if len(lines) > 0:
            print(f"   Sample line: {lines[0].strip()}")

    print("\n=== Library Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
