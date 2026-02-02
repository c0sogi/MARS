import os
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd

# Import from the provided library
from library.config import Config
from library.data_utils import save_submission, load_skeleton_data
from library.features import extract_features, FeatureNormalizer
from library.dataset import GestureDataset
from library.model import SA_AKN
from library.loss import CascadedLoss
from library.train_eval import (
    train_epoch,
    run_inference_on_sequence,
    decode_predictions,
    calculate_levenshtein,
)


def setup_demo_environment():
    """
    Sets up a temporary directory and overrides Config for a fast demo run.
    """
    # Define demo paths
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config for speed and isolation
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Enable Debug mode to load only a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 20  # Small number of samples for speed

    # Training hyperparameters for demo
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.GRU_HIDDEN_DIM = 64  # Smaller model for speed
    Config.TCN_CHANNELS = 32
    Config.TCN_LAYERS = 2

    # Ensure directories exist
    Config.setup()

    # Set seeds
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    print(f"Demo environment setup at {demo_dir}")
    print("Configured for DEBUG mode with reduced model size and samples.")


def test_feature_extraction():
    """
    Verifies the logic of feature extraction from raw skeleton data.
    """
    print("\n=== Testing Feature Extraction ===")

    # Create synthetic skeleton data: (Time=100, Joints=20, Coords=3)
    T, J, C = 100, 20, 3
    dummy_skeleton = np.random.rand(T, J, C).astype(np.float32)

    # Extract features
    # Expected output dim: 20 joints * 12 channels (Pos, Bone, Vel, Acc) = 240
    features = extract_features(dummy_skeleton, augment=False)

    print(f"Input Skeleton Shape: {dummy_skeleton.shape}")
    print(f"Extracted Features Shape: {features.shape}")

    assert features.shape == (
        T,
        240,
    ), f"Expected shape ({T}, 240), got {features.shape}"
    assert not np.isnan(features).any(), "Features contain NaNs"

    # Test Augmentation
    aug_features = extract_features(dummy_skeleton, augment=True)
    assert aug_features.shape == (T, 240), "Augmented features shape mismatch"
    assert not np.array_equal(
        features, aug_features
    ), "Augmentation did not change data"

    print("Feature extraction logic verified.")


def test_dataset_pipeline():
    """
    Verifies dataset loading, sliding window generation, and batching.
    """
    print("\n=== Testing Dataset Pipeline ===")

    # Initialize Dataset (Train split)
    # This will trigger cache generation for the debug subset
    dataset = GestureDataset("train", load_cached_data=False)

    print(
        f"Dataset initialized with {len(dataset)} windows from {len(dataset.sample_ids)} samples."
    )

    assert len(dataset) > 0, "Dataset is empty"

    # Check __getitem__
    features, labels, sample_idx, start_frame = dataset[0]

    print(f"Sample 0 Features Shape: {features.shape}")  # Should be (WindowSize, 240)
    print(f"Sample 0 Labels Shape: {labels.shape}")  # Should be (WindowSize,)

    expected_window = Config.WINDOW_SIZE
    expected_dim = Config.INPUT_DIM

    assert features.shape == (
        expected_window,
        expected_dim,
    ), f"Expected features ({expected_window}, {expected_dim}), got {features.shape}"
    assert labels.shape == (
        expected_window,
    ), f"Expected labels ({expected_window},), got {labels.shape}"
    assert isinstance(features, torch.Tensor), "Features must be a Tensor"
    assert isinstance(labels, torch.Tensor), "Labels must be a Tensor"

    # Check DataLoader
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    batch_feat, batch_lbl, _, _ = next(iter(loader))

    print(f"Batch Features Shape: {batch_feat.shape}")
    assert batch_feat.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"

    print("Dataset pipeline verified.")
    return dataset, loader


def test_model_and_loss(loader):
    """
    Verifies model instantiation, forward pass, and loss calculation.
    """
    print("\n=== Testing Model and Loss ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate Model
    model = SA_AKN().to(device)
    print("SA_AKN model instantiated.")

    # Instantiate Loss
    criterion = CascadedLoss().to(device)

    # Get a batch
    inputs, targets, _, _ = next(iter(loader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Forward Pass
    logits1, logits2, logits3 = model(inputs)

    print(f"Logits1 Shape: {logits1.shape}")
    print(f"Logits2 Shape: {logits2.shape}")
    print(f"Logits3 Shape: {logits3.shape}")

    expected_shape = (Config.BATCH_SIZE, Config.WINDOW_SIZE, Config.NUM_CLASSES)
    assert logits1.shape == expected_shape, "Logits1 shape mismatch"
    assert logits3.shape == expected_shape, "Logits3 shape mismatch"

    # Loss Calculation
    loss, metrics = criterion(logits1, logits2, logits3, targets)

    print(f"Calculated Loss: {loss.item():.4f}")
    print(f"Loss Metrics: {metrics}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    print("Model and Loss verified.")
    return model, criterion, device


def test_training_loop(model, loader, criterion, device):
    """
    Simulates a training epoch.
    """
    print("\n=== Testing Training Loop ===")

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Run one epoch
    avg_loss = train_epoch(model, loader, criterion, optimizer, device)
    print(f"Epoch complete. Average Loss: {avg_loss:.4f}")

    assert avg_loss > 0, "Average loss should be positive"

    # Save model
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved"
    print(f"Model saved to {Config.MODEL_SAVE_PATH}")


def test_inference_and_submission(model, device):
    """
    Verifies inference logic, decoding, metrics, and submission file generation.
    """
    print("\n=== Testing Inference and Submission ===")

    # Use Validation dataset (pre-computed features mode)
    val_dataset = GestureDataset("val", load_cached_data=False)

    if len(val_dataset.sample_ids) == 0:
        print(
            "Validation set empty in debug mode (might happen if split is small). Skipping inference test."
        )
        return

    # Pick the first sample
    sample_idx = 0
    sample_id = val_dataset.sample_ids[sample_idx]
    features_np = val_dataset.processed_features[sample_idx]
    features_tensor = torch.from_numpy(features_np).float().to(device)

    print(f"Running inference on sample {sample_id} with length {features_np.shape[0]}")

    # Run Inference
    stride = Config.WINDOW_STRIDE_TEST
    window_size = Config.WINDOW_SIZE

    avg_probs = run_inference_on_sequence(
        model, features_tensor, device, window_size, stride
    )

    assert avg_probs.shape == (
        features_np.shape[0],
        Config.NUM_CLASSES,
    ), "Inference output shape mismatch"

    # Decode
    pred_seq = decode_predictions(avg_probs)
    print(f"Predicted Sequence: {pred_seq}")

    # Get Ground Truth for metric check
    gt_meta = val_dataset.raw_labels_meta[sample_idx]
    gt_seq = [int(l["id"]) for l in gt_meta]
    print(f"Ground Truth Sequence: {gt_seq}")

    # Calculate Metric
    dist = calculate_levenshtein(pred_seq, gt_seq)
    print(f"Levenshtein Distance: {dist}")

    # Mock Submission Generation
    # We'll create a dummy submission for the samples in the validation set
    predictions = [pred_seq]  # Just one for demo
    sample_ids = [sample_id]

    save_submission(predictions, sample_ids, Config.SUBMISSION_PATH)
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    # Verify file content
    with open(Config.SUBMISSION_PATH, "r") as f:
        lines = f.readlines()
        print(f"Submission File Content (First line): {lines[0].strip()}")
        assert len(lines) == 1, "Expected 1 line in submission file"
        assert lines[0].startswith(sample_id), "Submission ID mismatch"

    print("Inference and Submission logic verified.")


if __name__ == "__main__":
    # 1. Setup
    setup_demo_environment()

    # 2. Feature Logic
    test_feature_extraction()

    # 3. Dataset Logic
    dataset, loader = test_dataset_pipeline()

    # 4. Model & Loss Logic
    model, criterion, device = test_model_and_loss(loader)

    # 5. Training Logic
    test_training_loop(model, loader, criterion, device)

    # 6. Inference Logic
    test_inference_and_submission(model, device)

    print("\nAll demonstrations completed successfully.")
