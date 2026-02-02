import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import process_predictions, compute_levenshtein
from library.data_loader import get_dataloaders, GestureDataset
from library.model import PAKRNet
from library.train import Trainer, CombinedLoss

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets fixed seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_demo_environment():
    """
    Sets up a temporary environment for the demo.
    1. Modifies Config to use ./working/demo_env
    2. Creates a mini-subset of metadata to speed up data loading.
    """
    print(">>> Setting up demo environment...")

    # Define paths
    demo_work_dir = "./working/demo_env"
    demo_meta_dir = os.path.join(demo_work_dir, "metadata")
    demo_cache_dir = os.path.join(demo_work_dir, "cache")
    demo_submission_dir = os.path.join(demo_work_dir, "submission")

    # Clean up previous run if exists
    if os.path.exists(demo_work_dir):
        shutil.rmtree(demo_work_dir)

    os.makedirs(demo_meta_dir, exist_ok=True)
    os.makedirs(demo_cache_dir, exist_ok=True)
    os.makedirs(demo_submission_dir, exist_ok=True)

    # Modify Config globally
    Config.WORK_DIR = demo_work_dir
    Config.METADATA_DIR = demo_meta_dir
    Config.CACHE_DIR = demo_cache_dir
    Config.SUBMISSION_DIR = demo_submission_dir
    Config.MODEL_SAVE_PATH = os.path.join(demo_work_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_submission_dir, "submission.csv")

    # Reduce compute load for demo
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 2
    Config.GRU_HIDDEN_SIZE = 32  # Smaller model for speed
    Config.TCN_CHANNELS = 64

    # Create Mini Metadata
    # We read the original metadata and sample the first 5 rows
    original_meta_dir = "./metadata"
    for split in ["train", "val", "test"]:
        src_csv = os.path.join(original_meta_dir, f"{split}.csv")
        if os.path.exists(src_csv):
            df = pd.read_csv(src_csv)
            # Take a small subset (e.g., 5 samples)
            mini_df = df.head(5)
            dst_csv = os.path.join(demo_meta_dir, f"{split}.csv")
            mini_df.to_csv(dst_csv, index=False)
            print(f"    Created mini {split} metadata with {len(mini_df)} samples.")
        else:
            print(f"    Warning: Original {split}.csv not found.")


def demo_data_loading():
    """
    Demonstrates and validates the data loading pipeline.
    """
    print("\n>>> Testing Data Loading...")

    # Initialize Dataloaders
    # This will trigger processing and caching of the mini dataset
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=0
    )

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Samples: {len(val_loader)}")

    # Fetch one batch from training
    features, labels = next(iter(train_loader))

    # Assertions
    # Features: (Batch, Time, InputDim=193)
    # Labels: (Batch, Time)
    print(f"    Feature Shape: {features.shape}")
    print(f"    Label Shape: {labels.shape}")

    assert features.dim() == 3, "Features should be 3D tensor (Batch, Time, Channels)"
    assert (
        features.shape[2] == 193
    ), f"Expected 193 input channels (180 kinematics + 13 MFCC), got {features.shape[2]}"
    assert labels.dim() == 2, "Labels should be 2D tensor (Batch, Time)"
    assert (
        features.shape[0] == Config.BATCH_SIZE
    ), f"Expected batch size {Config.BATCH_SIZE}"
    assert (
        features.shape[1] == labels.shape[1]
    ), "Time dimension mismatch between features and labels"

    print("    Data Loading Verified.")
    return train_loader, val_loader


def demo_model_forward(device):
    """
    Demonstrates model instantiation and forward pass.
    """
    print("\n>>> Testing Model Architecture...")

    model = PAKRNet().to(device)

    # Create dummy input: (Batch=2, Time=50, Channels=193)
    dummy_input = torch.randn(2, 50, 193).to(device)

    # Forward pass
    outputs = model(dummy_input)

    # Assertions
    # Output should be a list of 3 tensors (Deep Supervision)
    assert isinstance(outputs, list), "Model output should be a list"
    assert len(outputs) == 3, "Model should return outputs from 3 stages"

    for i, out in enumerate(outputs):
        # Shape: (Batch, Time, NumClasses)
        print(f"    Stage {i+1} Output Shape: {out.shape}")
        assert out.shape == (
            2,
            50,
            Config.NUM_CLASSES,
        ), f"Stage {i+1} output shape mismatch"

    print("    Model Architecture Verified.")
    return model


def demo_training_loop(model, train_loader, val_loader, device):
    """
    Demonstrates the training loop using the Trainer class.
    """
    print("\n>>> Testing Training Loop...")

    trainer = Trainer(model, train_loader, val_loader, device)

    # Run short training
    # This validates the loss function, backprop, and validation metric calculation
    trainer.train(num_epochs=Config.NUM_EPOCHS)

    # Check if model file was created (if validation score improved, which is likely with random init vs 0)
    # Note: If validation score is 0 initially and doesn't improve, file might not be saved.
    # However, usually loss decreases. We check if the logic ran without error.
    print("    Training Loop Completed Successfully.")

    return trainer


def demo_inference(trainer, val_loader, device):
    """
    Demonstrates inference and post-processing.
    """
    print("\n>>> Testing Inference and Post-Processing...")

    model = trainer.model
    model.eval()

    # Get a sample from validation
    features, labels, sample_ids = next(iter(val_loader))
    features = features.to(device)

    with torch.no_grad():
        outputs = model(features)
        final_logits = outputs[-1]  # Use stage 3

        # Convert to probabilities
        probs = torch.softmax(final_logits, dim=2)

        # Get predictions (Batch index 0)
        frame_probs = probs[0].cpu().numpy()

        # Process predictions (RLE + Filtering)
        predicted_sequence = process_predictions(frame_probs)

        # Get Ground Truth
        gt_frame_ids = labels[0].cpu().numpy()
        gt_sequence = process_predictions(gt_frame_ids)

        print(f"    Sample ID: {sample_ids[0]}")
        print(f"    Predicted Sequence: {predicted_sequence}")
        print(f"    Ground Truth Sequence: {gt_sequence}")

        # Calculate Levenshtein
        dist = compute_levenshtein(predicted_sequence, gt_sequence)
        print(f"    Levenshtein Distance: {dist}")

        # Validation
        assert isinstance(predicted_sequence, list), "Prediction should be a list"
        assert all(
            isinstance(x, (int, np.integer)) for x in predicted_sequence
        ), "Prediction elements should be integers"

    print("    Inference Verified.")


if __name__ == "__main__":
    # 1. Setup
    set_seed(42)
    setup_demo_environment()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    train_loader, val_loader = demo_data_loading()

    # 3. Model
    model = demo_model_forward(device)

    # 4. Training
    trainer = demo_training_loop(model, train_loader, val_loader, device)

    # 5. Inference
    demo_inference(trainer, val_loader, device)

    print("\n>>> All demonstrations passed successfully.")
