import os
import shutil
import pandas as pd
import torch
import numpy as np

# Import from the provided library
from library.config import PathConfig, TrainConfig, AudioConfig, ModelConfig
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model import SpecFPN_CRNN
from library.layers import CoordinateAttention, SpecFPN, AttentionPooling
from library.trainer import fit_model, generate_submission


def setup_demo_environment():
    """
    Sets up a restricted environment for the demo to run quickly.
    Creates subset metadata and overrides configuration paths.
    """
    print("Setting up demo environment...")

    # Define demo directories
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override PathConfig to use the demo directory
    PathConfig.WORKING_DIR = demo_dir
    PathConfig.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    PathConfig.setup_directories()

    # Create subset metadata to speed up data loading and training
    # Read original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Take small subsets (e.g., 50 samples each)
    subset_size = 50
    demo_train = orig_train.head(subset_size)
    demo_val = orig_val.head(subset_size)
    demo_test = orig_test.head(subset_size)

    # Save subset metadata to demo directory
    PathConfig.TRAIN_CSV = os.path.join(demo_dir, "train_small.csv")
    PathConfig.VAL_CSV = os.path.join(demo_dir, "val_small.csv")
    PathConfig.TEST_CSV = os.path.join(demo_dir, "test_small.csv")
    PathConfig.SUBMISSION_FILE = os.path.join(
        PathConfig.SUBMISSION_DIR, "submission.csv"
    )

    demo_train.to_csv(PathConfig.TRAIN_CSV, index=False)
    demo_val.to_csv(PathConfig.VAL_CSV, index=False)
    demo_test.to_csv(PathConfig.TEST_CSV, index=False)

    print(f"Created subset metadata in {demo_dir}")

    # Override TrainConfig for speed
    TrainConfig.EPOCHS = 1
    TrainConfig.SEEDS = [42]  # Only run one seed
    TrainConfig.BATCH_SIZE = 8  # Smaller batch for the small subset

    # Ensure AudioConfig uses defaults (already optimized, but good to confirm)
    # We rely on the existing config values.


def verify_layers():
    """
    Verifies the logic and shapes of custom layers.
    """
    print("\n--- Verifying Custom Layers ---")
    device = torch.device("cpu")

    # 1. Coordinate Attention
    # Input: (B, C, H, W) -> Output: (B, C, H, W)
    in_channels = 32
    ca = CoordinateAttention(in_channels=in_channels, reduction=8).to(device)
    dummy_input = torch.randn(2, in_channels, 64, 64).to(device)
    out = ca(dummy_input)

    assert (
        out.shape == dummy_input.shape
    ), f"CoordinateAttention output shape mismatch: {out.shape}"
    print("CoordinateAttention: OK")

    # 2. SpecFPN
    # Inputs: List of [Layer2, Layer3, Layer4] features
    # Shapes (assuming ResNet-like strides):
    # L2: (B, 128, F/2, T)
    # L3: (B, 256, F/4, T) (Time preserved, Freq downsampled)
    # L4: (B, 512, F/8, T)
    fpn = SpecFPN(in_channels_list=[128, 256, 512], out_channels=64).to(device)

    # Simulate features with frequency downsampling but constant time dimension (as per TimePreservingResNet)
    T = 100
    F_base = 32
    c2 = torch.randn(2, 128, F_base, T).to(device)
    c3 = torch.randn(2, 256, F_base // 2, T).to(device)
    c4 = torch.randn(2, 512, F_base // 4, T).to(device)

    out_fpn = fpn([c2, c3, c4])

    # Expect output to match L2 spatial resolution: (B, OutCh, F_base, T)
    expected_shape = (2, 64, F_base, T)
    assert (
        out_fpn.shape == expected_shape
    ), f"SpecFPN output shape mismatch: {out_fpn.shape} vs {expected_shape}"
    print("SpecFPN: OK")

    # 3. Attention Pooling
    # Input: (B, T, C) -> Output: (B, C)
    feat_dim = 128
    att_pool = AttentionPooling(input_dim=feat_dim).to(device)
    dummy_seq = torch.randn(2, T, feat_dim).to(device)
    out_pool = att_pool(dummy_seq)

    assert out_pool.shape == (
        2,
        feat_dim,
    ), f"AttentionPooling output shape mismatch: {out_pool.shape}"
    print("AttentionPooling: OK")


def verify_model_forward():
    """
    Verifies the full model forward pass.
    """
    print("\n--- Verifying Model Forward Pass ---")
    model = SpecFPN_CRNN()
    model.eval()

    # Input shape: (B, 1, Freq, Time)
    # AudioConfig: N_MELS=128. Time depends on duration/hop.
    # Duration 2.0s, SR 2000, Hop 40 -> ~50-51 frames.
    # Let's use a dummy tensor matching typical dimensions.
    dummy_input = torch.randn(4, 1, 128, 101)

    output = model(dummy_input)

    # Output should be (B, 1) and values in [0, 1] (Sigmoid)
    assert output.shape == (4, 1), f"Model output shape mismatch: {output.shape}"
    assert (
        output.min() >= 0 and output.max() <= 1
    ), "Model output values out of range [0, 1]"

    print("Model Forward Pass: OK")


def run_pipeline_demo():
    """
    Runs the full data loading, training, and submission pipeline on the subset.
    """
    print("\n--- Running Pipeline Demo ---")

    # 1. Get DataLoaders
    # load_cached_data=False forces processing the new subset files
    print("Initializing DataLoaders (processing audio subset)...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify DataLoader yields correct shapes
    sample_batch, sample_targets = next(iter(train_loader))
    print(f"Batch Shape: {sample_batch.shape}, Target Shape: {sample_targets.shape}")
    # Expected: (Batch, 1, 128, T)
    assert sample_batch.dim() == 4 and sample_batch.shape[1] == 1

    # 2. Train Model
    print("Initializing Model...")
    model = SpecFPN_CRNN()

    print("Starting Training (1 Epoch)...")
    seed = TrainConfig.SEEDS[0]
    trained_model = fit_model(model, train_loader, val_loader, seed=seed)

    # Verify model file was saved
    model_path = os.path.join(PathConfig.WORKING_DIR, f"model_seed_{seed}.pth")
    assert os.path.exists(model_path), f"Model checkpoint not found at {model_path}"
    print("Training Complete. Checkpoint verified.")

    # 3. Generate Submission
    print("Generating Submission...")
    generate_submission([trained_model], test_loader)

    # Verify submission file
    sub_path = PathConfig.SUBMISSION_FILE
    assert os.path.exists(sub_path), f"Submission file not found at {sub_path}"

    df_sub = pd.read_csv(sub_path)
    print(f"Submission generated with {len(df_sub)} rows.")
    assert len(df_sub) == 50, "Submission should have 50 rows (matching the subset)"
    assert (
        "clip" in df_sub.columns and "probability" in df_sub.columns
    ), "Submission columns incorrect"

    print("Pipeline Demo: OK")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # Setup
    setup_demo_environment()

    # Verification
    verify_layers()
    verify_model_forward()

    # Execution
    run_pipeline_demo()

    print("\nAll demonstrations completed successfully.")
