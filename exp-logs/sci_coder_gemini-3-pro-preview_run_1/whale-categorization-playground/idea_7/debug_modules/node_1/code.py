import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, WhaleLabelEncoder
from library.dataset import WhaleDataset, get_transforms
from library.model import WhaleDenseNet
from library.train import train_model, get_logits_inference


def run_demo():
    print("==== Starting Whale Species Prediction Demo ====")

    # -------------------------------------------------------------------------
    # 1. Patch Configuration for Speed and Demo Purposes
    # -------------------------------------------------------------------------
    print("\n[1] Patching Configuration...")

    # Set a specific directory for demo outputs
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Modify Config attributes globally
    Config.WORKING_DIR = demo_dir
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use very small subset
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Reduce training stages to minimum
    Config.STAGE_1_EPOCHS = 1
    Config.STAGE_1_IMG_SIZE = 128  # Smaller size for speed
    Config.STAGE_2_EPOCHS = 1
    Config.STAGE_2_IMG_SIZE = 160

    # Use a single seed
    Config.ENSEMBLE_SEEDS = [42]

    # Ensure submission dir exists
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")

    # -------------------------------------------------------------------------
    # 2. Verify Utilities (Label Encoder)
    # -------------------------------------------------------------------------
    print("\n[2] Verifying WhaleLabelEncoder...")

    dummy_ids = ["whale_a", "whale_b", "whale_a", "new_whale", "whale_c"]
    encoder = WhaleLabelEncoder()

    # Fit
    cache_path = os.path.join(demo_dir, "classes_demo.parquet")
    encoder.fit(dummy_ids, cache_path=cache_path, load_cached_data=False)

    # Check classes
    assert (
        encoder.num_classes() == 4
    ), f"Expected 4 classes, got {encoder.num_classes()}"
    print(f"Classes found: {encoder.classes_}")

    # Transform
    encoded = encoder.transform(["whale_a", "new_whale"])
    print(f"Encoded ['whale_a', 'new_whale']: {encoded}")
    assert len(encoded) == 2

    # Inverse Transform
    decoded = encoder.inverse_transform(encoded)
    print(f"Decoded back: {decoded}")
    assert decoded[0] == "whale_a"
    assert decoded[1] == "new_whale"

    print("LabelEncoder verification passed.")

    # -------------------------------------------------------------------------
    # 3. Verify Dataset
    # -------------------------------------------------------------------------
    print("\n[3] Verifying WhaleDataset...")

    # Initialize dataset (Training mode)
    # Note: We use the encoder fitted on the real metadata implicitly via train_model later,
    # but here we just want to test data loading mechanics.
    # We'll let the dataset build its own encoder from the CSV for this isolated test.
    dataset = WhaleDataset(
        csv_path=Config.TRAIN_CSV,
        transform=get_transforms("train", image_size=128),
        debug=True,
        load_cached_data=False,
    )

    print(f"Dataset size (Debug): {len(dataset)}")
    assert len(dataset) > 0, "Dataset should not be empty."

    # Fetch one sample
    image, label, fname = dataset[0]

    print(f"Sample Image Shape: {image.shape}")
    print(f"Sample Label: {label} (Type: {type(label)})")
    print(f"Sample Filename: {fname}")

    # Assertions
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 128, 128)
    assert isinstance(label, torch.Tensor)

    print("Dataset verification passed.")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4] Verifying WhaleDenseNet Model...")

    # Instantiate model
    # We use a small embedding size for the demo
    num_classes_demo = 10
    model = WhaleDenseNet(
        num_classes=num_classes_demo,
        embedding_size=128,
        pretrained=False,  # Speed up initialization
        dropout_rate=0.1,
    )
    model.eval()

    # Create dummy input (Batch Size 2, 3 Channels, 128x128)
    dummy_input = torch.randn(2, 3, 128, 128)

    # Forward pass (Inference mode, no labels)
    embeddings = model(dummy_input)
    print(f"Embeddings Shape: {embeddings.shape}")
    assert embeddings.shape == (2, 128)  # (B, Embedding_Size)

    # Forward pass (Training mode, with labels)
    dummy_labels = torch.tensor([0, 1], dtype=torch.long)
    arcface_logits = model(dummy_input, dummy_labels)
    print(f"ArcFace Logits Shape: {arcface_logits.shape}")
    assert arcface_logits.shape == (2, num_classes_demo)  # (B, Num_Classes)

    print("Model architecture verification passed.")

    # -------------------------------------------------------------------------
    # 5. Execute Training Loop (Mini-Run)
    # -------------------------------------------------------------------------
    print("\n[5] Executing Training Loop (Mini-Run)...")

    # This calls the library function which handles:
    # - Loading data (Stage 1 & 2)
    # - Model init
    # - Training epochs
    # - Saving checkpoints
    # We use seed 42 as configured in Config.ENSEMBLE_SEEDS
    seed = 42
    try:
        train_model(seed)
        print("Training function executed successfully.")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e

    # Verify artifacts
    expected_ckpt = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")
    assert os.path.exists(expected_ckpt), f"Checkpoint not found at {expected_ckpt}"
    print(f"Checkpoint verified at: {expected_ckpt}")

    # Verify classes cache exists
    assert os.path.exists(os.path.join(Config.WORKING_DIR, "classes.parquet"))

    # -------------------------------------------------------------------------
    # 6. Verify Inference Logic with Trained Model
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Inference Logic...")

    device = torch.device(Config.DEVICE)

    # Load the trained model
    # We need to know num_classes. The training loop cached it.
    df_classes = pd.read_parquet(os.path.join(Config.WORKING_DIR, "classes.parquet"))
    num_classes_trained = len(df_classes)

    model_inf = WhaleDenseNet(
        num_classes=num_classes_trained,
        embedding_size=Config.EMBEDDING_SIZE,
        pretrained=False,
    )
    model_inf.load_state_dict(torch.load(expected_ckpt, map_location=device))
    model_inf.to(device)
    model_inf.eval()

    # Create a dummy batch for inference
    dummy_batch = torch.randn(
        4, 3, Config.STAGE_2_IMG_SIZE, Config.STAGE_2_IMG_SIZE
    ).to(device)

    # Run get_logits_inference
    with torch.no_grad():
        logits = get_logits_inference(model_inf, dummy_batch, device, tta=False)

    print(f"Inference Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (4, num_classes_trained)
    assert not torch.isnan(logits).any(), "Logits contain NaNs"

    print("Inference logic verification passed.")

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
