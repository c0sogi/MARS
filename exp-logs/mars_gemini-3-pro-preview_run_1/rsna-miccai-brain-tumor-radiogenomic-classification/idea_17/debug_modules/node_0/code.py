import os
import sys
import pandas as pd
import torch
import numpy as np
import warnings

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import BraTSDataset, get_transforms
from library.model import AAWIISNet
from library.train import run_training

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("--- Starting Library Usage Demonstration ---")

    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config for speed
    # Note: We pass these as arguments to functions where possible,
    # or rely on the fact that we are using small datasets.
    DEMO_BATCH_SIZE = 2
    DEMO_EPOCHS = 1

    # 2. Prepare Mini-Datasets
    # Load original metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    full_train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Select a tiny subset: 4 subjects for train, 2 for val
    # This ensures the code runs in seconds rather than hours
    mini_train_df = full_train_df.head(4).copy()
    mini_val_df = full_train_df.iloc[4:6].copy()  # Use next 2 for val

    mini_train_path = os.path.join(demo_dir, "mini_train.csv")
    mini_val_path = os.path.join(demo_dir, "mini_val.csv")

    mini_train_df.to_csv(mini_train_path, index=False)
    mini_val_df.to_csv(mini_val_path, index=False)

    print(f"Created mini-datasets: Train={len(mini_train_df)}, Val={len(mini_val_df)}")

    # 3. Verify Dataset and DataLoader
    print("\n--- Verifying Dataset Logic ---")

    # Instantiate dataset
    # We use a specific cache name to avoid messing with the main training cache
    dataset = BraTSDataset(
        mini_train_df,
        transform=get_transforms("train"),
        is_train=True,
        cache_name="demo_train_roi",
    )

    # Check Length: 4 subjects * 3 slab depths = 12 samples
    expected_len = len(mini_train_df) * len(Config.SLAB_DEPTHS)
    print(f"Dataset length: {len(dataset)} (Expected: {expected_len})")
    if len(dataset) != expected_len:
        raise AssertionError(
            f"Dataset length mismatch. Got {len(dataset)}, expected {expected_len}"
        )

    # Check Item Structure
    sample_img, sample_target = dataset[0]

    # Shape check: (9, 224, 224) -> 3 modalities * 3 slices
    print(f"Sample image shape: {sample_img.shape}")
    if sample_img.shape != (9, Config.IMAGE_SIZE, Config.IMAGE_SIZE):
        raise AssertionError(
            f"Image shape mismatch. Expected (9, {Config.IMAGE_SIZE}, {Config.IMAGE_SIZE}), got {sample_img.shape}"
        )

    # Target check
    print(f"Sample target: {sample_target}")
    if not isinstance(sample_target, torch.Tensor):
        raise AssertionError("Target is not a tensor")

    # 4. Verify Model Architecture
    print("\n--- Verifying Model Architecture ---")

    # Initialize model (pretrained=False for speed/offline demo)
    model = AAWIISNet(pretrained=False)
    model.eval()

    # Create a dummy batch
    # Unsqueeze to add batch dimension: (1, 9, 224, 224)
    input_tensor = sample_img.unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    print(f"Model output shape: {output.shape}")

    # Expect output shape (Batch_Size, 1)
    if output.shape != (1, 1):
        raise AssertionError(
            f"Model output shape mismatch. Expected (1, 1), got {output.shape}"
        )

    print("Model forward pass successful.")

    # 5. Verify Training Loop
    print("\n--- Verifying Training Pipeline ---")

    # Run the training function provided in library.train
    # We pass the paths to our mini-datasets
    best_model_path = run_training(
        train_metadata_path=mini_train_path,
        val_metadata_path=mini_val_path,
        output_dir=demo_dir,
        num_epochs=DEMO_EPOCHS,
        batch_size=DEMO_BATCH_SIZE,
        learning_rate=1e-4,
        weight_decay=1e-2,
        patience=1,
    )

    # Check if artifacts were created
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Training did not produce a model file at {best_model_path}"
        )

    print(f"Training complete. Model saved to: {best_model_path}")

    # 6. Verify Inference with Saved Model
    print("\n--- Verifying Inference with Saved Model ---")

    # Load the saved model
    loaded_model = AAWIISNet(pretrained=False)
    loaded_model.load_state_dict(
        torch.load(best_model_path, map_location=Config.DEVICE)
    )
    loaded_model.to(Config.DEVICE)
    loaded_model.eval()

    # Run inference on one sample
    input_tensor = input_tensor.to(Config.DEVICE)
    with torch.no_grad():
        logits = loaded_model(input_tensor)
        probs = torch.sigmoid(logits)

    print(f"Inference Probability: {probs.item():.4f}")
    if not (0.0 <= probs.item() <= 1.0):
        raise AssertionError("Probability output is out of bounds [0, 1]")

    print("\n--- Demonstration Complete Successfully ---")


if __name__ == "__main__":
    run_demo()
