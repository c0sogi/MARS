import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.dataset import DogCatDataset, get_transforms
from library.models import get_model
from library.engine import fit
from library.inference import predict_ensemble
from library.utils import set_seed


def run_demo():
    print("Initializing Demo...")

    # -------------------------------------------------------------------------
    # 1. Configure for Speed and Demo
    # -------------------------------------------------------------------------
    # We modify the Config class attributes to run a fast, minimal demonstration.
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 10  # Use a tiny subset
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Disable multiprocessing for tiny data to avoid overhead

    # Use a single model architecture for the demo to save time
    demo_model_name = "convnext_tiny.fb_in1k"
    Config.MODEL_ARCHS = [demo_model_name]

    # Ensure reproducibility
    set_seed(Config.SEED)

    print(
        f"Configuration: Debug={Config.DEBUG}, Model={demo_model_name}, Device={Config.DEVICE}"
    )

    # -------------------------------------------------------------------------
    # 2. Demonstrate Dataset Loading
    # -------------------------------------------------------------------------
    print("\n--- Demonstrating Dataset ---")
    # Initialize Dataset (Train)
    train_dataset = DogCatDataset(split="train", transform=get_transforms("train"))

    # Verify dataset length matches debug samples
    assert (
        len(train_dataset) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} samples, got {len(train_dataset)}"

    # Verify item structure
    sample_img, sample_label = train_dataset[0]

    # Check Image Shape: (Channels, Height, Width)
    expected_shape = (3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert (
        sample_img.shape == expected_shape
    ), f"Image shape mismatch. Expected {expected_shape}, got {sample_img.shape}"

    # Check Label: Should be a float tensor (for BCEWithLogitsLoss)
    assert isinstance(sample_label, torch.Tensor), "Label should be a torch tensor"
    print("Dataset loaded and verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Demonstrate Model Instantiation
    # -------------------------------------------------------------------------
    print("\n--- Demonstrating Model Initialization ---")
    # Initialize model (pretrained=False for speed/offline capability in demo)
    model = get_model(demo_model_name, pretrained=False)
    model.to(Config.DEVICE)

    # Verify model output shape
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(Config.DEVICE)
    with torch.no_grad():
        output = model(dummy_input)

    # Expect output shape (Batch_Size, 1)
    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"
    print(f"Model {demo_model_name} initialized and verified.")

    # -------------------------------------------------------------------------
    # 4. Demonstrate Training (Engine)
    # -------------------------------------------------------------------------
    print("\n--- Demonstrating Training Loop ---")

    # Prepare DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    # Validation dataset (also reduced by DEBUG mode)
    val_dataset = DogCatDataset(split="val", transform=get_transforms("val"))
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Define save directory structure expected by inference engine
    # Config.WORKING_DIR/model_name
    save_dir = os.path.join(Config.WORKING_DIR, demo_model_name)

    # Run training for 1 epoch
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=Config.DEVICE,
        epochs=Config.EPOCHS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        save_dir=save_dir,
        patience=1,
    )

    # Verify that the checkpoint was saved
    checkpoint_path = os.path.join(save_dir, "model_best.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print("Training loop completed and checkpoint saved.")

    # -------------------------------------------------------------------------
    # 5. Demonstrate Inference
    # -------------------------------------------------------------------------
    print("\n--- Demonstrating Inference ---")

    # Run ensemble inference
    # This function relies on Config.MODEL_ARCHS and the checkpoints saved in WORKING_DIR
    predict_ensemble(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        device=Config.DEVICE,
        debug=True,
        debug_samples=Config.DEBUG_SAMPLES,
    )

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check row count (should match debug samples)
    assert (
        len(df_sub) == Config.DEBUG_SAMPLES
    ), f"Submission length mismatch. Expected {Config.DEBUG_SAMPLES}, got {len(df_sub)}"

    # Check columns
    assert list(df_sub.columns) == [
        "id",
        "label",
    ], f"Submission columns mismatch. Got {list(df_sub.columns)}"

    # Check value range
    assert (
        df_sub["label"].min() >= 0 and df_sub["label"].max() <= 1
    ), "Probabilities out of range [0, 1]"

    print(f"Inference successful. Submission generated at {Config.SUBMISSION_PATH}")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
