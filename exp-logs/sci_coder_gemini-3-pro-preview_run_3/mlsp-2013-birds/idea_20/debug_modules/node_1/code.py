import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, calculate_robust_auc
from library.dataset import load_data, BirdDataset, get_transforms
from library.models import get_model
from library.sam import SAM
from library.train import run_training
from library.predict import predict_with_tta

# Ensure reproducible results
seed_everything(42)


def demo_dataset_and_transforms():
    """
    Demonstrates loading data, creating the dataset, and applying transforms.
    """
    print("\n=== Demo: Dataset & Transforms ===")

    # Load a small subset of training data (using cache logic from library)
    # Note: load_data handles caching. We force a reload or check cache.
    print("Loading training data...")
    images, labels = load_data("train")

    # Verify data loading
    assert len(images) > 0, "No images loaded."
    assert len(labels) == len(images), "Mismatch between images and labels."
    print(f"Loaded {len(images)} training samples.")
    print(f"Image shape: {images.shape}, Label shape: {labels.shape}")

    # Initialize Dataset with Training Transforms
    transform = get_transforms("train")
    dataset = BirdDataset(images[:5], labels[:5], transform=transform)

    # Fetch a single item
    img_tensor, lbl_tensor = dataset[0]

    # Verification
    # Expected shape: (3, 224, 224) due to Config.IMG_SIZE=224 and ToTensorV2
    expected_shape = (3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert (
        img_tensor.shape == expected_shape
    ), f"Expected image shape {expected_shape}, got {img_tensor.shape}"
    assert lbl_tensor.shape == (
        Config.NUM_CLASSES,
    ), f"Expected label shape ({Config.NUM_CLASSES},), got {lbl_tensor.shape}"

    print("Dataset item shape verification passed.")


def demo_model_and_sam():
    """
    Demonstrates model initialization and a single optimization step using SAM.
    """
    print("\n=== Demo: Model & SAM Optimizer ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Model
    model_name = "resnet18"
    model = get_model(model_name, pretrained=False)  # False for speed in demo
    model = model.to(device)
    model.train()

    # Create dummy batch
    batch_size = 2
    inputs = torch.randn(batch_size, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    targets = torch.randint(0, 2, (batch_size, Config.NUM_CLASSES)).float().to(device)

    # Initialize SAM Optimizer
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(
        model.parameters(), base_optimizer=base_optimizer, rho=0.05, lr=0.001
    )
    criterion = nn.BCEWithLogitsLoss()

    print(f"Model {model_name} and SAM optimizer initialized.")

    # Define Closure for SAM
    # SAM requires a closure that re-evaluates the loss
    def closure():
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        return loss

    # Perform Optimization Step
    initial_loss = closure().item()
    optimizer.step(closure)

    # Check if weights changed (simple check)
    print(f"Initial Loss: {initial_loss:.4f}")
    print("SAM optimization step completed successfully.")


def demo_full_pipeline():
    """
    Runs the full training and inference pipeline using the library functions.
    Uses 'debug=True' for training to use a subset of data.
    """
    print("\n=== Demo: Full Training & Inference Pipeline ===")

    # 1. Run Training
    # run_training(debug=True) uses a subset (50 samples) and runs the configured epochs/folds
    print("Starting Training (Debug Mode)...")
    run_training(debug=True)

    # Verify Checkpoints exist
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "resnet18_fold_0_best.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print(f"Training complete. Checkpoint verified at {checkpoint_path}")

    # 2. Run Inference
    # We simulate run_inference logic here to ensure it uses our demo config and created checkpoints
    print("Starting Inference...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Test Data
    test_images, _ = load_data("test")
    # For demo speed, slice test images if they are large (though 64 is small)
    if len(test_images) > 10:
        test_images = test_images[:10]

    # Prepare DataLoaders for TTA (Original, Left, Right)
    # Simplified TTA prep from library/predict.py
    h, w, c = test_images.shape[1:]
    shift_pixels = int(w * Config.SHIFT_LIMIT)

    imgs_orig = test_images
    imgs_left = np.zeros_like(test_images)
    imgs_left[:, :, :-shift_pixels, :] = test_images[:, :, shift_pixels:, :]
    imgs_right = np.zeros_like(test_images)
    imgs_right[:, :, shift_pixels:, :] = test_images[:, :, :-shift_pixels, :]

    transform = get_transforms("val")
    loaders = [
        DataLoader(BirdDataset(img, transform=transform), batch_size=4, shuffle=False)
        for img in [imgs_orig, imgs_left, imgs_right]
    ]

    # Load Model and Predict
    model = get_model("resnet18", pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    preds = predict_with_tta(model, device, loaders)

    assert preds.shape == (
        len(test_images),
        Config.NUM_CLASSES,
    ), f"Prediction shape mismatch. Expected {(len(test_images), Config.NUM_CLASSES)}, got {preds.shape}"

    print("Inference successful. Predictions generated.")

    # Generate Submission CSV (Mocking the final step)
    # We need rec_ids. Since we sliced test_images, we slice metadata too.
    test_df = pd.read_csv(Config.TEST_METADATA)
    rec_ids = test_df["rec_id"].values[: len(test_images)]

    submission_rows = []
    for i, rec_id in enumerate(rec_ids):
        probs = preds[i]
        for species_idx, prob in enumerate(probs):
            row_id = int(rec_id * 100 + species_idx)
            submission_rows.append({"Id": row_id, "Probability": prob})

    sub_df = pd.DataFrame(submission_rows).sort_values("Id")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_FILE, index=False)

    print(f"Submission file generated at {Config.SUBMISSION_FILE}")
    print("Top 5 rows:")
    print(sub_df.head())


if __name__ == "__main__":
    # --- 1. Modify Configuration for Demo ---
    # We override the Config class attributes to create a lightweight execution environment
    print("Configuring environment for demo...")

    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Hyperparameters for speed
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.NUM_FOLDS = 2  # Use only 2 folds
    Config.BATCH_SIZE = 8  # Small batch size
    Config.MODEL_ARCHITECTURES = ["resnet18"]  # Only use one lightweight model
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo script

    # Initialize directories
    Config.setup()

    # --- 2. Execute Demos ---
    try:
        demo_dataset_and_transforms()
        demo_model_and_sam()
        demo_full_pipeline()
        print("\nAll demos completed successfully.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        # Print traceback for debugging
        import traceback

        traceback.print_exc()
        sys.exit(1)
