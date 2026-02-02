import os
import shutil
import torch
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint
from library.data_loader import get_dataloaders
from library.model import HybridSECNN
from library.train import run_fold


def demo_utils():
    print("\n=== Demo: Utils ===")
    # 1. Test Seed
    set_seed(42)
    print("Random seed set to 42.")

    # 2. Test Checkpointing
    dummy_state = {
        "epoch": 1,
        "model_state": {"layer1.weight": torch.tensor([0.1, 0.2])},
        "optimizer_state": {},
    }
    ckpt_name = "demo_checkpoint.pth"

    print(f"Saving dummy checkpoint to {Config.WORKING_DIR}...")
    save_checkpoint(dummy_state, ckpt_name)

    print("Loading checkpoint...")
    loaded_state = load_checkpoint(ckpt_name, device="cpu")

    # Verification
    assert loaded_state["epoch"] == 1
    assert torch.equal(
        loaded_state["model_state"]["layer1.weight"], torch.tensor([0.1, 0.2])
    )
    print("Checkpoint save/load verified successfully.")


def demo_data_loader():
    print("\n=== Demo: Data Loader ===")
    # Trigger data processing and loader creation
    # We use load_cached_data=False to ensure the processing logic runs
    print("Initializing DataLoaders (forcing data processing)...")
    train_loader, val_loader, test_loader = get_dataloaders(
        fold_idx=0,
        load_cached_data=False,
        debug=True,  # Uses Config.DEBUG_SUBSET_SIZE (100 samples)
    )

    # 1. Verify Train Loader
    print("Verifying Train Loader batch...")
    images, angles, labels = next(iter(train_loader))

    # Expected: (Batch, 3, 75, 75)
    assert images.dim() == 4
    assert images.shape[1] == 3
    assert images.shape[2] == 75
    assert images.shape[3] == 75
    # Expected: (Batch,)
    assert angles.dim() == 1
    assert labels.dim() == 1
    assert images.shape[0] == angles.shape[0] == labels.shape[0]

    print(
        f"Train Batch Shapes: Images {images.shape}, Angles {angles.shape}, Labels {labels.shape}"
    )

    # 2. Verify Test Loader
    print("Verifying Test Loader batch...")
    t_images, t_angles, t_ids = next(iter(test_loader))

    assert t_images.dim() == 4
    assert t_images.shape[1] == 3
    assert t_ids.shape[0] == t_images.shape[0]

    print(f"Test Batch Shapes: Images {t_images.shape}, IDs {t_ids.shape}")

    return images, angles


def demo_model(sample_images, sample_angles):
    print("\n=== Demo: Model ===")
    device = Config.DEVICE
    print(f"Instantiating HybridSECNN on {device}...")

    model = HybridSECNN().to(device)

    # Prepare inputs
    x = sample_images.to(device)
    a = sample_angles.to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x, a)

    print(f"Model Output Shape: {output.shape}")

    # Verification
    # Output should be (Batch, 1) logits
    assert output.shape == (x.size(0), 1)
    assert torch.isfinite(output).all(), "Model output contains NaNs or Infs"
    print("Model forward pass verified successfully.")


def demo_training():
    print("\n=== Demo: Training Loop ===")
    print(f"Running Fold 0 for {Config.EPOCHS} epochs (Debug Mode)...")

    # run_fold handles the full training loop including validation and saving
    best_loss = run_fold(fold_idx=0)

    print(f"Training completed. Best Validation Loss: {best_loss}")

    # Verify artifact creation
    expected_model_path = os.path.join(Config.WORKING_DIR, "model_fold_0.pth")
    assert os.path.exists(
        expected_model_path
    ), f"Model file not found at {expected_model_path}"
    print(f"Verified existence of saved model: {expected_model_path}")


if __name__ == "__main__":
    # 1. Configure for Demo
    # We override Config attributes to ensure the demo runs quickly and
    # in a contained environment.
    Config.WORKING_DIR = "./working/demo_run"
    Config.DEBUG = True
    Config.EPOCHS = 2
    Config.N_FOLDS = 2  # Reduced folds, though we only run fold 0

    # Ensure clean state
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR)

    print(f"Demo Configuration:")
    print(f"  Working Dir: {Config.WORKING_DIR}")
    print(f"  Debug Mode: {Config.DEBUG}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Device: {Config.DEVICE}")

    # 2. Run Demos
    try:
        demo_utils()

        # Get sample data from loader demo to pass to model demo
        imgs, angles = demo_data_loader()

        demo_model(imgs, angles)

        demo_training()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\nAssertion Failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        exit(1)
