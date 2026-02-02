import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
import gc

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, probabilistic_f1, load_dicom_image
from library.transforms import PairedAugmentation
from library.data import get_loaders, SiameseMammogramDataset
from library.model import PyramidDiffSiameseNet
from library.engine import run_training, make_submission


def cleanup_resources():
    """
    Aggressively cleans up GPU memory by purging system tracebacks and global variables.
    """
    # Cite debug_lesson_18: Purge System Tracebacks to Release Zombie GPU Memory
    if hasattr(sys, "last_traceback"):
        del sys.last_traceback
    if hasattr(sys, "last_value"):
        del sys.last_value
    if hasattr(sys, "last_type"):
        del sys.last_type

    # Cite debug_lesson_7: Explicitly Purge Global Variables
    whitelist = {
        "sys",
        "os",
        "np",
        "pd",
        "torch",
        "cv2",
        "gc",
        "Config",
        "seed_everything",
        "probabilistic_f1",
        "load_dicom_image",
        "PairedAugmentation",
        "get_loaders",
        "SiameseMammogramDataset",
        "PyramidDiffSiameseNet",
        "run_training",
        "make_submission",
        "run_demo",
        "main",
        "cleanup_resources",
        "__name__",
        "__builtins__",
        "__doc__",
        "__package__",
        "__loader__",
        "__spec__",
        "__file__",
    }

    for name in list(globals().keys()):
        # Cite debug_lesson_19: Include Hidden REPL Variables in Memory Cleanup
        if name not in whitelist:
            try:
                del globals()[name]
            except:
                pass

    gc.collect()
    torch.cuda.empty_cache()


def run_demo():
    print("==== Starting Demonstration Script ====")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring Environment for Demo...")

    # Monkey-patch Config for speed and debugging
    Config.DEBUG = True
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4  # Smaller batch size for safety
    Config.NUM_WORKERS = 2
    # Cite debug_lesson_9: Disable pin_memory to Resolve Data Loader Initialization OOMs
    Config.PIN_MEMORY = False
    # Reduce Image Size to prevent OOM and speed up demo
    Config.IMAGE_SIZE = (256, 256)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"    Device: {device}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Image Size: {Config.IMAGE_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utilities...")

    # Test Probabilistic F1 Score
    y_true = np.array([1, 0, 1])
    y_pred_perfect = np.array([1.0, 0.0, 1.0])
    pf1_perfect = probabilistic_f1(y_true, y_pred_perfect)
    assert np.isclose(pf1_perfect, 1.0), f"Expected pF1=1.0, got {pf1_perfect}"

    y_pred_uncertain = np.array([0.5, 0.5, 0.5])
    pf1_uncertain = probabilistic_f1(y_true, y_pred_uncertain)
    assert 0.5 < pf1_uncertain < 0.6, f"Expected pF1 ~0.57, got {pf1_uncertain}"

    print("    probabilistic_f1: Passed")

    # Test Image Loading
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    sample_rel_path = df_train.iloc[0]["file_path"]
    sample_full_path = os.path.join(Config.INPUT_DIR, sample_rel_path)

    if os.path.exists(sample_full_path):
        img = load_dicom_image(sample_full_path)
        assert isinstance(img, np.ndarray), "Image should be a numpy array"
        assert img.ndim >= 2, "Image should be at least 2D"
        print(f"    load_dicom_image: Passed (Loaded {img.shape})")
    else:
        print("    load_dicom_image: Skipped (Sample file not found in input)")

    # -------------------------------------------------------------------------
    # 3. Verify Transforms
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Transforms...")

    augmenter = PairedAugmentation()

    # Create dummy images (H, W)
    dummy_target = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    dummy_contra = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

    # Test Train Mode
    t_target, t_contra = augmenter(dummy_target, dummy_contra, mode="train")

    # Checks
    assert torch.is_tensor(t_target), "Output should be a tensor"
    assert t_target.shape == (
        1,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    ), f"Expected shape (1, {Config.IMAGE_SIZE[0]}, {Config.IMAGE_SIZE[1]}), got {t_target.shape}"
    assert t_target.shape == t_contra.shape, "Target and Contra shapes must match"

    print("    PairedAugmentation: Passed")

    # -------------------------------------------------------------------------
    # 4. Verify Data Loading
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Data Loading...")

    # Initialize Loaders (Debug mode is active, so this will be fast)
    loaders = get_loaders(load_cached_data=False)
    train_loader = loaders["train"]

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify Batch Structure
    assert "image" in batch
    assert "image_contra" in batch
    assert "label" in batch

    # Verify Dimensions
    imgs = batch["image"]
    assert imgs.shape[0] == Config.BATCH_SIZE
    assert imgs.shape[1] == 3
    assert imgs.shape[2] == Config.IMAGE_SIZE[0]

    print(f"    DataLoader: Passed (Batch shape {imgs.shape})")

    # -------------------------------------------------------------------------
    # 5. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Model Architecture...")

    model = PyramidDiffSiameseNet().to(device)

    # Forward pass with the batch from step 4
    images_t = batch["image"].to(device)
    images_c = batch["image_contra"].to(device)

    logits = model(images_t, images_c)

    # Verify Output
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected output (B, 1), got {logits.shape}"

    print("    PyramidDiffSiameseNet: Passed")

    # -------------------------------------------------------------------------
    # 6. Verify Training Engine
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Training Engine...")

    # Run a short training loop (1 epoch, debug subset)
    run_training(model, loaders["train"], loaders["val"], device)

    # Check if checkpoint was saved
    if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        print("    Training: Passed (Checkpoint saved)")
    else:
        # If model didn't improve (possible in 1 epoch on random data), save manually for next step
        torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
        print("    Training: Completed (Manual checkpoint saved for inference test)")

    # -------------------------------------------------------------------------
    # 7. Verify Inference Engine
    # -------------------------------------------------------------------------
    print("\n[7] Verifying Inference Engine...")

    # Generate submission
    make_submission(model, loaders["test"], device)

    # Verify Submission File
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)

        # Check columns
        assert "prediction_id" in sub_df.columns
        assert "cancer" in sub_df.columns

        # Check rows (Debug mode limits test set to 50 rows, but aggregation might reduce it)
        # We just check it's not empty
        assert len(sub_df) > 0

        # Check probability range
        probs = sub_df["cancer"].values
        assert np.all((probs >= 0) & (probs <= 1)), "Probabilities must be in [0, 1]"

        print(f"    Inference: Passed (Generated {len(sub_df)} predictions)")
        print(f"    Submission saved to: {Config.SUBMISSION_PATH}")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n==== Demonstration Complete ====")


def main():
    # Perform cleanup before starting
    cleanup_resources()

    try:
        run_demo()
    finally:
        # Cite debug_lesson_21: Wrap GPU Allocations in Try-Finally
        cleanup_resources()


if __name__ == "__main__":
    main()
