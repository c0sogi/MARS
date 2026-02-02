import os
import torch
import numpy as np
import pandas as pd
import warnings
import sys
import gc
import importlib


# Explicitly cleanup memory BEFORE importing heavy libraries or starting execution
# This clears zombie references from previous crashed runs (Cite debug_lesson_18)
def cleanup_memory():
    print("Cleaning up GPU and RAM...")
    if hasattr(sys, "last_traceback"):
        sys.last_traceback = None

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Cleanup complete.")


cleanup_memory()

# Reload libraries to ensure fixes are applied (Cite debug_lesson_2)
import library.config
import library.utils
import library.data

importlib.reload(library.config)
importlib.reload(library.utils)
importlib.reload(library.data)

from library.config import DEVICE, IMG_SIZE, BATCH_SIZE, SUBMISSION_PATH, WORKING_DIR
from library.utils import seed_everything, probabilistic_f1
from library.data import get_dataloaders
from library.model import PyramidSiameseEfficientNet
from library.trainer import Trainer
from library.inference import generate_predictions

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("==== Starting Library Usage Demo ====")
    print(f"Batch Size: {BATCH_SIZE}")

    # 1. Reproducibility
    print("\n[Step 1] Setting Seeds...")
    seed_everything(42)
    print("Seeds set.")

    # 2. Data Loading (Subset)
    print("\n[Step 2] Loading DataLoaders (max_samples=16)...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, max_samples=16
    )

    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches:   {len(val_loader)}")
    print(f"Test Loader Batches:  {len(test_loader)}")

    # 3. Verify Data Batch
    print("\n[Step 3] Verifying Data Batch Structure...")
    try:
        target_img, contra_img, labels, pred_ids = next(iter(train_loader))
    except StopIteration:
        print("Error: DataLoader is empty!")
        return

    # Check shapes
    expected_shape = (BATCH_SIZE, 3, IMG_SIZE[0], IMG_SIZE[1])

    print(f"Target Image Shape: {target_img.shape}")
    print(f"Contra Image Shape: {contra_img.shape}")
    print(f"Labels Shape:       {labels.shape}")

    assert (
        target_img.shape == expected_shape
    ), f"Target shape mismatch: {target_img.shape} vs {expected_shape}"
    assert (
        contra_img.shape == expected_shape
    ), f"Contra shape mismatch: {contra_img.shape} vs {expected_shape}"
    assert len(labels) == BATCH_SIZE, "Label batch size mismatch."

    print("Data batch structure verified.")

    # 4. Model Initialization
    print("\n[Step 4] Initializing PyramidSiameseEfficientNet...")
    # Wrap model allocation in try-finally to ensure cleanup (Cite debug_lesson_21)
    model = None
    try:
        model = PyramidSiameseEfficientNet()
        model = model.to(DEVICE)
        print("Model initialized and moved to device.")

        # 5. Forward Pass Verification
        print("\n[Step 5] Running Forward Pass on Batch...")
        target_img = target_img.to(DEVICE)
        contra_img = contra_img.to(DEVICE)

        with torch.no_grad():
            logits = model(target_img, contra_img)

        print(f"Logits Shape: {logits.shape}")
        assert logits.shape == (BATCH_SIZE, 1), f"Output shape mismatch: {logits.shape}"
        print("Forward pass successful.")

        # 6. Training Loop Demo
        print("\n[Step 6] Running Training Loop (1 Epoch)...")
        trainer = Trainer(model, train_loader, val_loader, test_loader)
        trainer.fit(epochs=1)

        # 7. Inference Demo
        print("\n[Step 7] Running Inference Pipeline...")
        # Explicitly delete model to free memory before inference (Cite debug_lesson_27)
        del model
        del trainer
        torch.cuda.empty_cache()

        df_submission = generate_predictions(load_cached_data=False, max_samples=16)

        if os.path.exists(SUBMISSION_PATH):
            print(f"Submission file generated at: {SUBMISSION_PATH}")
            df_check = pd.read_csv(SUBMISSION_PATH)
            print(df_check.head())
        else:
            raise FileNotFoundError("Submission file was not created.")

        print("\n==== Demo Completed Successfully ====")

    finally:
        # Final cleanup
        if model is not None:
            del model
        cleanup_memory()


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        import traceback

        traceback.print_exc()
    finally:
        cleanup_memory()
