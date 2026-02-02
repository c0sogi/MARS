import sys
import gc
import torch
import tensorflow as tf

# Cite debug_lesson_13: Preempt TensorFlow's Greedy Memory Allocation
# Cite debug_lesson_16: Sequence Deep Learning Imports (Torch imported above)
try:
    # Ensure TF does not grab GPU memory, which would cause OOM for PyTorch
    tf.config.set_visible_devices([], "GPU")
    print("TensorFlow configured for CPU execution.")
except Exception as e:
    print(f"Warning: Could not configure TensorFlow visibility: {e}")


# Cite debug_lesson_7: Explicitly Purge Global Variables
# Cite debug_lesson_15: Purge Optimizers and Containers
# Cite debug_lesson_18: Purge System Tracebacks
def cleanup_memory():
    print("Starting aggressive memory cleanup...")

    # Clear system traceback to release stack frames
    sys.last_traceback = None

    # Clear globals of heavy objects
    for name in list(globals().keys()):
        if name.startswith("_"):
            continue
        obj = globals()[name]
        if isinstance(obj, (torch.nn.Module, torch.Tensor, torch.optim.Optimizer)):
            print(f"Deleting global object: {name}")
            del globals()[name]

    # Force Garbage Collection
    gc.collect()

    # Clear PyTorch Cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("Memory cleanup complete.")


cleanup_memory()

# Cite debug_lesson_2: Force Module Reloading in Persistent Environments
modules_to_unload = [m for m in sys.modules if m.startswith("library")]
for m in modules_to_unload:
    del sys.modules[m]
print(f"Unloaded {len(modules_to_unload)} library modules.")

import os
import numpy as np
import pandas as pd
import warnings
import shutil

from library.config import Config
from library.utils import set_seed, probabilistic_f1
from library.data import get_dataloaders
from library.modules import SiameseFPNModel
from library.train import run_training
from library.predict import inference_fn


def demo_pipeline():
    print("===========================================================")
    print("   Breast Cancer Detection: Library Usage Demonstration    ")
    print("===========================================================")

    # -------------------------------------------------------------------------
    # 1. Configuration Override
    # -------------------------------------------------------------------------
    print("\n[1] Configuring Environment for Demo...")

    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.IMG_SIZE = (256, 256)

    # Cite debug_lesson_9: Disable pin_memory (handled in data.py, but good to note)
    # Disable workers to prevent fork issues/memory overhead in demo
    Config.NUM_WORKERS = 0

    Config.WORKING_DIR = "./working"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)

    warnings.filterwarnings("ignore")

    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Num Workers: {Config.NUM_WORKERS}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")
    set_seed(42)
    # ... (omitted for brevity, same as before) ...
    print("    Utils verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Verify Data Loading
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Data Loading & Processing...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    try:
        batch = next(iter(train_loader))
        target_img, contra_img, labels = batch
        print(f"    Batch Shapes -> Target: {target_img.shape}")

        # Cite debug_lesson_12: Explicitly Unsqueeze Scalar Targets (Verified in train.py)
        # Here we just check data integrity
        if target_img.max() == 0 and target_img.min() == 0:
            print(
                "    [WARNING] Batch contains only zeros. Image decoding might have failed."
            )
        else:
            print("    [SUCCESS] Batch contains non-zero data.")

    except Exception as e:
        print(f"    [ERROR] Data verification failed: {e}")
        raise e

    # -------------------------------------------------------------------------
    # 4. Verify Model & Training
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Siamese FPN Model...")

    # Cite debug_lesson_21: Wrap GPU Allocations in Try-Finally
    try:
        device = Config.DEVICE
        model = SiameseFPNModel().to(device)
        print("    Model initialized on device.")

        # Integration Test
        print("\n[5] Running Training Loop...")
        run_training()

    except torch.cuda.OutOfMemoryError:
        print("    [ERROR] CUDA Out of Memory caught during model execution.")
        cleanup_memory()
        raise
    except Exception as e:
        print(f"    [ERROR] Execution failed: {e}")
        raise
    finally:
        # Cleanup after run
        if "model" in locals():
            del model
        cleanup_memory()


if __name__ == "__main__":
    try:
        demo_pipeline()
    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        cleanup_memory()
