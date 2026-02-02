import os
import sys
import types
import torch
import pandas as pd
import numpy as np

# ==========================================
# 1. Suppress Progress Bars (Mock tqdm)
# ==========================================
# We inject a silent tqdm class into sys.modules before importing the library.
# This ensures that 'from tqdm.auto import tqdm' in library files gets our silent version.


class SilentTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable is not None else []

    def __iter__(self):
        for item in self.iterable:
            yield item

    def update(self, n=1):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def close(self):
        pass


# Safely patch tqdm to ensure __spec__ exists for torch._dynamo
try:
    import tqdm
    import tqdm.auto

    tqdm.tqdm = SilentTqdm
    tqdm.auto.tqdm = SilentTqdm
except ImportError:
    from importlib.machinery import ModuleSpec

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.__spec__ = ModuleSpec(name="tqdm", loader=None)
    tqdm_module.tqdm = SilentTqdm
    sys.modules["tqdm"] = tqdm_module

    tqdm_auto_module = types.ModuleType("tqdm.auto")
    tqdm_auto_module.__spec__ = ModuleSpec(name="tqdm.auto", loader=None)
    tqdm_auto_module.tqdm = SilentTqdm
    sys.modules["tqdm.auto"] = tqdm_auto_module

# ==========================================
# 2. Import Library Modules
# ==========================================
from library import config
from library import utils
from library import dataset
from library import model
from library import trainer
from library import predict


def run_demo():
    print("==== Starting Library Usage Demo ====")

    # ==========================================
    # 3. Configuration Overrides for Speed
    # ==========================================
    print("\n[Step 1] Configuring environment for rapid execution...")
    config.seed_everything(42)

    # Override constants to run a tiny, fast experiment
    config.BATCH_SIZE = 4
    config.NUM_WORKERS = 2
    config.EPOCHS = 1

    # Define small subset sizes
    TRAIN_DEBUG_SIZE = 20
    TEST_DEBUG_SIZE = 10

    print(f"   Batch Size: {config.BATCH_SIZE}")
    print(f"   Epochs: {config.EPOCHS}")
    print(f"   Train Debug Size: {TRAIN_DEBUG_SIZE}")

    # ==========================================
    # 4. Verify Hierarchy Mappings
    # ==========================================
    print("\n[Step 2] Verifying Hierarchy Mappings...")
    mappings = utils.build_hierarchy_mappings()

    assert "cat_to_idx" in mappings
    assert "idx_to_cat" in mappings
    assert "num_classes_l3" in mappings
    assert mappings["num_classes_l3"] > 0

    print(
        f"   Successfully loaded hierarchy. Found {mappings['num_classes_l3']} L3 classes."
    )

    # ==========================================
    # 5. Verify Dataset & DataLoader
    # ==========================================
    print("\n[Step 3] Verifying Dataset and DataLoader...")

    # Instantiate Dataset directly
    ds_train = dataset.ProductDataset(subset="train", debug_size=TRAIN_DEBUG_SIZE)
    assert (
        len(ds_train) == TRAIN_DEBUG_SIZE
    ), f"Dataset length mismatch. Expected {TRAIN_DEBUG_SIZE}, got {len(ds_train)}"

    # Check single item structure
    img_tensor, labels = ds_train[0]
    # Expected shape: (4 images, 3 channels, H, W)
    expected_shape = (4, 3, config.IMG_SIZE, config.IMG_SIZE)
    assert (
        img_tensor.shape == expected_shape
    ), f"Image tensor shape mismatch. Expected {expected_shape}, got {img_tensor.shape}"
    assert (
        isinstance(labels, tuple) and len(labels) == 3
    ), "Labels must be a tuple of (l1, l2, l3)"

    # Check DataLoader
    train_loader, val_loader = dataset.get_dataloaders(debug_size=TRAIN_DEBUG_SIZE)
    batch_imgs, batch_labels = next(iter(train_loader))

    assert batch_imgs.shape[0] == config.BATCH_SIZE, "Batch size mismatch in DataLoader"
    print("   Dataset and DataLoader verified successfully.")

    # ==========================================
    # 6. Verify Model Architecture
    # ==========================================
    print("\n[Step 4] Verifying Model Architecture...")

    # Initialize model (pretrained=False for speed in initialization)
    net = model.DeepSupervisedResNet50(pretrained=False)
    net.eval()

    # Run forward pass on CPU
    with torch.no_grad():
        outputs = net(batch_imgs)

    # Check outputs
    assert "coarse" in outputs
    assert "mid" in outputs
    assert "fine" in outputs

    # Check output shape for fine-grained head
    expected_out_shape = (config.BATCH_SIZE, mappings["num_classes_l3"])
    assert (
        outputs["fine"].shape == expected_out_shape
    ), f"Model output shape mismatch. Expected {expected_out_shape}, got {outputs['fine'].shape}"

    print("   Model forward pass verified successfully.")

    # ==========================================
    # 7. Verify Training Loop (Trainer)
    # ==========================================
    print("\n[Step 5] Verifying Training Loop...")

    # Initialize Trainer
    # We use a slightly larger debug size than batch size to ensure at least one update step
    demo_trainer = trainer.Trainer(debug_size=TRAIN_DEBUG_SIZE, epochs=config.EPOCHS)

    # Run training
    print("   Running trainer.fit()...")
    demo_trainer.fit()

    # Verify checkpoint creation
    checkpoint_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"

    print(f"   Training complete. Checkpoint saved to {checkpoint_path}.")

    # ==========================================
    # 8. Verify Prediction
    # ==========================================
    print("\n[Step 6] Verifying Prediction and Submission...")

    # We use library.predict.generate_submission directly because it allows 'debug_size'
    # library.trainer.predict() would process the full test set, which is too slow for this demo.

    print("   Running inference on subset of test data...")
    predict.generate_submission(
        checkpoint_path=checkpoint_path, debug_size=TEST_DEBUG_SIZE
    )

    # Verify submission file
    assert os.path.exists(
        config.SUBMISSION_PATH
    ), f"Submission file not found at {config.SUBMISSION_PATH}"

    # Verify submission content
    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    assert len(df_sub) > 0, "Submission file is empty"
    assert "_id" in df_sub.columns, "Missing '_id' column in submission"
    assert "category_id" in df_sub.columns, "Missing 'category_id' column in submission"

    # Check that we have at least the number of rows we requested (or fewer if batching aligns perfectly)
    # Note: The predict loop breaks after processed_samples >= debug_size.
    # If batch_size=4 and debug_size=10, it processes 4, 8, 12 -> stops. So we expect ~12 rows.
    print(f"   Submission generated with {len(df_sub)} rows.")
    print("   Prediction verified successfully.")

    print("\n==== All Demonstrations Passed Successfully ====")


if __name__ == "__main__":
    run_demo()
