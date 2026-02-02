import os
import sys
import pandas as pd
import torch
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_class_weights
from library.dataset import get_loaders
from library.model import get_model
from library.engine import run_swa_training, predict_and_submit


def main():
    print("==== Starting Demonstration Script ====")

    # 1. Configuration Overrides for Speed
    # We modify the Config class attributes directly to ensure the demo runs quickly.
    print("Configuring parameters for rapid execution...")
    Config.BURN_IN_EPOCHS = 1
    Config.SWA_EPOCHS = 1
    Config.EPOCHS = Config.BURN_IN_EPOCHS + Config.SWA_EPOCHS
    Config.BATCH_SIZE = 8
    Config.PRETRAINED = False  # Disable downloading weights for speed/offline safety
    Config.NUM_WORKERS = 2  # Reduce workers for small subset

    # Ensure working directory structure exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Reproducibility
    seed_everything(Config.SEED)
    print(f"Random seed set to {Config.SEED}")

    # 3. Create Data Subsets (Mocking Cache)
    # To make the training loop run in seconds, we create small subset parquet files.
    # The library's load_dataframes function checks for these files first.
    print("\nCreating data subsets for demonstration...")

    if not os.path.exists(Config.TRAIN_METADATA_PATH) or not os.path.exists(
        Config.TEST_METADATA_PATH
    ):
        raise FileNotFoundError(
            "Metadata files not found. Please ensure ./metadata exists."
        )

    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Take a tiny subset: 32 train samples (4 batches), 16 test samples (2 batches)
    train_subset = train_meta.head(32).copy()
    test_subset = test_meta.head(16).copy()

    train_cache_path = os.path.join(Config.WORKING_DIR, "train_df.parquet")
    test_cache_path = os.path.join(Config.WORKING_DIR, "test_df.parquet")

    train_subset.to_parquet(train_cache_path, index=False)
    test_subset.to_parquet(test_cache_path, index=False)
    print(f"Subset cached: Train={len(train_subset)}, Test={len(test_subset)}")

    # 4. Verify Utility Functions
    print("\n[Test] library.utils.calculate_class_weights")
    # Note: This reads the full metadata CSV, not the parquet cache, which is fine/fast.
    weights = calculate_class_weights(
        metadata_path=Config.TRAIN_METADATA_PATH,
        target_columns=Config.CLASS_LABELS,
        load_cached_data=False,  # Force recalculation for demo
    )
    print(f"Class Weights: {weights}")
    assert isinstance(weights, torch.Tensor), "Weights should be a torch Tensor"
    assert weights.shape == (4,), f"Expected weights shape (4,), got {weights.shape}"
    assert weights.device.type == Config.DEVICE, "Weights not on configured device"

    # 5. Verify Dataset and Loaders
    print("\n[Test] library.dataset.get_loaders")
    # This should pick up our cached parquet subsets
    train_loader, test_loader = get_loaders(
        load_cached_data=True, batch_size=Config.BATCH_SIZE
    )

    assert len(train_loader.dataset) == 32, "Train loader did not load subset"
    assert len(test_loader.dataset) == 16, "Test loader did not load subset"

    # Fetch one batch
    images, targets = next(iter(train_loader))
    print(f"Batch Shapes - Images: {images.shape}, Targets: {targets.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect image batch shape"
    assert targets.shape == (Config.BATCH_SIZE, 4), "Incorrect target batch shape"
    assert images.dtype == torch.float32, "Images should be float32"

    # 6. Verify Model
    print("\n[Test] library.model.get_model")
    model = get_model(pretrained=False)
    model.eval()

    # Move batch to device
    images = images.to(Config.DEVICE)

    with torch.no_grad():
        outputs = model(images)

    print(f"Model Output Shape: {outputs.shape}")
    assert outputs.shape == (Config.BATCH_SIZE, 4), "Model output shape mismatch"

    # 7. Verify Engine - Training Loop
    print("\n[Test] library.engine.run_swa_training")
    print("Running training loop (2 epochs on subset)...")

    # This function handles the full loop: Burn-in -> SWA -> BN Update -> Save
    swa_model = run_swa_training()

    assert isinstance(
        swa_model, torch.nn.Module
    ), "run_swa_training did not return a model"

    expected_model_path = os.path.join(Config.MODEL_OUTPUT_DIR, "swa_model.pth")
    if os.path.exists(expected_model_path):
        print(f"Model successfully saved to {expected_model_path}")
    else:
        raise FileNotFoundError("SWA model file was not saved.")

    # 8. Verify Engine - Prediction & Submission
    print("\n[Test] library.engine.predict_and_submit")
    predict_and_submit(swa_model)

    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file saved to {Config.SUBMISSION_PATH}")
    else:
        raise FileNotFoundError("Submission file was not created.")

    # Check submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission DataFrame Shape: {sub_df.shape}")
    print("First 3 rows of submission:")
    print(sub_df.head(3))

    assert sub_df.shape == (
        16,
        5,
    ), f"Expected submission shape (16, 5), got {sub_df.shape}"
    assert (
        list(sub_df.columns) == ["image_id"] + Config.CLASS_LABELS
    ), "Submission columns mismatch"

    # Verify probabilities sum to roughly 1 (optional, depending on model output, here just checking range)
    # Since we used softmax in predict_and_submit, rows should sum to 1.
    row_sums = sub_df[Config.CLASS_LABELS].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"

    print("\n==== All demonstrations and verifications passed successfully! ====")


if __name__ == "__main__":
    main()
