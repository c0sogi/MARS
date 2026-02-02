import os
import sys
import shutil
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import seed_everything
from library.dataset import load_data_to_memory, CactusDataset, get_transforms
from library.model import MetadataFusedRepVGG, RepVGGBlock
from library.trainer import Trainer
from library.inference import run_inference


def setup_demo_config():
    """
    Modifies the global Config to use a demo directory and fast training parameters.
    """
    print("Setting up Demo Configuration...")

    # Define a specific directory for this demo run
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Patch Config paths
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Cache paths
    Config.CACHE_TRAIN_IMGS = os.path.join(demo_dir, "cache_train_imgs.npy")
    Config.CACHE_TRAIN_LABELS = os.path.join(demo_dir, "cache_train_labels.npy")
    Config.CACHE_TRAIN_FILESIZES = os.path.join(demo_dir, "cache_train_filesizes.npy")
    Config.CACHE_VAL_IMGS = os.path.join(demo_dir, "cache_val_imgs.npy")
    Config.CACHE_VAL_LABELS = os.path.join(demo_dir, "cache_val_labels.npy")
    Config.CACHE_VAL_FILESIZES = os.path.join(demo_dir, "cache_val_filesizes.npy")
    Config.CACHE_TEST_IMGS = os.path.join(demo_dir, "cache_test_imgs.npy")
    Config.CACHE_TEST_IDS = os.path.join(demo_dir, "cache_test_ids.npy")
    Config.CACHE_TEST_FILESIZES = os.path.join(demo_dir, "cache_test_filesizes.npy")

    # Model paths
    Config.BEST_MODEL_PATH = os.path.join(demo_dir, "best_model_demo.pth")
    Config.FINAL_SWA_MODEL_PATH = os.path.join(demo_dir, "final_swa_model_demo.pth")

    # Reduce compute parameters for speed
    Config.EPOCHS = 2  # Run 2 epochs to test transition logic
    Config.SWA_START_EPOCH = 1  # Start SWA after epoch 1
    Config.BATCH_SIZE = 64
    Config.NUM_WORKERS = 2

    Config.print_config()
    return demo_dir


def verify_dataset_logic():
    """
    Verifies data loading, caching, and Dataset item retrieval.
    """
    print("\n--- Verifying Dataset Logic ---")

    # 1. Test load_data_to_memory (Train split)
    print("Loading training data...")
    imgs, labels, filesizes, ids = load_data_to_memory(
        Config.TRAIN_METADATA_PATH,
        Config.CACHE_TRAIN_IMGS,
        Config.CACHE_TRAIN_FILESIZES,
        Config.CACHE_TRAIN_LABELS,
        load_cached_data=False,  # Force load from CSV/Images first time
    )

    assert len(imgs) > 0, "No images loaded"
    assert len(imgs) == len(labels) == len(filesizes), "Data length mismatch"
    assert imgs.shape[1:] == (32, 32, 3), f"Incorrect image shape: {imgs.shape}"
    assert imgs.dtype == np.float32, "Images should be float32"

    print(f"Loaded {len(imgs)} training samples.")

    # 2. Test CactusDataset
    print("Instantiating CactusDataset...")
    dataset = CactusDataset(
        images=imgs,
        filesizes=filesizes,
        labels=labels,
        transform=get_transforms(mode="train"),
        filesize_mean=filesizes.mean(),
        filesize_std=filesizes.std(),
    )

    # 3. Verify __getitem__
    item_idx = 0
    (img_tensor, meta_tensor), label_tensor = dataset[item_idx]

    # Check shapes and types
    assert isinstance(img_tensor, torch.Tensor), "Image should be a Tensor"
    assert img_tensor.shape == (
        3,
        32,
        32,
    ), f"Image tensor shape mismatch: {img_tensor.shape}"
    assert isinstance(meta_tensor, torch.Tensor), "Metadata should be a Tensor"
    assert meta_tensor.ndim == 0 or meta_tensor.shape == (
        1,
    ), "Metadata should be scalar or size 1"
    assert isinstance(label_tensor, torch.Tensor), "Label should be a Tensor"

    print("Dataset verification successful.")
    return dataset


def verify_model_logic():
    """
    Verifies model instantiation, forward pass, and re-parameterization.
    """
    print("\n--- Verifying Model Logic ---")

    device = Config.DEVICE
    model = MetadataFusedRepVGG(num_classes=1, deploy=False).to(device)

    # 1. Test Forward Pass (Training Mode)
    batch_size = 4
    dummy_img = torch.randn(batch_size, 3, 32, 32).to(device)
    dummy_meta = torch.randn(batch_size, 1).to(device)

    print("Running forward pass (Training Mode)...")
    output = model((dummy_img, dummy_meta))

    assert output.shape == (batch_size, 1), f"Output shape mismatch: {output.shape}"

    # Check if RepVGGBlock has training branches
    block = model.stage1[0]
    assert hasattr(
        block, "rbr_dense"
    ), "Model should have dense branch in training mode"
    assert hasattr(block, "rbr_1x1"), "Model should have 1x1 branch in training mode"
    assert not hasattr(block, "rbr_reparam"), "Model should not have reparam branch yet"

    # 2. Test Switch to Deploy
    print("Switching to deploy mode...")
    model.switch_to_deploy()

    # Check if branches are fused
    assert hasattr(
        block, "rbr_reparam"
    ), "Model should have reparam branch in deploy mode"
    assert not hasattr(block, "rbr_dense"), "Dense branch should be removed"
    assert not hasattr(block, "rbr_1x1"), "1x1 branch should be removed"

    # 3. Test Forward Pass (Deploy Mode)
    print("Running forward pass (Deploy Mode)...")
    with torch.no_grad():
        output_deploy = model((dummy_img, dummy_meta))

    assert output_deploy.shape == (
        batch_size,
        1,
    ), "Output shape mismatch in deploy mode"

    print("Model verification successful.")


def verify_training_pipeline(train_dataset):
    """
    Verifies the Trainer class and the training loop.
    """
    print("\n--- Verifying Training Pipeline ---")

    # Create Loaders
    # We use the same dataset for train and val to save time in this demo
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for small demo
    )
    val_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = MetadataFusedRepVGG(num_classes=1, deploy=False).to(Config.DEVICE)

    trainer = Trainer(model, train_loader, val_loader, Config)

    print(f"Starting training for {Config.EPOCHS} epochs...")
    trainer.fit()

    # Check if artifacts exist
    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not created"
    assert os.path.exists(Config.FINAL_SWA_MODEL_PATH), "SWA model file was not created"

    print("Training pipeline verification successful.")


def verify_inference_pipeline():
    """
    Verifies the inference script logic.
    """
    print("\n--- Verifying Inference Pipeline ---")

    # run_inference relies on Config paths which we patched in setup_demo_config
    # It will load test metadata, load the model we just trained, and produce a submission

    try:
        run_inference()
    except Exception as e:
        print(f"Inference failed with error: {e}")
        raise e

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Rows: {len(df)}")

    assert (
        "id" in df.columns and "has_cactus" in df.columns
    ), "Submission columns missing"
    assert len(df) > 0, "Submission file is empty"

    # Verify values are probabilities
    assert (
        df["has_cactus"].min() >= 0.0 and df["has_cactus"].max() <= 1.0
    ), "Predictions out of range [0, 1]"

    print("Inference pipeline verification successful.")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # 1. Setup
    setup_demo_config()

    # 2. Dataset
    train_ds = verify_dataset_logic()

    # 3. Model
    verify_model_logic()

    # 4. Training
    verify_training_pipeline(train_ds)

    # 5. Inference
    verify_inference_pipeline()

    print("\nALL CHECKS PASSED.")
