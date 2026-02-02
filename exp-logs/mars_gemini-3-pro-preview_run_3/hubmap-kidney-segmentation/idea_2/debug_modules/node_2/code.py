import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from provided library files
from library.utils import CFG, seed_everything
from library.data_processing import prepare_data
from library.dataset import HuBMAPDataset, get_transforms
from library.model import UNetPlusPlus
from library.losses import DeepSupervisionLoss
from library.trainer import Trainer
from library.inference import predict_tile


def run_demo():
    print("--- Starting HuBMAP Pipeline Demo ---")

    # 1. Setup and Configuration Override
    # We modify the global Configuration class to optimize for a quick demo run.
    seed_everything(42)

    DEMO_WORK_DIR = "./working/demo_run"
    if os.path.exists(DEMO_WORK_DIR):
        shutil.rmtree(DEMO_WORK_DIR)
    os.makedirs(DEMO_WORK_DIR, exist_ok=True)

    # Override CFG settings for speed
    CFG.cache_dir = DEMO_WORK_DIR
    CFG.img_size = 256  # Smaller tiles for faster processing
    CFG.epochs = 1  # Single epoch for demonstration
    CFG.batch_size = 2  # Small batch size
    CFG.num_workers = 0  # Avoid multiprocessing overhead in demo
    CFG.backbone = "efficientnet_b0"  # Lighter backbone

    print(
        f"Configuration: Epochs={CFG.epochs}, Tile Size={CFG.img_size}, Device={CFG.device}"
    )

    # 2. Data Preparation
    print("\n[Step 1] Preparing Data...")
    train_metadata_path = "./metadata/train.csv"

    # Load metadata and take a tiny subset (1 image) to generate tiles from
    df_train_meta = pd.read_csv(train_metadata_path)
    df_train_subset = df_train_meta.head(1).copy()

    # Generate tiles
    # We use a large overlap relative to tile size just to ensure we get some tiles
    # but for speed we keep threshold high enough.
    df_tiles = prepare_data(
        metadata_df=df_train_subset,
        tile_size=CFG.img_size,
        overlap=32,
        cache_dir=CFG.cache_dir,
        load_cached_data=False,
        split="train_demo",
    )

    # Validation: Check if tiles were generated
    if len(df_tiles) == 0:
        raise AssertionError(
            "Data preparation resulted in 0 tiles. Check input data or threshold."
        )

    print(f"Generated {len(df_tiles)} tiles from {len(df_train_subset)} image(s).")

    # Limit to a very small number of tiles for the training loop demo
    df_tiles_subset = df_tiles.head(4).copy()

    # 3. Dataset and DataLoader
    print("\n[Step 2] Initializing Dataset and DataLoader...")
    train_dataset = HuBMAPDataset(
        df=df_tiles_subset, transforms=get_transforms("train"), mode="train"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        drop_last=True,
    )

    # Validation: Fetch one batch
    batch = next(iter(train_loader))
    images, masks = batch["image"], batch["mask"]

    print(f"Batch shapes - Image: {images.shape}, Mask: {masks.shape}")

    if images.shape != (CFG.batch_size, 3, CFG.img_size, CFG.img_size):
        raise AssertionError(f"Unexpected image shape: {images.shape}")
    if masks.shape != (CFG.batch_size, 1, CFG.img_size, CFG.img_size):
        raise AssertionError(f"Unexpected mask shape: {masks.shape}")

    # 4. Model Initialization
    print("\n[Step 3] Initializing Model...")
    # pretrained=False to avoid downloading weights during the timed run
    model = UNetPlusPlus(
        backbone_name=CFG.backbone,
        in_channels=3,
        classes=CFG.num_classes,
        pretrained=False,
    )
    model.to(CFG.device)

    # Validation: Forward pass
    model.train()
    with torch.no_grad():
        outputs = model(images.to(CFG.device).float())

    # UNetPlusPlus with Deep Supervision returns a list of tensors
    if not isinstance(outputs, list):
        raise AssertionError(
            "Model in training mode should return a list (Deep Supervision)."
        )

    print(f"Forward pass successful. Output heads: {len(outputs)}")
    if outputs[0].shape != (
        CFG.batch_size,
        CFG.num_classes,
        CFG.img_size,
        CFG.img_size,
    ):
        raise AssertionError(
            f"Output shape mismatch. Expected {(CFG.batch_size, CFG.num_classes, CFG.img_size, CFG.img_size)}, got {outputs[0].shape}"
        )

    # 5. Loss Function
    print("\n[Step 4] Testing Loss Function...")
    criterion = DeepSupervisionLoss(weights=[1.0, 0.5, 0.25, 0.125])

    # Calculate loss on the dummy batch
    loss = criterion(outputs, masks.to(CFG.device).float())
    print(f"Calculated Loss: {loss.item()}")

    if torch.isnan(loss):
        raise AssertionError("Loss is NaN.")

    # 6. Training Loop
    print("\n[Step 5] Running Training Loop (1 Epoch)...")
    # We use the same loader for val to ensure it runs without needing more data
    trainer = Trainer(train_loader, train_loader, device=CFG.device)

    # Overwrite model in trainer with our non-pretrained one for consistency/speed
    trainer.model = model

    # Run fit
    trainer.fit(epochs=CFG.epochs)

    # Validation: Check if checkpoint exists
    checkpoint_path = os.path.join(CFG.cache_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise AssertionError(f"Checkpoint not found at {checkpoint_path}")
    print("Training finished and checkpoint saved.")

    # 7. Inference
    print("\n[Step 6] Demonstrating Inference on a Single Tile...")
    # Load the best model
    state_dict = torch.load(checkpoint_path, map_location=CFG.device)
    model.load_state_dict(state_dict)
    model.eval()

    # Take a sample image (numpy array HWC) from the dataset (before transform)
    # We access the dataset directly to get the raw image
    sample_data = train_dataset[0]
    # The dataset returns transformed tensors. We need to simulate the raw input for predict_tile
    # predict_tile expects a numpy array (H, W, 3)

    # Let's read a raw tile using the info from the dataframe
    row = df_tiles_subset.iloc[0]
    from library.data_processing import read_tiff, read_tiff_region

    img_path = os.path.join(CFG.input_root, row["image_path"])

    with read_tiff(img_path) as src:
        raw_tile = read_tiff_region(src, row["x"], row["y"], row["w"], row["h"])

    # Ensure 3 channels
    if raw_tile.shape[2] == 1:
        raw_tile = np.repeat(raw_tile, 3, axis=2)
    elif raw_tile.shape[2] > 3:
        raw_tile = raw_tile[:, :, :3]

    print(f"Input tile shape for inference: {raw_tile.shape}")

    # Run prediction
    prob_map = predict_tile(model, raw_tile, CFG.device)

    print(f"Prediction output shape: {prob_map.shape}")
    print(f"Prediction value range: [{prob_map.min():.4f}, {prob_map.max():.4f}]")

    if prob_map.shape != (CFG.img_size, CFG.img_size):
        raise AssertionError("Inference output shape mismatch.")

    if prob_map.min() < 0 or prob_map.max() > 1:
        raise AssertionError("Inference probabilities out of range [0, 1].")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
