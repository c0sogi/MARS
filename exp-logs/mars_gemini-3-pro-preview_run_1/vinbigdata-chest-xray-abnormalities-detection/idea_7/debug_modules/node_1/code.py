import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, calculate_map
from library.dataset import ThoracicDataset
from library.model import EfficientNetBiFPN
from library.loss import CenterNetLoss
from library.engine import train_one_epoch, validate
from library.inference import run_inference


def run_demo():
    print("--- Starting Library Usage Demonstration ---")

    # 1. Setup & Configuration Overrides
    seed_everything(42)

    # Define temporary paths
    demo_dir = "./working/demo_run"
    os.makedirs(demo_dir, exist_ok=True)

    demo_train_path = os.path.join(demo_dir, "train_subset.csv")
    demo_val_path = os.path.join(demo_dir, "val_subset.csv")

    # Override Config for speed
    print("Configuring environment for demo...")
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.CHECKPOINT_DIR = os.path.join(demo_dir, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.TRAIN_META_PATH = demo_train_path
    Config.VAL_META_PATH = demo_val_path

    # Hyperparameters for demo
    Config.IMG_SIZE = 512  # Slightly smaller for speed
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 2

    # Re-run setup to create new directories
    Config.setup()

    # 2. Create Data Subsets
    print("Creating data subsets...")
    full_train_df = pd.read_csv("./metadata/train_meta.csv")

    # Select 20 unique images
    unique_imgs = full_train_df["image_id"].unique()
    if len(unique_imgs) > 20:
        selected_imgs = np.random.choice(unique_imgs, 20, replace=False)
    else:
        selected_imgs = unique_imgs

    subset_df = full_train_df[full_train_df["image_id"].isin(selected_imgs)].copy()

    # Split 16 train, 4 val
    train_imgs = selected_imgs[:16]
    val_imgs = selected_imgs[16:]

    train_subset = subset_df[subset_df["image_id"].isin(train_imgs)]
    val_subset = subset_df[subset_df["image_id"].isin(val_imgs)]

    train_subset.to_csv(demo_train_path, index=False)
    val_subset.to_csv(demo_val_path, index=False)

    print(f"Train subset: {len(train_subset)} rows, {len(train_imgs)} images")
    print(f"Val subset: {len(val_subset)} rows, {len(val_imgs)} images")

    # 3. Verify Dataset
    print("\n--- Verifying Dataset ---")
    # Initialize dataset
    train_ds = ThoracicDataset(split="train", load_cached_data=False)

    # Check length
    assert len(train_ds) == len(
        train_imgs
    ), f"Dataset length mismatch: {len(train_ds)} vs {len(train_imgs)}"

    # Check item structure
    sample = train_ds[0]
    img = sample["image"]
    target = sample["target"]

    print(f"Image Shape: {img.shape}")
    print(f"Heatmap Shape: {target['heatmap'].shape}")

    # Assertions
    assert img.shape == (3, Config.IMG_SIZE, Config.IMG_SIZE), "Incorrect image shape"
    assert target["heatmap"].shape == (
        Config.NUM_CLASSES,
        Config.IMG_SIZE // 4,
        Config.IMG_SIZE // 4,
    ), "Incorrect heatmap shape"
    assert target["wh"].shape == (
        2,
        Config.IMG_SIZE // 4,
        Config.IMG_SIZE // 4,
    ), "Incorrect wh shape"
    assert target["global_label"].shape == (), "Global label should be scalar"

    # 4. Verify Model
    print("\n--- Verifying Model ---")
    device = Config.DEVICE
    model = EfficientNetBiFPN().to(device)

    # Create dummy batch
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)

    # Forward pass
    outputs = model(dummy_input)

    print("Model forward pass successful.")
    print(f"Output Heatmap: {outputs['heatmap'].shape}")

    # Assertions
    assert outputs["heatmap"].shape == (
        2,
        Config.NUM_CLASSES,
        Config.IMG_SIZE // 4,
        Config.IMG_SIZE // 4,
    )
    assert outputs["wh"].shape == (2, 2, Config.IMG_SIZE // 4, Config.IMG_SIZE // 4)
    assert outputs["global_logits"].shape == (2, 1)

    # 5. Verify Loss
    print("\n--- Verifying Loss ---")
    criterion = CenterNetLoss()

    # Prepare dummy targets on device
    dummy_targets = {
        "heatmap": torch.zeros(
            2, Config.NUM_CLASSES, Config.IMG_SIZE // 4, Config.IMG_SIZE // 4
        ).to(device),
        "wh": torch.zeros(2, 2, Config.IMG_SIZE // 4, Config.IMG_SIZE // 4).to(device),
        "offset": torch.zeros(2, 2, Config.IMG_SIZE // 4, Config.IMG_SIZE // 4).to(
            device
        ),
        "reg_mask": torch.zeros(2, Config.IMG_SIZE // 4, Config.IMG_SIZE // 4).to(
            device
        ),
        "global_label": torch.zeros(2).to(device),
    }

    loss, loss_stats = criterion(outputs, dummy_targets)
    print(f"Calculated Loss: {loss.item()}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive (or close to zero)"

    # 6. Verify Training Loop
    print("\n--- Verifying Training Loop (1 Epoch) ---")
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead in demo
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Run one epoch
    epoch_loss = train_one_epoch(model, train_loader, optimizer, device, epoch=1)
    print(f"Epoch 1 Loss: {epoch_loss}")

    # Save a checkpoint for inference test
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    torch.save(model.state_dict(), ckpt_path)
    print("Saved checkpoint for inference test.")

    # 7. Verify Validation & mAP
    print("\n--- Verifying Validation ---")
    val_ds = ThoracicDataset(split="val", load_cached_data=False)
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # We need the ground truth dataframe for mAP calculation
    gt_df = val_ds.df

    val_loss, val_map = validate(model, val_loader, device, gt_df)
    print(f"Validation mAP: {val_map}")

    # 8. Verify Inference
    print("\n--- Verifying Inference ---")
    # We will run inference on a tiny subset of the test set (5 images)
    # run_inference handles loading the model from the checkpoint we just saved
    try:
        run_inference(
            checkpoint_path=ckpt_path, subset_size=5, batch_size=2, device=device
        )

        # Check if submission file exists
        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        if os.path.exists(sub_path):
            sub_df = pd.read_csv(sub_path)
            print(f"Submission generated with {len(sub_df)} rows.")
            print(sub_df.head())
        else:
            raise FileNotFoundError("Submission file was not generated.")

    except Exception as e:
        print(f"Inference failed: {e}")
        raise e

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
