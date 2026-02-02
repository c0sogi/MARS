import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import glob
import shutil

# Import provided library modules
from library import config
from library import utils
from library import data
from library import model
from library import engine


def run_demo():
    print("=== Starting RSNA-MICCAI Radiogenomics Demo ===")

    # 1. Setup Reproducibility
    utils.seed_everything(config.SEED)

    # Define temporary paths for this demo
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)

    mini_train_path = os.path.join(demo_dir, "mini_train.csv")
    mini_val_path = os.path.join(demo_dir, "mini_val.csv")

    # Override CACHE_DIR in config temporarily to keep demo isolated
    # Note: We can't easily change the imported config variable globally if it's used inside functions,
    # but load_expert_data uses config.CACHE_DIR. We will monkeypatch it for the demo.
    original_cache_dir = config.CACHE_DIR
    config.CACHE_DIR = demo_dir

    try:
        # 2. Prepare Mini Dataset (Speed Optimization)
        print("\n[Step 1] Preparing Mini Metadata...")
        full_train_df = pd.read_csv(config.TRAIN_METADATA_PATH)

        # Select a small subset: 8 for train, 4 for val
        # Ensure we pick subjects that actually exist (metadata script verified this)
        subset_df = full_train_df.head(12)
        train_df = subset_df.iloc[:8]
        val_df = subset_df.iloc[8:]

        train_df.to_csv(mini_train_path, index=False)
        val_df.to_csv(mini_val_path, index=False)

        print(f"Created mini datasets: Train ({len(train_df)}), Val ({len(val_df)})")

        # 3. Demonstrate Utility Functions
        print("\n[Step 2] Demonstrating Utility Functions...")

        # Pick a sample file path from the dataframe
        sample_row = train_df.iloc[0]
        flair_rel_path = sample_row["flair_path"]
        flair_full_path = os.path.join(config.INPUT_DIR, flair_rel_path)

        # Test: calculate_modality_com
        # This function scans the directory to find the center of mass
        print(f"Calculating Center of Mass for: {flair_full_path}")
        selected_slice_path = utils.calculate_modality_com(
            flair_full_path, offset_ratio=0.0
        )

        assert selected_slice_path is not None, "calculate_modality_com returned None"
        assert os.path.exists(selected_slice_path), "Selected slice path does not exist"
        print(f"Selected Slice: {os.path.basename(selected_slice_path)}")

        # Test: read_dicom_image
        img_array = utils.read_dicom_image(selected_slice_path)
        assert img_array is not None, "Failed to read DICOM image"
        assert isinstance(img_array, np.ndarray), "Image is not a numpy array"
        print(f"DICOM read successfully. Shape: {img_array.shape}")

        # 4. Demonstrate Data Loading
        print("\n[Step 3] Demonstrating Data Loading & Preprocessing...")

        # We use a custom split name 'demo_train' to avoid colliding with real cache
        # and to force the function to use our passed metadata path if we were modifying the loader logic,
        # but here load_expert_data takes the path explicitly.

        # Load Train Data
        train_images, train_labels, train_ids = data.load_expert_data(
            expert_name="center",
            split="demo_train",
            metadata_path=mini_train_path,
            load_cached_data=False,  # Force processing
        )

        assert len(train_images) == 8
        assert train_images.shape == (8, 224, 224, 3)  # (N, H, W, C)
        assert train_labels.shape == (8,)
        print("Mini train data loaded successfully.")

        # Load Val Data
        val_images, val_labels, val_ids = data.load_expert_data(
            expert_name="center",
            split="demo_val",
            metadata_path=mini_val_path,
            load_cached_data=False,
        )

        # Create Datasets and Loaders
        # We manually create them to bypass get_expert_dataloader's hardcoded metadata paths
        train_transform = data.get_transforms(phase="train")
        val_transform = data.get_transforms(phase="val")

        train_dataset = data.SDCDataset(
            train_images, train_labels, transform=train_transform
        )
        val_dataset = data.SDCDataset(val_images, val_labels, transform=val_transform)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=4, shuffle=True, num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=4, shuffle=False, num_workers=0
        )

        # Verify Batch
        batch_imgs, batch_lbls = next(iter(train_loader))
        assert batch_imgs.shape == (4, 3, 224, 224)  # (B, C, H, W)
        assert batch_lbls.shape == (4,)
        print("DataLoader produced valid batch.")

        # 5. Demonstrate Model
        print("\n[Step 4] Demonstrating Model Initialization...")
        # Use pretrained=False to avoid downloading weights during demo
        net = model.EfficientNetExpert(
            model_name="efficientnet_b0", pretrained=False, num_classes=1
        )
        net = net.to(config.DEVICE)

        # Forward pass check
        with torch.no_grad():
            dummy_input = batch_imgs.to(config.DEVICE)
            output = net(dummy_input)
            assert output.shape == (
                4,
                1,
            ), f"Model output shape mismatch: {output.shape}"
        print("Model initialized and forward pass successful.")

        # 6. Demonstrate Engine (Training & Eval)
        print("\n[Step 5] Demonstrating Training and Evaluation...")

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)

        # Train one epoch
        print("Running training step...")
        train_loss = engine.train_one_epoch(
            net, train_loader, criterion, optimizer, config.DEVICE
        )
        print(f"Train Loss: {train_loss:.4f}")
        assert not np.isnan(train_loss), "Training loss is NaN"

        # Evaluate
        print("Running evaluation step...")
        val_loss, val_auc, val_preds, val_targets = engine.evaluate(
            net, val_loader, criterion, config.DEVICE
        )
        print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        # Predict
        print("Running prediction step...")
        preds = engine.predict_expert(net, val_loader, config.DEVICE)
        assert len(preds) == 4, "Prediction count mismatch"
        print(f"Predictions: {preds}")

        print("\n=== Demo Completed Successfully ===")

    except Exception as e:
        print(f"\n!!! DEMO FAILED: {e}")
        raise e
    finally:
        # Restore config
        config.CACHE_DIR = original_cache_dir

        # Cleanup temporary files
        if os.path.exists(demo_dir):
            shutil.rmtree(demo_dir)
            print("Temporary demo files cleaned up.")


if __name__ == "__main__":
    run_demo()
