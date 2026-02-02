import os
import torch
import numpy as np
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.model as model
import library.train as train


def run_demo():
    print("Initializing Demo...")

    # 1. Setup and Config Overrides for Speed
    # We override the configuration to ensure the demo runs quickly within the constraints.
    utils.set_seed(42)

    print("Overriding configuration for fast execution...")
    config.NUM_EPOCHS = 1  # Run only 1 epoch
    config.NUM_FOLDS = 2  # Run only 2 folds
    config.BATCH_SIZE = 4  # Small batch size
    config.WORK_DIR = "./working/demo_run"  # Separate working dir for demo
    config.CACHE_PATH = os.path.join(config.WORK_DIR, "cache", "processed_data.npz")

    # Create necessary directories
    os.makedirs(config.WORK_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.CACHE_PATH), exist_ok=True)

    # 2. Data Processing Demonstration
    print("\n--- Testing Data Processing ---")
    # Force processing from scratch to verify logic
    train_data, test_data, stats = data_loader.process_and_cache_data(
        load_cached_data=False
    )

    # Assertions to verify data integrity
    print("Verifying data structures...")
    assert "images" in train_data
    assert "angles" in train_data
    assert "labels" in train_data
    assert train_data["images"].ndim == 4  # (N, 75, 75, 3)
    assert train_data["images"].shape[1:] == (75, 75, 3)
    assert len(train_data["images"]) == len(train_data["labels"])
    assert len(train_data["images"]) == len(train_data["angles"])

    print(f"Train images shape: {train_data['images'].shape}")
    print(
        f"Stats - Min: {stats['min']}, Max: {stats['max']}, Angle Mean: {stats['angle_mean']}"
    )

    # 3. DataLoader Demonstration
    print("\n--- Testing DataLoader ---")
    # Use a small debug size to create loaders quickly
    debug_size = 20
    train_loader, val_loader, test_loader = data_loader.get_loaders(
        fold_idx=0, load_cached_data=True, debug_size=debug_size
    )

    # Fetch one batch
    images, angles, labels, ids = next(iter(train_loader))

    print(
        f"Batch shapes - Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    # Assertions for DataLoader
    assert images.shape == (config.BATCH_SIZE, 3, 75, 75)
    assert angles.shape == (config.BATCH_SIZE,)
    assert labels.shape == (config.BATCH_SIZE,)
    assert isinstance(images, torch.Tensor)
    assert isinstance(angles, torch.Tensor)

    # 4. Model Demonstration
    print("\n--- Testing Model Architecture ---")
    device = torch.device(config.DEVICE)
    net = model.InputAnchoredWideBodyNet().to(device)

    # Move batch to device
    images = images.to(device)
    angles = angles.to(device)

    # Forward pass
    outputs = net(images, angles)

    print(f"Model output shape: {outputs.shape}")

    # Assertions for Model
    assert outputs.shape == (config.BATCH_SIZE, 1)
    assert not torch.isnan(outputs).any(), "Model output contains NaNs"

    # 5. Training Loop Demonstration
    print("\n--- Testing Training Loop ---")
    # This will run the training function which includes the loop, validation, and saving
    # We use a small debug_size to make it extremely fast
    train.run_training(debug_size=32)

    # Verify artifacts were created
    fold_0_path = os.path.join(config.WORK_DIR, "model_fold_0.pth")
    assert os.path.exists(
        fold_0_path
    ), f"Model file for fold 0 not found at {fold_0_path}"
    print(f"Successfully verified training artifact: {fold_0_path}")

    # 6. Checkpointing Utility Demonstration
    print("\n--- Testing Checkpoint Utilities ---")
    dummy_state = {"model_state": net.state_dict(), "epoch": 1}
    ckpt_path = os.path.join(config.WORK_DIR, "dummy_ckpt.pth")

    # Save
    utils.save_checkpoint(dummy_state, is_best=True, filepath=ckpt_path)
    assert os.path.exists(ckpt_path)
    assert os.path.exists(ckpt_path.replace(".pth", "_best.pth"))

    # Load
    loaded_ckpt = utils.load_checkpoint(ckpt_path, net)
    assert "model_state" in loaded_ckpt
    assert loaded_ckpt["epoch"] == 1
    print("Checkpoint save/load verified.")

    # 7. Inference / Submission Generation (Mini-Example)
    print("\n--- Testing Inference on Test Set ---")
    # Load the model trained in step 5
    net.load_state_dict(torch.load(fold_0_path, map_location=device))
    net.eval()

    predictions = []
    test_ids = []

    # Run inference on a few batches of test loader
    with torch.no_grad():
        for i, (imgs, angs, t_ids) in enumerate(test_loader):
            if i >= 2:
                break  # Limit to 2 batches
            imgs = imgs.to(device)
            angs = angs.to(device)

            out = net(imgs, angs)
            probs = torch.sigmoid(out).cpu().numpy().flatten()

            predictions.extend(probs)
            test_ids.extend(t_ids)

    print(f"Generated {len(predictions)} predictions.")
    assert len(predictions) == len(test_ids)
    assert all(0.0 <= p <= 1.0 for p in predictions)

    # Save a mini submission file
    sub_path = os.path.join(config.WORK_DIR, "submission", "submission.csv")
    os.makedirs(os.path.dirname(sub_path), exist_ok=True)
    with open(sub_path, "w") as f:
        f.write("id,is_iceberg\n")
        for tid, prob in zip(test_ids, predictions):
            f.write(f"{tid},{prob:.6f}\n")

    print(f"Mini submission saved to {sub_path}")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
