import os
import torch
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import DataLoader

# Import provided library modules
from library import config, utils, bbox_handler, dataset, model, loss, engine, inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # 1. Setup & Reproducibility
    utils.seed_everything(42)
    device = utils.get_device()

    # Define a working directory for demo artifacts
    demo_dir = os.path.join(config.WORKING_DIR, "demo")
    utils.ensure_directory(demo_dir)
    print(f"Device: {device}")
    print(f"Demo Directory: {demo_dir}")

    # 2. Demonstrate BBoxHandler (Context-Aware Cropping Logic)
    print("\n--- Testing BBoxHandler ---")
    # Initialize handler (loads/caches MegaDetector results)
    bbox_h = bbox_handler.BBoxHandler(load_cached_data=True)

    # Verify logic with a sample ID from the training set
    train_meta_full = pd.read_csv(config.TRAIN_METADATA_PATH)
    sample_id = train_meta_full.iloc[0]["image_id"]

    # Get expanded bounding box (x, y, w, h) in normalized coordinates
    bbox = bbox_h.get_expanded_bbox(sample_id)

    # Validation
    assert len(bbox) == 4, "BBox must have 4 coordinates"
    assert all(
        0.0 <= c <= 1.0 for c in bbox
    ), "BBox coordinates must be normalized [0, 1]"
    print(f"BBox for {sample_id}: {bbox}")
    print("BBoxHandler verified.")

    # 3. Demonstrate Dataset & DataLoader
    print("\n--- Testing Dataset & DataLoader ---")
    # Create a mini training set (16 samples) for speed
    mini_train_path = os.path.join(demo_dir, "mini_train.csv")
    train_meta_full.head(16).to_csv(mini_train_path, index=False)

    # Instantiate Dataset with the mini metadata
    # Note: We pass the bbox_handler we already initialized
    train_ds = dataset.WildCamDataset(
        metadata_path=mini_train_path, mode="train", bbox_handler=bbox_h
    )

    # Check single item retrieval
    img_tensor, label = train_ds[0]

    # Validation
    assert img_tensor.shape == (
        3,
        config.IMAGE_SIZE,
        config.IMAGE_SIZE,
    ), f"Expected image shape (3, {config.IMAGE_SIZE}, {config.IMAGE_SIZE}), got {img_tensor.shape}"
    assert isinstance(label, torch.Tensor), "Label should be a torch Tensor"

    # Create DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,  # Use 0 workers for simple demo to avoid multiprocessing overhead
    )
    print(f"Dataset size: {len(train_ds)}")
    print("Dataset & DataLoader verified.")

    # 4. Demonstrate Model Initialization
    print("\n--- Testing Model Initialization ---")
    # Initialize model (pretrained=False for speed in this demo)
    net = model.get_model(
        num_classes=config.NUM_CLASSES, pretrained=False, device=device
    )

    # Validation: Forward pass with dummy input
    dummy_input = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
    with torch.no_grad():
        outputs = net(dummy_input)

    assert outputs.shape == (
        2,
        config.NUM_CLASSES,
    ), f"Expected output shape (2, {config.NUM_CLASSES}), got {outputs.shape}"
    print("Model forward pass successful.")

    # 5. Demonstrate Loss Function
    print("\n--- Testing Focal Loss ---")
    criterion = loss.FocalLoss()
    dummy_targets = torch.tensor([0, 1], device=device)  # Dummy labels

    loss_val = criterion(outputs, dummy_targets)

    # Validation
    assert not torch.isnan(loss_val), "Loss should not be NaN"
    assert loss_val.item() > 0, "Loss should be positive"
    print(f"Calculated Loss: {loss_val.item():.4f}")

    # 6. Demonstrate Training Engine
    print("\n--- Testing Training Loop ---")
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
    save_path = os.path.join(demo_dir, "demo_best_model.pth")

    # Run training for 1 epoch using the mini loader
    # We reuse train_loader as val_loader for simplicity in this demo
    best_acc = engine.run_training(
        model=net,
        train_loader=train_loader,
        val_loader=train_loader,
        optimizer=optimizer,
        scheduler=None,
        criterion=criterion,
        num_epochs=1,
        device=device,
        save_path=save_path,
    )

    # Validation
    assert os.path.exists(save_path), "Best model checkpoint was not saved"
    print(f"Training finished. Best Acc: {best_acc}")

    # 7. Demonstrate Inference
    print("\n--- Testing Inference Pipeline ---")
    # Create a mini test set (10 samples)
    test_meta_full = pd.read_csv(config.TEST_METADATA_PATH)
    mini_test_path = os.path.join(demo_dir, "mini_test.csv")
    test_meta_full.head(10).to_csv(mini_test_path, index=False)

    output_sub_path = os.path.join(demo_dir, "submission.csv")

    # Monkey-patch config.TEST_METADATA_PATH so inference.py uses our mini test set
    original_test_path = config.TEST_METADATA_PATH
    config.TEST_METADATA_PATH = mini_test_path

    try:
        inference.run_inference(
            checkpoint_path=save_path,
            output_path=output_sub_path,
            batch_size=4,
            device=device,
        )
    finally:
        # Restore configuration
        config.TEST_METADATA_PATH = original_test_path

    # Validation
    assert os.path.exists(output_sub_path), "Submission file not created"

    sub_df = pd.read_csv(output_sub_path)
    print(f"Submission head:\n{sub_df.head()}")

    assert len(sub_df) == 10, "Submission should have 10 rows"
    assert list(sub_df.columns) == [
        "Id",
        "Predicted",
    ], f"Expected columns ['Id', 'Predicted'], got {list(sub_df.columns)}"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
