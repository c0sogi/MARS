import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn as nn

# Import provided library modules
import library.config as config
import library.utils as utils
import library.dataset as dataset
import library.model as model_lib
import library.engine as engine


def run_pipeline_demo():
    print("=== Starting Diabetic Retinopathy Classification Demo ===")

    # 1. Configuration & Setup
    # Override config for speed and isolation
    config.NUM_WORKERS = 2  # Reduce worker overhead for small data
    config.NUM_EPOCHS = 1  # Train for only 1 epoch
    config.BATCH_SIZE = 4  # Small batch size for demo

    # Define working paths
    demo_dir = "./working/demo_run"
    os.makedirs(demo_dir, exist_ok=True)

    temp_train_csv = os.path.join(demo_dir, "train_subset.csv")
    temp_val_csv = os.path.join(demo_dir, "val_subset.csv")
    temp_test_csv = os.path.join(demo_dir, "test_subset.csv")

    # Redirect model save path to demo directory
    # We modify the config module directly so that utils.save_checkpoint picks it up
    original_save_path = config.MODEL_SAVE_PATH
    demo_model_path = os.path.join(demo_dir, "best_model.pth")
    config.MODEL_SAVE_PATH = demo_model_path

    # Set seeds
    utils.seed_everything(config.SEED)
    print("Configuration set and seeds fixed.")

    # 2. Prepare Data Subsets
    print("\n--- Preparing Data Subsets ---")
    # Load full metadata
    df_train = pd.read_csv(config.TRAIN_META_PATH)
    df_val = pd.read_csv(config.VAL_META_PATH)
    df_test = pd.read_csv(config.TEST_META_PATH)

    # Create tiny subsets
    # We need enough data for at least one batch
    train_subset = df_train.head(16)
    val_subset = df_val.head(8)
    test_subset = df_test.head(8)

    # Save to temp files
    train_subset.to_csv(temp_train_csv, index=False)
    val_subset.to_csv(temp_val_csv, index=False)
    test_subset.to_csv(temp_test_csv, index=False)
    print(f"Subsets saved to {demo_dir}")

    # 3. Verify Dataset Logic
    print("\n--- Verifying Dataset Logic ---")
    # Initialize dataset
    ds = dataset.RetinopathyDataset(
        csv_path=temp_train_csv,
        phase="train",
        transform=dataset.get_transforms("train"),
    )

    # Get one sample
    sample = ds[0]

    # Check keys
    required_keys = ["image", "label", "target", "id_code"]
    for k in required_keys:
        assert k in sample, f"Missing key {k} in dataset sample"

    # Check Image Dimensions
    img = sample["image"]
    expected_shape = (3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert (
        img.shape == expected_shape
    ), f"Image shape mismatch. Got {img.shape}, expected {expected_shape}"

    # Check Ordinal Label Encoding
    # If target is 2, label should be [1, 1, 0, 0] (assuming 5 classes -> 4 ordinal outputs)
    target = sample["target"].item()
    label_vec = sample["label"].numpy()

    expected_vec = np.zeros(config.NUM_ORDINAL_OUTPUTS, dtype=np.float32)
    if target > 0:
        expected_vec[:target] = 1.0

    np.testing.assert_array_equal(
        label_vec,
        expected_vec,
        err_msg=f"Ordinal encoding failed for target {target}. Got {label_vec}",
    )
    print("Dataset verification passed.")

    # 4. Initialize DataLoaders
    print("\n--- Initializing DataLoaders ---")
    train_loader, val_loader, test_loader = dataset.create_dataloaders(
        temp_train_csv, temp_val_csv, temp_test_csv, batch_size=config.BATCH_SIZE
    )

    # Verify batch loading
    batch = next(iter(train_loader))
    assert (
        batch["image"].shape[0] == config.BATCH_SIZE
    ), "DataLoader batch size mismatch"
    print("DataLoaders initialized.")

    # 5. Initialize Model
    print("\n--- Initializing Model ---")
    device = config.DEVICE
    model = model_lib.OrdinalConvNeXt(
        model_name=config.MODEL_NAME,
        pretrained=False,  # Disable download for speed/offline safety
        num_classes=config.NUM_ORDINAL_OUTPUTS,
    ).to(device)

    # Verify Forward Pass
    with torch.no_grad():
        logits = model(batch["image"].to(device))

    expected_out_shape = (config.BATCH_SIZE, config.NUM_ORDINAL_OUTPUTS)
    assert (
        logits.shape == expected_out_shape
    ), f"Model output shape mismatch. Got {logits.shape}"
    print("Model initialized and forward pass verified.")

    # 6. Run Training Loop
    print("\n--- Running Training Loop (1 Epoch) ---")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    engine.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=config.NUM_EPOCHS,
        patience=1,
    )

    # Verify Checkpoint
    assert os.path.exists(
        demo_model_path
    ), f"Model checkpoint not found at {demo_model_path}"
    print("Training complete. Checkpoint saved.")

    # 7. Run Inference
    print("\n--- Running Inference ---")
    submission_path = os.path.join(demo_dir, "submission.csv")

    engine.predict_and_submit(
        model=model, test_loader=test_loader, device=device, output_path=submission_path
    )

    # Verify Submission
    assert os.path.exists(submission_path), "Submission file not generated"
    df_sub = pd.read_csv(submission_path)

    assert (
        "id_code" in df_sub.columns and "diagnosis" in df_sub.columns
    ), "Submission columns incorrect"
    assert len(df_sub) == len(
        test_subset
    ), f"Submission length mismatch. Expected {len(test_subset)}, got {len(df_sub)}"

    print("Inference complete. Submission generated.")
    print(df_sub.head())

    # Restore config (good practice)
    config.MODEL_SAVE_PATH = original_save_path
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_pipeline_demo()
