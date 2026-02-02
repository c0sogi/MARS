import os
import sys
import numpy as np
import pandas as pd
import torch
import glob
import shutil

# Import the provided library modules
from library import config, utils, data, model, train


def main():
    print(">>> Starting Task Demonstration...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Define a working directory for this demo to keep artifacts isolated
    DEMO_DIR = os.path.join(config.WORKING_DIR, "demo_execution")
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override config parameters for speed and isolation
    print(f"Setting up configuration in {DEMO_DIR}...")
    config.IDEA_DIR = DEMO_DIR
    config.CACHE_DIR = DEMO_DIR
    config.EPOCHS = 2  # Run only 2 epochs
    config.BATCH_SIZE = 4  # Small batch size
    config.DEBUG_SAMPLE_SIZE = 10  # Only use 10 samples for debug/train
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set seed for reproducibility
    utils.set_seed(42)

    # ==========================================
    # 2. Demonstrate Library: utils.py
    # ==========================================
    print("\n[1/5] Testing library.utils...")

    # Locate a sample file using metadata
    train_meta_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    sample_row = train_meta_df.iloc[0]
    # Construct full path to FLAIR directory
    flair_dir = os.path.join(config.INPUT_DIR, sample_row["flair_path"])

    # Find the first .dcm file
    dcm_files = glob.glob(os.path.join(flair_dir, "*.dcm"))

    if dcm_files:
        sample_path = dcm_files[0]
        # Test loading
        img_arr = utils.load_dicom_as_array(sample_path, size=(224, 224))

        # Verify shape and type
        print(f"  Loaded Image Shape: {img_arr.shape}")
        print(f"  Loaded Image Dtype: {img_arr.dtype}")
        assert img_arr.shape == (224, 224), "Utils: Image resizing failed."
        assert img_arr.dtype == np.float32, "Utils: Image dtype mismatch."

        # Test scaling
        scaled_arr = utils.min_max_scale(img_arr)
        print(f"  Scaled Range: [{scaled_arr.min()}, {scaled_arr.max()}]")
        assert (
            scaled_arr.min() >= 0.0 and scaled_arr.max() <= 1.0
        ), "Utils: Scaling failed."
    else:
        print("  Warning: No DICOM files found to test utils (unexpected).")

    # ==========================================
    # 3. Demonstrate Library: data.py
    # ==========================================
    print("\n[2/5] Testing library.data...")

    # 3a. Prepare mini test metadata for speed
    # The provided get_test_dataset reads the full CSV. We create a mini version.
    full_test_df = pd.read_csv(config.TEST_METADATA_PATH)
    mini_test_path = os.path.join(config.WORKING_DIR, "mini_test.csv")
    full_test_df.head(5).to_csv(mini_test_path, index=False)
    # Point config to this mini file
    config.TEST_METADATA_PATH = mini_test_path

    # 3b. Load Train/Val Datasets (Debug Mode)
    # debug=True causes the function to slice the dataframe to config.DEBUG_SAMPLE_SIZE
    train_ds, val_ds = data.get_train_val_datasets(load_cached_data=False, debug=True)

    print(f"  Train Dataset Length: {len(train_ds)}")
    print(f"  Val Dataset Length: {len(val_ds)}")

    # Verify Dataset Item
    sample_img, sample_label = train_ds[0]
    # Expected tensor shape: (9, 224, 224)
    print(f"  Sample Tensor Shape: {sample_img.shape}")
    print(f"  Sample Label: {sample_label}")

    assert isinstance(sample_img, torch.Tensor), "Data: Output is not a tensor."
    assert sample_img.shape == (
        9,
        224,
        224,
    ), f"Data: Incorrect tensor shape {sample_img.shape}."

    # 3c. Verify DataLoaders
    train_loader, val_loader = data.get_dataloaders(
        train_ds, val_ds, batch_size=config.BATCH_SIZE
    )
    batch_imgs, batch_lbls = next(iter(train_loader))

    print(f"  Batch Images Shape: {batch_imgs.shape}")
    print(f"  Batch Labels Shape: {batch_lbls.shape}")

    assert batch_imgs.shape == (
        config.BATCH_SIZE,
        9,
        224,
        224,
    ), "Data: DataLoader batch shape mismatch."

    # ==========================================
    # 4. Demonstrate Library: model.py
    # ==========================================
    print("\n[3/5] Testing library.model...")

    # Instantiate model
    net = model.WIVENet()
    device = config.DEVICE
    net.to(device)

    # Create dummy input matching batch shape
    dummy_input = torch.randn(2, 9, 224, 224).to(device)

    # Forward pass
    with torch.no_grad():
        output = net(dummy_input)

    print(f"  Model Output Shape: {output.shape}")
    assert output.shape == (2, 1), "Model: Output shape mismatch (expected Bx1)."

    # ==========================================
    # 5. Demonstrate Library: train.py
    # ==========================================
    print("\n[4/5] Testing library.train...")

    # Run training loop (Debug mode uses the small datasets loaded earlier)
    # This will save 'best_model.pth' in config.IDEA_DIR (which we set to DEMO_DIR)
    best_auc = train.run_training(debug=True, load_cached_data=False)

    print(f"  Training completed. Best AUC: {best_auc}")

    expected_model_path = os.path.join(DEMO_DIR, "best_model.pth")
    assert os.path.exists(expected_model_path), "Train: Best model file was not saved."

    # ==========================================
    # 6. Inference & Submission Generation
    # ==========================================
    print("\n[5/5] Demonstrating Inference...")

    # Load Test Dataset (uses the mini_test.csv we created)
    test_ds = data.get_test_dataset(load_cached_data=False)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False
    )

    # Load the trained model
    net.eval()
    net.load_state_dict(torch.load(expected_model_path, map_location=device))

    predictions = []
    ids = []

    print("  Running inference on test set...")
    with torch.no_grad():
        for inputs, subject_ids in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            logits = net(inputs)
            probs = torch.sigmoid(logits)

            predictions.extend(probs.cpu().numpy().flatten())
            ids.extend(subject_ids.numpy())

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

    print("  Inference complete.")
    print("  Sample Submission:")
    print(submission_df.head())

    # Save submission (optional, but good practice to show)
    sub_path = os.path.join(DEMO_DIR, "demo_submission.csv")
    submission_df.to_csv(sub_path, index=False)
    print(f"  Submission saved to {sub_path}")

    print("\n>>> Demonstration Successfully Completed.")


if __name__ == "__main__":
    main()
