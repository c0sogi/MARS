import os
import shutil
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, compute_metrics
from library.dataset import prepare_data, UWMadisonDataset, get_transforms
from library.model import SegFormer
from library.loss import ComboLoss
from library.trainer import Trainer
from library.inference import run_inference


def main():
    # Set seed for reproducibility
    set_seed(42)

    print("=== UW-Madison GI Tract Segmentation Pipeline Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Overrides
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Define a separate working directory for this demo to avoid clutter
    Config.WORKING_DIR = "./working/demo"
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Override Config parameters for speed
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Use 0 workers to avoid multiprocessing overhead in demo

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Epochs: {Config.EPOCHS}, Batch Size: {Config.BATCH_SIZE}")

    # ---------------------------------------------------------
    # 2. Dataset & Transforms Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Dataset and Transforms...")

    # Load metadata (prepare_data handles caching)
    # We force load_cached_data=False to ensure we test the processing logic
    df_train = prepare_data(
        Config.TRAIN_METADATA_PATH, mode="train", load_cached_data=False
    )

    # Create a small subset for verification
    df_subset = df_train.head(10).copy()

    # Instantiate Dataset
    dataset = UWMadisonDataset(
        df_subset, transforms=get_transforms("train"), mode="train"
    )

    # Retrieve one sample
    sample = dataset[0]
    image = sample["image"]
    mask = sample["mask"]

    # Verification
    print(f"Sample Image Shape: {image.shape}")
    print(f"Sample Mask Shape: {mask.shape}")

    # Assertions
    assert isinstance(image, torch.Tensor), "Image should be a torch.Tensor"
    assert isinstance(mask, torch.Tensor), "Mask should be a torch.Tensor"
    # Expected shape: (Channels, Height, Width)
    assert image.shape == (Config.IN_CHANNELS, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    assert mask.shape == (Config.NUM_CLASSES, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    print("Dataset verification passed.")

    # ---------------------------------------------------------
    # 3. Model & Loss Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Model and Loss...")

    # Instantiate Model (Disable pretrained weights download for speed/safety in demo)
    model = SegFormer(pretrained=False).to(Config.DEVICE)
    model.eval()

    # Create dummy input: (Batch, Channels, Height, Width)
    dummy_input = torch.randn(
        2, Config.IN_CHANNELS, Config.IMAGE_SIZE, Config.IMAGE_SIZE
    ).to(Config.DEVICE)

    # Forward Pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, Config.NUM_CLASSES, Config.IMAGE_SIZE, Config.IMAGE_SIZE)

    # Loss Calculation
    criterion = ComboLoss()
    # Dummy targets: (Batch, Classes, Height, Width)
    dummy_target = (
        torch.randint(
            0, 2, (2, Config.NUM_CLASSES, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
        )
        .float()
        .to(Config.DEVICE)
    )

    loss = criterion(output, dummy_target)
    print(f"Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss should not be NaN"
    print("Model and Loss verification passed.")

    # ---------------------------------------------------------
    # 4. Training Loop (Trainer)
    # ---------------------------------------------------------
    print("\n[4] Running Training Loop (Debug Mode)...")

    # Initialize Trainer
    # Note: Trainer init creates a model with pretrained=True by default.
    # Assuming environment has internet or cached weights.
    trainer = Trainer(config=Config)

    # Run fit with debug=True. This subsamples the dataset significantly.
    trainer.fit(epochs=Config.EPOCHS, debug=True)

    # Verify model checkpoint was saved
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    print(f"Training complete. Model saved to {Config.MODEL_SAVE_PATH}")

    # ---------------------------------------------------------
    # 5. Inference Pipeline
    # ---------------------------------------------------------
    print("\n[5] Running Inference Pipeline...")

    # To make inference fast, we create a temporary test metadata file with only one case.
    full_test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if not full_test_df.empty:
        # Select the first case
        first_case_id = full_test_df.iloc[0]["case"]
        subset_test_df = full_test_df[full_test_df["case"] == first_case_id].copy()

        # Save to temp file
        temp_test_meta_path = os.path.join(Config.WORKING_DIR, "temp_test_metadata.csv")
        subset_test_df.to_csv(temp_test_meta_path, index=False)

        # Update Config to point to this temp file
        Config.TEST_METADATA_PATH = temp_test_meta_path
        print(f"Running inference on subset of {len(subset_test_df)} slices...")

        # Run Inference
        run_inference(model_path=Config.MODEL_SAVE_PATH)

        # Verify submission file
        assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not generated."
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission generated with {len(sub_df)} rows.")
        print(sub_df.head())
    else:
        print("Test metadata is empty, skipping inference run.")

    # ---------------------------------------------------------
    # 6. Metrics Verification
    # ---------------------------------------------------------
    print("\n[6] Verifying Metrics Calculation...")

    # Create synthetic 3D volumes (Depth, Height, Width)
    # Case 1: Perfect match
    vol_gt = np.zeros((10, 50, 50))
    vol_gt[2:8, 10:40, 10:40] = 1
    vol_pred_perfect = vol_gt.copy()

    metrics_perfect = compute_metrics(vol_pred_perfect, vol_gt)
    print(f"Perfect Match Metrics: {metrics_perfect}")
    assert metrics_perfect["dice"] == 1.0
    assert metrics_perfect["hausdorff"] == 0.0

    # Case 2: Partial overlap
    vol_pred_partial = np.zeros((10, 50, 50))
    vol_pred_partial[2:8, 20:50, 20:50] = 1  # Shifted

    metrics_partial = compute_metrics(vol_pred_partial, vol_gt)
    print(f"Partial Match Metrics: {metrics_partial}")
    assert 0.0 < metrics_partial["dice"] < 1.0
    assert metrics_partial["hausdorff"] > 0.0

    print("Metrics verification passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
