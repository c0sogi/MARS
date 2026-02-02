import os
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Import library components
from library.config import Config
from library.utils import natural_sort_key, load_dicom_and_process, get_study_paths
from library.dataset import RSNADataset
from library.model import FractureMILModel
from library.loss import WeightedLogLoss
from library.train import Trainer
from library.inference import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== RSNA Cervical Spine Fracture Detection: Library Demo ===\n")

    # --- 1. Configuration Setup ---
    print("[1] Configuring environment for rapid demonstration...")
    # Override Config parameters to run fast on CPU/GPU with minimal memory
    Config.IMG_SIZE = 64  # Small image size
    Config.NUM_SLICES = 8  # Few slices per bag
    Config.BATCH_SIZE = 2  # Small batch size
    Config.EPOCHS = 1  # Single epoch
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Redirect outputs to a demo directory
    Config.CACHE_DIR = "./working/demo_cache"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Re-run setup to create these directories
    Config.setup_directories()
    Config.seed_everything(Config.SEED)
    print("    Configuration updated: IMG_SIZE=64, NUM_SLICES=8, EPOCHS=1")

    # --- 2. Utility Verification ---
    print("\n[2] Verifying Utility Functions...")

    # Test natural sorting
    unsorted_files = ["10.dcm", "1.dcm", "2.dcm"]
    sorted_files = sorted(unsorted_files, key=natural_sort_key)
    assert sorted_files == ["1.dcm", "2.dcm", "10.dcm"], "Natural sort failed"
    print("    natural_sort_key: OK")

    # Test DICOM loading
    # We need a real path from the metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    sample_study_path = os.path.join(Config.INPUT_DIR, train_meta.iloc[0]["image_path"])

    # Find a dcm file
    dcm_files = get_study_paths(sample_study_path)
    if dcm_files:
        sample_dcm = dcm_files[0]
        img = load_dicom_and_process(sample_dcm, size=Config.IMG_SIZE)

        assert isinstance(img, np.ndarray), "Image is not a numpy array"
        assert img.shape == (
            Config.IMG_SIZE,
            Config.IMG_SIZE,
        ), f"Wrong shape: {img.shape}"
        assert img.min() >= 0 and img.max() <= 1.0, "Image not normalized to [0, 1]"
        assert img.dtype == np.float32, "Image is not float32"
        print(f"    load_dicom_and_process: OK (Shape: {img.shape})")
    else:
        print("    Warning: No DICOM files found in sample study for testing.")

    # --- 3. Dataset Verification ---
    print("\n[3] Verifying RSNADataset...")
    # Create a small subset dataframe
    subset_df = train_meta.head(4).copy()

    # Instantiate dataset
    dataset = RSNADataset(subset_df, Config, load_cached_paths=False)

    # Fetch one item
    images, labels = dataset[0]

    # Verify Shapes
    # Expected Image Shape: (NUM_SLICES, IN_CHANNELS, IMG_SIZE, IMG_SIZE)
    expected_img_shape = (
        Config.NUM_SLICES,
        Config.IN_CHANNELS,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    )
    # Expected Label Shape: (8,) -> 7 vertebrae + 1 overall
    expected_label_shape = (8,)

    assert (
        images.shape == expected_img_shape
    ), f"Dataset image shape mismatch. Got {images.shape}"
    assert (
        labels.shape == expected_label_shape
    ), f"Dataset label shape mismatch. Got {labels.shape}"
    assert isinstance(images, torch.Tensor), "Images should be a torch Tensor"
    print(
        f"    __getitem__: OK (Image Shape: {images.shape}, Label Shape: {labels.shape})"
    )

    # --- 4. Model Architecture Verification ---
    print("\n[4] Verifying FractureMILModel...")
    # Initialize model (pretrained=False for speed/offline safety in demo)
    model = FractureMILModel(pretrained=False)
    model.to(Config.DEVICE)
    model.eval()

    # Create dummy input batch
    # Shape: (Batch, Slices, Channels, H, W)
    dummy_input = torch.randn(
        2, Config.NUM_SLICES, Config.IN_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE
    ).to(Config.DEVICE)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        2,
        8,
    ), f"Model output shape mismatch. Expected (2, 8), got {output.shape}"
    assert (
        output.min() >= 0 and output.max() <= 1
    ), "Model output not in probability range [0, 1]"
    print(f"    Forward pass: OK (Output Shape: {output.shape})")

    # --- 5. Loss Function Verification ---
    print("\n[5] Verifying WeightedLogLoss...")
    criterion = WeightedLogLoss()

    # Dummy predictions (0.9 for true, 0.1 for false) and targets
    preds = torch.tensor(
        [[0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9]], dtype=torch.float32
    )
    targets = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32
    )

    loss = criterion(preds, targets)

    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() > 0, "Loss should be positive"
    print(f"    Loss calculation: OK (Value: {loss.item():.4f})")

    # --- 6. Training Loop Demonstration ---
    print("\n[6] Demonstrating Training Loop (Trainer)...")
    # Initialize Trainer in debug mode (uses subset of data)
    trainer = Trainer(config=Config, debug=True)

    # Run fit for 1 epoch
    print("    Starting training (1 epoch, debug mode)...")
    trainer.fit(epochs=1)

    # Verify model checkpoint exists
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not saved."
    print(f"    Training complete. Checkpoint saved to {best_model_path}")

    # --- 7. Inference Demonstration ---
    print("\n[7] Demonstrating Inference (generate_submission)...")

    # Ensure sample submission exists (it's in input, read-only)
    assert os.path.exists(
        Config.SAMPLE_SUBMISSION_PATH
    ), "Sample submission file missing."

    # Run inference
    generate_submission(config=Config, debug=True, load_cached_data=False)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "row_id" in sub_df.columns and "fractured" in sub_df.columns
    ), "Submission columns missing."
    assert len(sub_df) > 0, "Submission file is empty."
    print(f"    Inference complete. Submission generated at {Config.SUBMISSION_PATH}")
    print(f"    Submission Head:\n{sub_df.head(3)}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
