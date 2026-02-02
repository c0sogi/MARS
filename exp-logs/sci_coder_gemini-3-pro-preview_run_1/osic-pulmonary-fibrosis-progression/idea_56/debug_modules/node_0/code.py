import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, score
from library.data import (
    LungDataset,
    get_transforms,
    load_dicom_volume,
    generate_tri_slab,
)
from library.model import BBSLNet
from library.train import LaplaceLoss, run_training

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== BBSL-Net Library Demonstration ===")

    # 1. Setup and Configuration
    print("\n[1] Environment Setup")
    seed_everything(Config.SEED)
    print(f"Device: {Config.DEVICE}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # Ensure working directory exists (Config.setup() does this, but verifying)
    assert os.path.exists(Config.WORKING_DIR), "Working directory was not created."

    # 2. Metric Verification
    print("\n[2] Verifying Metric Logic")
    # Test Case: Perfect prediction with fixed confidence
    # Formula: - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
    # If delta=0 and sigma=100: - ln(sqrt(2) * 100)
    y_true = [2000]
    y_pred = [2000]
    sigma = [100]

    calculated_score = score(y_true, y_pred, sigma)
    expected_score = -np.log(np.sqrt(2) * 100)

    print(f"Score (Perfect Prediction): {calculated_score:.5f}")
    print(f"Expected Score:             {expected_score:.5f}")

    assert np.isclose(
        calculated_score, expected_score, atol=1e-5
    ), "Metric calculation does not match expected formula."

    # 3. Data Pipeline Verification
    print("\n[3] Verifying Data Pipeline")

    # Load metadata to get a sample path
    train_df = pd.read_csv(Config.TRAIN_CSV)
    sample_patient = train_df.iloc[0]
    dicom_rel_path = sample_patient["dicom_dir"]
    full_dicom_path = os.path.join(Config.INPUT_ROOT, dicom_rel_path)

    print(f"Loading DICOMs from: {full_dicom_path}")

    # Test Volume Loading
    # Note: If pydicom is missing, this returns a zero-array (10, 224, 224)
    vol = load_dicom_volume(full_dicom_path)
    print(f"Loaded Volume Shape: {vol.shape}")

    # Test Tri-Slab Generation
    img_ax = generate_tri_slab(vol, view="axial")
    img_cor = generate_tri_slab(vol, view="coronal")

    print(f"Generated Axial Slab Shape: {img_ax.shape}")
    print(f"Generated Coronal Slab Shape: {img_cor.shape}")

    # Assertions for image processing
    assert img_ax.shape == (
        Config.IMG_SIZE,
        Config.IMG_SIZE,
        3,
    ), "Axial image shape incorrect"
    assert img_cor.shape == (
        Config.IMG_SIZE,
        Config.IMG_SIZE,
        3,
    ), "Coronal image shape incorrect"

    # Test Dataset Class
    print("Instantiating LungDataset (Debug Mode)...")
    dataset = LungDataset(
        csv_path=Config.TRAIN_CSV,
        mode="train",
        transform=get_transforms("train"),
        debug=True,
    )

    sample_item = dataset[0]
    print("Dataset Sample Keys:", list(sample_item.keys()))

    # Verify Tensor Shapes (Channel First for PyTorch)
    # img: (3, 224, 224)
    assert sample_item["img_ax"].shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Dataset Axial tensor shape mismatch"
    assert sample_item["img_cor"].shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Dataset Coronal tensor shape mismatch"
    # meta: (4,) -> [Percent, Age, Sex, Smoking]
    assert sample_item["meta"].shape == (4,), "Metadata vector shape mismatch"

    print("Data Pipeline Verified.")

    # 4. Model Architecture Verification
    print("\n[4] Verifying Model Architecture")
    model = BBSLNet().to(Config.DEVICE)
    model.eval()

    # Create dummy batch
    batch_size = 2
    dummy_ax = torch.randn(batch_size, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(
        Config.DEVICE
    )
    dummy_cor = torch.randn(batch_size, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(
        Config.DEVICE
    )
    dummy_meta = torch.randn(batch_size, 4).to(Config.DEVICE)

    print(f"Forward pass with batch size {batch_size}...")
    with torch.no_grad():
        outputs = model(dummy_ax, dummy_cor, dummy_meta)

    print(f"Output Shape: {outputs.shape}")

    # Assertions for Model Output
    assert outputs.shape == (batch_size, 3), "Model output must be (Batch, 3)"

    # Check that sigmas (indices 1 and 2) are positive (due to Softplus)
    sigmas = outputs[:, 1:]
    assert (sigmas >= 0).all(), "Predicted sigmas must be non-negative"

    print("Model Architecture Verified.")

    # 5. Loss Function Verification
    print("\n[5] Verifying Loss Function")
    criterion = LaplaceLoss()

    # Dummy Targets
    targets = torch.tensor([2500.0, 2000.0], device=Config.DEVICE)
    delta_weeks = torch.tensor([12.0, -5.0], device=Config.DEVICE)
    baseline_fvcs = torch.tensor([2600.0, 2100.0], device=Config.DEVICE)

    loss = criterion(outputs, targets, delta_weeks, baseline_fvcs)
    print(f"Computed Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Infinite"
    print("Loss Function Verified.")

    # 6. Full Training Loop Execution
    print("\n[6] Executing Training Loop (Debug Mode)")
    print("This will run for 2 epochs on a small subset of data.")

    # run_training(debug=True) handles the loop, optimization, and saving
    best_val_score = run_training(debug=True)

    print(f"Training Complete. Best Validation Score: {best_val_score:.4f}")

    # Verify that the model checkpoint was saved
    checkpoint_path = Config.CHECKPOINT_PATH
    if os.path.exists(checkpoint_path):
        print(f"SUCCESS: Model checkpoint found at {checkpoint_path}")
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"Checkpoint Size: {file_size:.2f} MB")
    else:
        raise FileNotFoundError(f"Model checkpoint was not saved at {checkpoint_path}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
