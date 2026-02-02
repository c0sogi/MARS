import os
import shutil
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library components
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.data import LungDataProcessor, LungDataset, get_transforms
from library.model import HighFidelityDualNet, predict
from library.train import run_training


def main():
    print("Initializing demonstration...")
    seed_everything(42)

    # Define paths
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    DEMO_CACHE_DIR = os.path.join(WORKING_DIR, "demo_cache")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Data Processing Demonstration
    # ---------------------------------------------------------
    print("\n[1] Testing LungDataProcessor...")

    # Load a small subset of training data for demonstration
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    subset_train_df = train_df.head(4).copy()  # Select first 4 rows

    # Initialize processor with a specific demo cache to avoid polluting main cache
    processor = LungDataProcessor(cache_dir=DEMO_CACHE_DIR)

    # Pick a patient from the subset
    sample_patient = subset_train_df.iloc[0]["Patient"]
    sample_dicom_dir = subset_train_df.iloc[0]["dicom_dir"]

    print(f"Processing images for patient: {sample_patient}")
    img_ax, img_cor = processor.get_images(
        sample_patient, sample_dicom_dir, load_cached_data=False
    )

    # Verify image shapes (Height, Width, Channels) -> (224, 224, 3)
    # The processor returns numpy arrays in HWC format
    print(f"Axial Image Shape: {img_ax.shape}")
    print(f"Coronal Image Shape: {img_cor.shape}")

    assert img_ax.shape == (224, 224, 3), f"Expected (224, 224, 3), got {img_ax.shape}"
    assert img_cor.shape == (
        224,
        224,
        3,
    ), f"Expected (224, 224, 3), got {img_cor.shape}"
    assert img_ax.dtype == np.uint8, "Expected uint8 dtype for image"

    # ---------------------------------------------------------
    # 2. Dataset and Transforms Demonstration
    # ---------------------------------------------------------
    print("\n[2] Testing LungDataset...")

    transforms = get_transforms("train")
    dataset = LungDataset(
        subset_train_df, processor, transforms=transforms, mode="train"
    )

    # Fetch one item
    sample_item = dataset[0]

    # Verify keys
    expected_keys = [
        "img_ax",
        "img_cor",
        "tab_vec",
        "rel_week",
        "baseline_fvc",
        "baseline_fvc_sc",
        "target",
        "patient_week",
    ]
    for k in expected_keys:
        assert k in sample_item, f"Missing key {k} in dataset item"

    # Verify Tensor Shapes (Channels, Height, Width) -> (3, 224, 224)
    # Transforms convert HWC to CHW
    print(f"Dataset Tensor Shape (Axial): {sample_item['img_ax'].shape}")
    assert sample_item["img_ax"].shape == (3, 224, 224)
    assert isinstance(sample_item["tab_vec"], torch.Tensor)
    assert sample_item["tab_vec"].shape == (6,)  # 6 tabular features

    # ---------------------------------------------------------
    # 3. Model Architecture Demonstration
    # ---------------------------------------------------------
    print("\n[3] Testing HighFidelityDualNet Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HighFidelityDualNet().to(device)
    model.eval()

    # Create a dummy batch (Batch Size = 2)
    b_img_ax = torch.stack([sample_item["img_ax"], sample_item["img_ax"]]).to(device)
    b_img_cor = torch.stack([sample_item["img_cor"], sample_item["img_cor"]]).to(device)
    b_tab_vec = torch.stack([sample_item["tab_vec"], sample_item["tab_vec"]]).to(device)
    b_rel_week = torch.stack([sample_item["rel_week"], sample_item["rel_week"]]).to(
        device
    )
    b_base_fvc = torch.stack(
        [sample_item["baseline_fvc"], sample_item["baseline_fvc"]]
    ).to(device)
    b_base_fvc_sc = torch.stack(
        [sample_item["baseline_fvc_sc"], sample_item["baseline_fvc_sc"]]
    ).to(device)

    with torch.no_grad():
        pred_fvc, pred_conf = model(
            b_img_ax, b_img_cor, b_tab_vec, b_rel_week, b_base_fvc, b_base_fvc_sc
        )

    print(f"Prediction FVC Shape: {pred_fvc.shape}")
    print(f"Prediction Confidence Shape: {pred_conf.shape}")

    assert pred_fvc.shape == (2, 1), "Expected output shape (Batch, 1)"
    assert pred_conf.shape == (2, 1), "Expected output shape (Batch, 1)"

    # ---------------------------------------------------------
    # 4. Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n[4] Testing Training Loop (Reduced Epochs)...")

    # Create temporary subset CSVs for training
    train_subset_path = os.path.join(WORKING_DIR, "train_subset.csv")
    val_subset_path = os.path.join(WORKING_DIR, "val_subset.csv")

    # Use 10 samples for train, 5 for val
    train_df.head(10).to_csv(train_subset_path, index=False)

    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    val_df.head(5).to_csv(val_subset_path, index=False)

    model_save_path = os.path.join(WORKING_DIR, "demo_model.pth")

    # Run training with minimal parameters
    trained_model = run_training(
        train_path=train_subset_path,
        val_path=val_subset_path,
        cache_dir=DEMO_CACHE_DIR,
        epochs=2,  # Minimal epochs
        batch_size=2,  # Minimal batch size
        lr=1e-4,
        patience=1,
        save_path=model_save_path,
    )

    assert os.path.exists(model_save_path), "Model file was not saved."
    print("Training simulation completed successfully.")

    # ---------------------------------------------------------
    # 5. Inference Demonstration
    # ---------------------------------------------------------
    print("\n[5] Testing Inference...")

    # Load test metadata subset
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))
    test_subset = test_df.head(6).copy()  # Small subset

    # Run prediction
    # Note: predict() saves to ./submission/submission.csv by default in the library code
    submission_df = predict(test_subset, model_path=model_save_path)

    print("Submission DataFrame Head:")
    print(submission_df.head())

    assert "FVC" in submission_df.columns
    assert "Confidence" in submission_df.columns
    assert len(submission_df) == 6

    # Verify file output
    submission_file = "./submission/submission.csv"
    assert os.path.exists(submission_file), "Submission file not found."

    # ---------------------------------------------------------
    # 6. Metric Verification
    # ---------------------------------------------------------
    print("\n[6] Verifying Metric Calculation...")

    # Manual Calculation
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    # delta = min(|true - pred|, 1000)
    # sigma_clipped = max(sigma, 70)

    y_true = torch.tensor([2500.0])
    y_pred = torch.tensor([2600.0])
    sigma = torch.tensor([50.0])  # Should be clipped to 70

    # Expected:
    # delta = |2500 - 2600| = 100
    # sigma_clipped = 70
    # term1 = (1.41421356 * 100) / 70 = 2.020305
    # term2 = ln(1.41421356 * 70) = ln(98.9949) = 4.59507
    # metric = -2.020305 - 4.59507 = -6.61537

    calculated_metric = laplace_log_likelihood_metric(y_true, y_pred, sigma)
    print(f"Calculated Metric: {calculated_metric.item()}")

    # Allow small float precision diff
    assert -6.62 < calculated_metric.item() < -6.61, "Metric calculation mismatch"

    print("\nAll demonstrations passed successfully.")


if __name__ == "__main__":
    main()
