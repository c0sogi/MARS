import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.dicom_processing import generate_dual_view_tri_slabs
from library.dataset import LungDataset
from library.model import GranularTabularNetwork
from library.loss import RobustLaplaceLogLikelihoodLoss
from library.training import train_one_epoch, validate_one_epoch
from library.inference import predict


def run_demo():
    print("Starting Library Demo...")

    # ==========================================
    # 1. Setup & Configuration Override
    # ==========================================
    print("\n[1] Configuring environment for rapid demonstration...")

    # Set seeds for reproducibility
    seed_everything(42)

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 10
    Config.N_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Use working directory for demo outputs to avoid permission issues
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Working directory set to: {Config.WORKING_DIR}")

    # ==========================================
    # 2. DICOM Processing Verification
    # ==========================================
    print("\n[2] Verifying DICOM Processing...")

    # Load metadata to find a valid patient
    train_meta = pd.read_csv(Config.TRAIN_CSV)
    sample_patient = train_meta.iloc[0]["Patient"]
    sample_dicom_rel_path = train_meta.iloc[0]["dicom_dir"]
    full_dicom_path = os.path.join(Config.INPUT_ROOT, sample_dicom_rel_path)

    print(f"Processing patient: {sample_patient}")

    # Generate Tri-Slabs
    axial, coronal = generate_dual_view_tri_slabs(
        sample_patient, full_dicom_path, load_cached_data=False  # Force generation
    )

    # Verify shapes and types
    print(f"Axial Shape: {axial.shape}, Coronal Shape: {coronal.shape}")

    assert axial.shape == (224, 224, 3), f"Expected (224, 224, 3), got {axial.shape}"
    assert coronal.shape == (
        224,
        224,
        3,
    ), f"Expected (224, 224, 3), got {coronal.shape}"
    assert axial.dtype == np.uint8, "Expected uint8 data type"

    # Check cache creation
    cache_file = os.path.join(Config.CACHE_DIR, f"{sample_patient}_axial.npy")
    assert os.path.exists(cache_file), "Cache file was not created."
    print("DICOM processing and caching successful.")

    # ==========================================
    # 3. Dataset Verification
    # ==========================================
    print("\n[3] Verifying LungDataset...")

    train_dataset = LungDataset(mode="train")

    # Get a single sample
    sample = train_dataset[0]

    required_keys = [
        "axial",
        "coronal",
        "age",
        "sex",
        "smoke",
        "percent",
        "priors",
        "time_delta",
        "target",
        "patient_week",
    ]

    for key in required_keys:
        assert key in sample, f"Missing key in dataset sample: {key}"

    # Check tensor shapes (C, H, W) for images
    assert sample["axial"].shape == (3, 224, 224), "Incorrect axial tensor shape"
    assert sample["coronal"].shape == (3, 224, 224), "Incorrect coronal tensor shape"
    assert isinstance(sample["time_delta"], torch.Tensor), "time_delta must be a tensor"
    assert isinstance(sample["target"], torch.Tensor), "target must be a tensor"

    print("Dataset sample structure verified.")

    # ==========================================
    # 4. Model Architecture Verification
    # ==========================================
    print("\n[4] Verifying GranularTabularNetwork...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GranularTabularNetwork().to(device)

    # Create a batch of size 2
    collate_fn = torch.utils.data.default_collate
    batch_list = [train_dataset[0], train_dataset[1]]
    batch = collate_fn(batch_list)

    # Move to device
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }

    # Forward pass
    fvc_pred, conf_pred = model(
        inputs["axial"],
        inputs["coronal"],
        inputs["age"],
        inputs["sex"],
        inputs["smoke"],
        inputs["percent"],
        inputs["priors"],
        inputs["time_delta"],
    )

    print(f"Prediction FVC: {fvc_pred.detach().cpu().numpy()}")
    print(f"Prediction Conf: {conf_pred.detach().cpu().numpy()}")

    # Assertions
    assert fvc_pred.shape == (2,), "Output FVC shape mismatch"
    assert conf_pred.shape == (2,), "Output Confidence shape mismatch"
    assert torch.all(conf_pred > 0), "Confidence must be positive (Softplus)"

    print("Model forward pass successful.")

    # ==========================================
    # 5. Loss Function Verification
    # ==========================================
    print("\n[5] Verifying RobustLaplaceLogLikelihoodLoss...")

    loss_fn = RobustLaplaceLogLikelihoodLoss()
    target = inputs["target"]

    loss = loss_fn(fvc_pred, conf_pred, target)
    print(f"Calculated Loss: {loss.item()}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.dim() == 0, "Loss should be a scalar"

    # Verify metric calculation utility
    metric_score = calculate_metric(target, fvc_pred, conf_pred)
    print(f"Metric Score: {metric_score}")
    assert isinstance(metric_score, float), "Metric should return a float"

    # ==========================================
    # 6. Training Loop Simulation
    # ==========================================
    print("\n[6] Simulating Training Loop...")

    # Create a small subset for training simulation
    indices = list(range(Config.DEBUG_SAMPLES))
    subset_train = Subset(train_dataset, indices)

    loader = DataLoader(
        subset_train, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Train one epoch
    train_loss, train_score = train_one_epoch(model, loader, optimizer, loss_fn, device)

    print(f"Epoch Train Loss: {train_loss:.4f}, Score: {train_score:.4f}")

    # Validate one epoch (using same loader for demo)
    val_loss, val_score = validate_one_epoch(model, loader, loss_fn, device)
    print(f"Epoch Val Loss: {val_loss:.4f}, Score: {val_score:.4f}")

    # Save model for inference step
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print(f"Model saved to {Config.BEST_MODEL_PATH}")

    # ==========================================
    # 7. Inference Pipeline Verification
    # ==========================================
    print("\n[7] Verifying Inference Pipeline...")

    # Ensure test dataset metadata exists (dataset.py loads metadata/test.csv)
    # We will run prediction using the saved model
    predict(
        model_path=Config.BEST_MODEL_PATH,
        output_path=Config.SUBMISSION_FILE,
        batch_size=2,
        num_workers=0,
        device=device.type,
    )

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not found"

    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print("Submission Head:")
    print(sub_df.head())

    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Columns mismatch. Expected {expected_cols}"
    assert len(sub_df) > 0, "Submission file is empty"

    print("Inference successful.")

    print("\n" + "=" * 40)
    print("ALL CHECKS PASSED SUCCESSFULLY")
    print("=" * 40)


if __name__ == "__main__":
    run_demo()
