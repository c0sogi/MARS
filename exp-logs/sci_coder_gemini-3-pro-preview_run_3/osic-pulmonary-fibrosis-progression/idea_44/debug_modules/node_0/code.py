import os
import sys
import numpy as np
import pandas as pd
import torch
import math

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.data import (
    CTPreprocessor,
    ClinicalPreprocessor,
    LungDataset,
    get_dataloaders,
)
from library.model import GPCRNet
from library.train import LaplaceLoss, Runner


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    print(">>> Setting up environment...")
    Config.setup()
    seed_everything(Config.SEED)

    # Create a temporary directory for our subset data
    temp_dir = os.path.join(Config.WORKING_DIR, "demo_test")
    os.makedirs(temp_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Metric Verification
    # -------------------------------------------------------------------------
    print(">>> Verifying Metric Logic...")
    # Case: Perfect prediction with confidence 100
    # Delta = 0, Sigma = 100 -> Sigma_clipped = 100
    # Metric = - (sqrt(2)*0)/100 - ln(sqrt(2)*100) = -ln(141.42) approx -4.95
    y_true = np.array([2000.0])
    y_pred = np.array([2000.0])
    sigma = np.array([100.0])

    score = laplace_log_likelihood_metric(y_true, y_pred, sigma)
    expected_score = -np.log(np.sqrt(2) * 100)

    assert np.isclose(
        score, expected_score, atol=1e-5
    ), f"Metric calculation mismatch. Got {score}, expected {expected_score}"
    print("    Metric verification passed.")

    # -------------------------------------------------------------------------
    # 3. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print(">>> Verifying Data Pipeline...")

    # Load original metadata to create a small subset
    full_train_df = pd.read_csv(Config.TRAIN_CSV)

    # Create a subset of 10 rows for training and 4 for validation
    # This ensures speed while testing the pipeline
    subset_train = full_train_df.iloc[:10].copy()
    subset_val = full_train_df.iloc[10:14].copy()

    temp_train_path = os.path.join(temp_dir, "train_subset.csv")
    temp_val_path = os.path.join(temp_dir, "val_subset.csv")

    subset_train.to_csv(temp_train_path, index=False)
    subset_val.to_csv(temp_val_path, index=False)

    # Patch Config to use these subsets
    Config.TRAIN_CSV = temp_train_path
    Config.VAL_CSV = temp_val_path
    Config.BATCH_SIZE = 2  # Small batch size for the subset
    Config.EPOCHS = 1  # Only 1 epoch for demonstration
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny dataset

    # Test CTPreprocessor
    print("    Testing CTPreprocessor...")
    img_processor = CTPreprocessor()
    sample_patient = subset_train.iloc[0]["Patient"]
    sample_path = subset_train.iloc[0]["image_path"]

    # Process one patient
    img_tensor = img_processor.process_patient(
        sample_patient, sample_path, load_cached_data=False
    )

    # Verify Image Shape: (3, 260, 260)
    assert img_tensor.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {img_tensor.shape}"
    # Verify Normalization: [0, 1]
    assert (
        img_tensor.min() >= 0.0 and img_tensor.max() <= 1.0
    ), "Image data not normalized to [0, 1]"
    print("    CTPreprocessor passed.")

    # Test ClinicalPreprocessor
    print("    Testing ClinicalPreprocessor...")
    clin_processor = ClinicalPreprocessor()
    processed_df = clin_processor.preprocess(subset_train, is_train=True)

    required_cols = [
        "Baseline_FVC_Scaled",
        "Relative_Time",
        "Age_Scaled",
        "Sex_Code",
        "Smoking_Code",
    ]
    for col in required_cols:
        assert col in processed_df.columns, f"Missing preprocessed column: {col}"
    print("    ClinicalPreprocessor passed.")

    # Test DataLoader
    print("    Testing DataLoader...")
    train_loader, val_loader = get_dataloaders(
        train_csv_path=Config.TRAIN_CSV,
        val_csv_path=Config.VAL_CSV,
        batch_size=Config.BATCH_SIZE,
    )

    # Fetch one batch
    images, clinical, targets = next(iter(train_loader))

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Batch image shape incorrect: {images.shape}"
    assert clinical.shape == (
        Config.BATCH_SIZE,
        Config.N_CLINICAL_FEATURES,
    ), f"Batch clinical shape incorrect: {clinical.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
    ), f"Batch target shape incorrect: {targets.shape}"
    print("    DataLoader passed.")

    # -------------------------------------------------------------------------
    # 4. Model Verification
    # -------------------------------------------------------------------------
    print(">>> Verifying Model Architecture...")
    device = "cpu"  # Use CPU for simple logic verification
    model = GPCRNet().to(device)
    model.eval()

    with torch.no_grad():
        # Forward pass with the batch fetched earlier
        (mu, sigma), (aux_mu, aux_sigma) = model(images.to(device), clinical.to(device))

    # Check shapes
    assert mu.shape == (Config.BATCH_SIZE,), "Main output Mu shape incorrect"
    assert sigma.shape == (Config.BATCH_SIZE,), "Main output Sigma shape incorrect"
    assert aux_mu.shape == (Config.BATCH_SIZE,), "Aux output Mu shape incorrect"

    # Check Sigma Positivity (Softplus constraint)
    assert torch.all(sigma > 0), "Main Sigma must be positive"
    assert torch.all(aux_sigma > 0), "Aux Sigma must be positive"
    print("    GPCRNet architecture verification passed.")

    # -------------------------------------------------------------------------
    # 5. Loss Verification
    # -------------------------------------------------------------------------
    print(">>> Verifying Loss Function...")
    criterion = LaplaceLoss()
    outputs = ((mu, sigma), (aux_mu, aux_sigma))
    loss = criterion(outputs, targets.to(device))

    assert torch.isfinite(loss), "Loss is not finite (NaN or Inf)"
    assert loss.ndim == 0, "Loss should be a scalar"
    print(f"    LaplaceLoss calculation passed. Loss: {loss.item():.4f}")

    # -------------------------------------------------------------------------
    # 6. Training Loop Integration (Runner)
    # -------------------------------------------------------------------------
    print(">>> Verifying Training Loop (Runner)...")

    # Initialize Runner (will use the patched Config paths)
    runner = Runner(device=Config.DEVICE)

    # Ensure checkpoint directory is clean or exists
    if os.path.exists(Config.BEST_MODEL_PATH):
        os.remove(Config.BEST_MODEL_PATH)

    # Run training (1 Epoch, tiny subset)
    runner.train()

    # Verify artifact creation
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not created."
    print("    Training loop execution passed.")

    print("\n>>> All verification steps completed successfully.")


if __name__ == "__main__":
    main()
