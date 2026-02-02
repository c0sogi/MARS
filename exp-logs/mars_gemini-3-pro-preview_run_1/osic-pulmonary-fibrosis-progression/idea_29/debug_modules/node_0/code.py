import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import logging

# Import provided library modules
from library.config import Config
from library.utils import calculate_metric, seed_everything
from library.preprocessing import Preprocessor
from library.dataset import LungDataset, get_dataloaders
from library.model import CVRNet
from library.loss import RobustLaplaceLoss
from library.runner import Trainer

# Setup basic logging for the script
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Demo")


def demonstrate_metric():
    logger.info("--- Demonstrating Metric Calculation ---")

    # Case 1: Perfect prediction
    # Metric = - (sqrt(2) * 0) / 70 - ln(sqrt(2) * 70) = -ln(sqrt(2)*70)
    y_true = np.array([2000.0])
    y_pred = np.array([2000.0])
    sigma = np.array([70.0])

    score = calculate_metric(y_true, y_pred, sigma)
    expected = -np.log(np.sqrt(2) * 70)

    logger.info(f"Perfect Score: {score:.4f} (Expected: {expected:.4f})")
    assert np.isclose(
        score, expected, atol=1e-4
    ), "Metric calculation for perfect prediction failed."

    # Case 2: Large Error (clipped at 1000) with low confidence (clipped at 70)
    y_true_bad = np.array([2000.0])
    y_pred_bad = np.array([4000.0])  # Error 2000 -> Clipped to 1000
    sigma_bad = np.array([10.0])  # Clipped to 70

    score_bad = calculate_metric(y_true_bad, y_pred_bad, sigma_bad)
    # Metric = - (sqrt(2) * 1000) / 70 - ln(sqrt(2) * 70)
    expected_bad = -(np.sqrt(2) * 1000) / 70 - np.log(np.sqrt(2) * 70)

    logger.info(f"Bad Score: {score_bad:.4f} (Expected: {expected_bad:.4f})")
    assert np.isclose(
        score_bad, expected_bad, atol=1e-4
    ), "Metric calculation for bad prediction failed."
    logger.info("Metric verification passed.\n")


def demonstrate_preprocessing():
    logger.info("--- Demonstrating Preprocessing ---")

    # Initialize Preprocessor
    preprocessor = Preprocessor(img_size=224)

    # Pick a sample directory from metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    sample_patient = train_df.iloc[0]["Patient"]
    sample_dir = train_df.iloc[0]["dicom_dir"]

    logger.info(f"Processing patient: {sample_patient} from {sample_dir}")

    # Run processing
    # Note: If pydicom is missing (likely in this env), this returns black images (zeros)
    axial, coronal = preprocessor.process_patient(sample_dir)

    logger.info(f"Axial Shape: {axial.shape}, Coronal Shape: {coronal.shape}")

    # Verify shapes
    assert axial.shape == (224, 224, 3), f"Incorrect Axial shape: {axial.shape}"
    assert coronal.shape == (224, 224, 3), f"Incorrect Coronal shape: {coronal.shape}"
    assert axial.dtype == np.uint8, "Image should be uint8"
    logger.info("Preprocessing verification passed.\n")


def setup_subset_data():
    logger.info("--- Setting up Subset Data for Speed ---")

    # Define paths for subset CSVs
    subset_train_path = os.path.join(Config.WORKING_ROOT, "train_subset.csv")
    subset_val_path = os.path.join(Config.WORKING_ROOT, "val_subset.csv")
    subset_test_path = os.path.join(Config.WORKING_ROOT, "test_subset.csv")

    # Load original metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # Create subsets (ensure we have enough data for a batch size of 2)
    # We select unique patients to avoid splitting a patient across batches weirdly in this tiny demo
    train_patients = df_train["Patient"].unique()[:5]
    val_patients = df_val["Patient"].unique()[:2]
    test_patients = df_test["Patient"].unique()[:2]

    sub_train = df_train[df_train["Patient"].isin(train_patients)].copy()
    sub_val = df_val[df_val["Patient"].isin(val_patients)].copy()
    sub_test = df_test[df_test["Patient"].isin(test_patients)].copy()

    # Save subsets
    sub_train.to_csv(subset_train_path, index=False)
    sub_val.to_csv(subset_val_path, index=False)
    sub_test.to_csv(subset_test_path, index=False)

    logger.info(
        f"Created subsets: Train={len(sub_train)}, Val={len(sub_val)}, Test={len(sub_test)}"
    )

    # --- OVERRIDE CONFIG ---
    # We modify the Config class attributes directly to point to our subsets
    Config.TRAIN_CSV = subset_train_path
    Config.VAL_CSV = subset_val_path
    Config.TEST_CSV = subset_test_path

    # Reduce training parameters for speed
    Config.N_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    logger.info("Config overridden for demonstration.\n")


def demonstrate_dataset_and_loader():
    logger.info("--- Demonstrating Dataset and DataLoader ---")

    # This will trigger preprocessing on the subset (caching .npy files)
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    logger.info(f"Train batches: {len(train_loader)}")

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify keys
    expected_keys = ["axial", "coronal", "fusion", "anchor", "meta", "target"]
    for k in expected_keys:
        assert k in batch, f"Batch missing key: {k}"

    # Verify shapes
    # Axial/Coronal: (B, 3, 224, 224) - channels first for PyTorch
    assert batch["axial"].shape == (Config.BATCH_SIZE, 3, 224, 224)
    assert batch["coronal"].shape == (Config.BATCH_SIZE, 3, 224, 224)

    # Fusion: (B, 6)
    assert batch["fusion"].shape == (Config.BATCH_SIZE, 6)

    # Anchor: (B, 3)
    assert batch["anchor"].shape == (Config.BATCH_SIZE, 3)

    # Target: (B,)
    assert batch["target"].shape == (Config.BATCH_SIZE,)

    logger.info("DataLoader batch shapes verified.\n")
    return batch


def demonstrate_model_architecture(batch):
    logger.info("--- Demonstrating Model Architecture ---")

    device = torch.device(Config.DEVICE)
    model = CVRNet().to(device)
    loss_fn = RobustLaplaceLoss().to(device)

    # Move batch to device
    axial = batch["axial"].to(device).float()
    coronal = batch["coronal"].to(device).float()
    fusion = batch["fusion"].to(device)
    anchor = batch["anchor"].to(device)
    meta = batch["meta"].to(device)
    targets = batch["target"].to(device)

    # Forward Pass
    alpha, sigma_base, sigma_growth = model(axial, coronal, fusion, anchor)

    logger.info(f"Outputs - Alpha: {alpha.shape}, Sigma Base: {sigma_base.shape}")

    assert alpha.shape == (Config.BATCH_SIZE,)
    assert sigma_base.shape == (Config.BATCH_SIZE,)
    assert sigma_growth.shape == (Config.BATCH_SIZE,)

    # Verify Positivity of Sigmas (Softplus)
    assert torch.all(sigma_base > 0), "Sigma base must be positive"
    assert torch.all(sigma_growth > 0), "Sigma growth must be positive"

    # Loss Calculation
    preds = (alpha, sigma_base, sigma_growth)
    loss = loss_fn(preds, targets, meta)

    logger.info(f"Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"

    logger.info("Model and Loss verification passed.\n")


def demonstrate_training_pipeline():
    logger.info("--- Demonstrating Full Trainer Pipeline ---")

    # Instantiate Trainer
    # This will initialize the model, optimizer, etc.
    trainer = Trainer()

    # Run Training (on the subset, for 2 epochs)
    logger.info("Starting Trainer.train()...")
    trainer.train()

    # Check if best model was saved
    assert os.path.exists(trainer.best_model_path), "Best model file was not created."

    # Run Inference
    logger.info("Starting Trainer.predict()...")
    trainer.predict()

    # Check submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    logger.info(f"Submission generated with {len(sub_df)} rows.")

    # Basic check on submission content
    assert "Patient_Week" in sub_df.columns
    assert "FVC" in sub_df.columns
    assert "Confidence" in sub_df.columns
    assert not sub_df.isnull().values.any(), "Submission contains NaNs"

    logger.info("Trainer pipeline verification passed.\n")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # 1. Test Metric
    demonstrate_metric()

    # 2. Test Preprocessing
    demonstrate_preprocessing()

    # 3. Setup Data Subsets (Crucial for speed)
    setup_subset_data()

    # 4. Test Dataset/Loader
    batch = demonstrate_dataset_and_loader()

    # 5. Test Model Logic
    demonstrate_model_architecture(batch)

    # 6. Test Full Training/Inference Loop
    demonstrate_training_pipeline()

    logger.info("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY.")
