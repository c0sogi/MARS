import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, metric_laplace_log_likelihood
from library.data import OSICDataset
from library.model import AVRDAN
from library.train import run_training
from library.predict import generate_submission_file

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def print_section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def verify_metric_logic():
    print_section("1. Verifying Metric Logic")

    # Test Case: Perfect prediction
    # FVC_true = 2000, FVC_pred = 2000, Sigma = 100
    # Delta = 0
    # Sigma_clipped = max(100, 70) = 100
    # Metric = - (sqrt(2)*0)/100 - ln(sqrt(2)*100)
    #        = - ln(141.421356)
    #        = - 4.9517...

    y_true = torch.tensor([2000.0])
    y_pred = torch.tensor([2000.0])
    sigma = torch.tensor([100.0])

    score = metric_laplace_log_likelihood(y_true, y_pred, sigma)
    expected = -np.log(np.sqrt(2) * 100)

    print(f"Calculated Score: {score.item():.4f}")
    print(f"Expected Score:   {expected:.4f}")

    assert np.isclose(score.item(), expected, atol=1e-4), "Metric calculation mismatch!"
    print("-> Metric logic verified.")


def verify_data_pipeline():
    print_section("2. Verifying Data Pipeline")

    # Initialize Dataset with debug subset
    dataset = OSICDataset(csv_path=Config.TRAIN_CSV, mode="train")
    print(f"Dataset initialized. Size: {len(dataset)}")

    # Get one sample
    sample = dataset[0]
    print("Sample keys:", sample.keys())

    # Verify Shapes
    # Image: (3, 224, 224)
    assert sample["img_ax"].shape == (
        3,
        224,
        224,
    ), f"Axial image shape mismatch: {sample['img_ax'].shape}"
    assert sample["img_cor"].shape == (
        3,
        224,
        224,
    ), f"Coronal image shape mismatch: {sample['img_cor'].shape}"

    # Tabular GLU: 7 dims (Age, Pct, Sex(2), Smoke(3))
    assert sample["tab_glu"].shape == (
        7,
    ), f"Tabular GLU shape mismatch: {sample['tab_glu'].shape}"

    # Tabular Skip: 3 dims (BaseFVC, BasePct, Age)
    assert sample["tab_skip"].shape == (
        3,
    ), f"Tabular Skip shape mismatch: {sample['tab_skip'].shape}"

    print("-> Data shapes verified.")


def verify_model_architecture():
    print_section("3. Verifying Model Architecture")

    # Instantiate Model
    model = AVRDAN()
    model.to(Config.DEVICE)
    model.eval()

    # Create a dummy batch
    B = 2
    img_ax = torch.randn(B, 3, 224, 224).to(Config.DEVICE)
    img_cor = torch.randn(B, 3, 224, 224).to(Config.DEVICE)
    tab_glu = torch.randn(B, 7).to(Config.DEVICE)
    tab_skip = torch.randn(B, 3).to(Config.DEVICE)
    delta_week = torch.tensor([10.0, 20.0]).to(Config.DEVICE)
    baseline_fvc = torch.tensor([2500.0, 3000.0]).to(Config.DEVICE)

    # Forward Pass
    with torch.no_grad():
        fvc_pred, conf_pred = model(
            img_ax, img_cor, tab_glu, tab_skip, delta_week, baseline_fvc
        )

    print(f"FVC Pred Shape: {fvc_pred.shape}")
    print(f"Conf Pred Shape: {conf_pred.shape}")

    # Assertions
    assert fvc_pred.shape == (B,), "FVC prediction shape incorrect"
    assert conf_pred.shape == (B,), "Confidence prediction shape incorrect"

    # Check positivity of confidence (softplus output)
    assert (conf_pred > 0).all(), "Confidence values must be positive"

    print("-> Model forward pass verified.")


def run_fast_training():
    print_section("4. Running Training Loop (Fast Mode)")

    # Override Config for speed
    print("Overriding Config for demonstration...")
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.DEBUG_SAMPLE_SIZE = 32  # Small subset
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data
    Config.EARLY_STOPPING_PATIENCE = 2

    # Execute training
    # This function handles data loading, model init, optimization, and saving
    run_training()

    # Verify artifact creation
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model file not found at {Config.MODEL_SAVE_PATH}"
    print(f"-> Training complete. Model saved to {Config.MODEL_SAVE_PATH}")


def run_inference_generation():
    print_section("5. Running Inference Generation")

    # Config is already overridden from previous step, but let's ensure test settings
    # The generate_submission_file function uses Config.TEST_CSV and Config.MODEL_SAVE_PATH

    generate_submission_file()

    # Verify submission file
    submission_path = "./submission/submission.csv"
    assert os.path.exists(submission_path), "Submission file not found!"

    df = pd.read_csv(submission_path)
    print("Submission Head:")
    print(df.head(3))

    required_cols = ["Patient_Week", "FVC", "Confidence"]
    assert all(
        col in df.columns for col in required_cols
    ), "Submission columns missing!"
    assert len(df) > 0, "Submission file is empty!"

    print("-> Inference complete and verified.")


if __name__ == "__main__":
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # 1. Verify Metric
    verify_metric_logic()

    # 2. Verify Data Pipeline
    # Note: We set DEBUG_SAMPLE_SIZE before init to speed up loading
    Config.DEBUG_SAMPLE_SIZE = 20
    verify_data_pipeline()

    # 3. Verify Model
    verify_model_architecture()

    # 4. Run Training
    run_fast_training()

    # 5. Run Inference
    run_inference_generation()

    print_section("Demonstration Complete")
