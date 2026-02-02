import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
import math

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.loss import LaplaceLogLikelihoodLoss
from library.data import OSICDataset, get_dataloaders
from library.model import OSICModel
from library.train import run_training
from library.inference import predict_test


def demonstrate_utils_and_loss():
    print("\n=== 1. Verifying Utils and Loss ===")

    # 1.1 Verify Metric
    # Case: Perfect prediction (delta=0) with confidence 100
    y_true = np.array([2000.0])
    y_pred = np.array([2000.0])
    sigma = np.array([100.0])

    # Manual Calc:
    # sigma_clipped = max(100, 70) = 100
    # delta = 0
    # metric = - (sqrt(2)*0)/100 - ln(sqrt(2)*100)
    # metric = - ln(141.421356) ~= -4.95174
    expected_metric = -np.log(np.sqrt(2) * 100)
    calculated_metric = laplace_log_likelihood_metric(y_true, y_pred, sigma)

    print(f"Metric Check: Expected {expected_metric:.4f}, Got {calculated_metric:.4f}")
    assert np.isclose(
        calculated_metric, expected_metric, atol=1e-4
    ), "Metric calculation mismatch!"

    # 1.2 Verify Loss
    # Loss = (sqrt(2) * delta) / sigma + ln(sigma)
    # Case: Delta=0, Sigma=100
    # Loss = 0 + ln(100) ~= 4.60517
    loss_fn = LaplaceLogLikelihoodLoss(reduction="mean")
    t_preds = torch.tensor([[2000.0, 100.0]], dtype=torch.float32)  # [FVC, Sigma]
    t_targets = torch.tensor([2000.0], dtype=torch.float32)

    expected_loss = np.log(100.0)
    calculated_loss = loss_fn(t_preds, t_targets).item()

    print(f"Loss Check:   Expected {expected_loss:.4f}, Got {calculated_loss:.4f}")
    assert np.isclose(
        calculated_loss, expected_loss, atol=1e-4
    ), "Loss calculation mismatch!"
    print("Utils and Loss verified successfully.")


def demonstrate_data_and_model():
    print("\n=== 2. Verifying Data Loading and Model ===")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)

    # Initialize Dataset (Debug mode with small subset handled by logic later,
    # but here we just take the first few rows manually for quick check)
    dataset = OSICDataset(train_df.head(10), mode="train", load_cached_data=True)

    print(f"Dataset Length: {len(dataset)}")

    # Fetch one item
    img, tab, target = dataset[0]

    # Verify Shapes
    # Image: (3, 256, 256)
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch. Got {img.shape}"
    # Tabular: 8 features (Weeks, FVC, Pct, Age, Sex, Smoke*3)
    assert tab.shape[0] == 8, f"Tabular feature dimension mismatch. Got {tab.shape[0]}"
    # Target: (1,)
    assert target.shape == (1,), f"Target shape mismatch. Got {target.shape}"

    print(
        f"Sample Shapes Verified: Img {tuple(img.shape)}, Tab {tuple(tab.shape)}, Target {tuple(target.shape)}"
    )

    # Initialize Model
    device = torch.device("cpu")  # Use CPU for this quick check
    model = OSICModel().to(device)
    model.eval()

    # Forward pass with a batch of size 2
    batch_imgs = torch.stack([img, img]).to(device)
    batch_tabs = torch.stack([torch.from_numpy(tab), torch.from_numpy(tab)]).to(device)

    with torch.no_grad():
        preds = model(batch_imgs, batch_tabs)

    # Verify Output Shape: (Batch_Size, 2) -> [FVC, Sigma]
    assert preds.shape == (
        2,
        2,
    ), f"Model output shape mismatch. Expected (2, 2), got {preds.shape}"
    print("Model forward pass successful. Output shape verified.")


def demonstrate_training_pipeline():
    print("\n=== 3. Running Training Pipeline (Debug Mode) ===")

    # Override Config for speed
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directory exists
    Config.setup()

    # Run Training
    # This function handles data loading, model init, training loop, and saving checkpoint
    run_training(debug=True, epochs=1)

    # Verify Checkpoint
    checkpoint_path = os.path.join(Config.MODEL_CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(
        checkpoint_path
    ), "Checkpoint file 'best_model.pth' was not created!"
    print(f"Training complete. Checkpoint found at {checkpoint_path}")


def demonstrate_inference():
    print("\n=== 4. Running Inference ===")

    # Run Inference
    # This uses the checkpoint generated in the previous step
    predict_test(batch_size=2)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created!"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(sub_df)} rows.")

    # Check Columns
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert all(
        col in sub_df.columns for col in expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    # Check content validity
    # Confidence should be >= 70 (clipped)
    min_conf = sub_df["Confidence"].min()
    assert min_conf >= 70, f"Found confidence value < 70: {min_conf}"

    print("Inference verified successfully.")
    print("\nSample Submission Rows:")
    print(sub_df.head())


if __name__ == "__main__":
    # Set global seed
    seed_everything(Config.SEED)

    print(f"Starting Demonstration | Device: {Config.DEVICE}")

    try:
        demonstrate_utils_and_loss()
        demonstrate_data_and_model()
        demonstrate_training_pipeline()
        demonstrate_inference()

        print("\nAll demonstrations completed successfully!")

    except AssertionError as e:
        print(f"\n[FAILED] Assertion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAILED] An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
