import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, metric_function
from library.data import DataProcessor, OSICDataset, get_dataloaders
from library.model import TSCPNet, laplace_log_likelihood_loss
from library.train import train_model, generate_predictions

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def demo_metric_verification():
    """
    Demonstrates and verifies the modified Laplace Log Likelihood metric.
    """
    print("\n[1] Verifying Metric Function...")

    # Case 1: Perfect prediction
    # Delta = 0, Sigma = 70 (clipped)
    # Metric = - (sqrt(2)*0)/70 - ln(sqrt(2)*70) = -ln(98.99) approx -4.595
    true_fvc = np.array([2000])
    pred_fvc = np.array([2000])
    conf = np.array([70])

    score = metric_function(true_fvc, pred_fvc, conf)
    expected = -np.log(np.sqrt(2) * 70)

    assert np.isclose(
        score, expected, atol=1e-4
    ), f"Metric mismatch for perfect prediction. Got {score}, expected {expected}"

    # Case 2: Large Error (Clipped at 1000)
    # Delta = 2000 -> Clipped to 1000
    # Sigma = 100
    # Metric = - (sqrt(2)*1000)/100 - ln(sqrt(2)*100)
    true_fvc = np.array([2000])
    pred_fvc = np.array([4000])
    conf = np.array([100])

    score = metric_function(true_fvc, pred_fvc, conf)
    delta_clipped = 1000
    sigma_clipped = 100
    expected = -(np.sqrt(2) * delta_clipped) / sigma_clipped - np.log(
        np.sqrt(2) * sigma_clipped
    )

    assert np.isclose(
        score, expected, atol=1e-4
    ), f"Metric mismatch for clipped error. Got {score}, expected {expected}"

    print("    Metric verification passed.")


def demo_data_processing():
    """
    Demonstrates data processing utilities and dataset loading.
    """
    print("\n[2] Verifying Data Processing...")

    # 1. Test Tri-Slab Generation (Static Method)
    # Create a dummy volume (Depth=10, H=32, W=32)
    vol_depth, h, w = 10, 32, 32
    dummy_volume = np.random.rand(vol_depth, h, w).astype(np.float32)

    # Generate slab
    slab = DataProcessor.generate_tri_slab(dummy_volume, axis=0, overlap=0.15)

    # Check shape: Should be (H, W, 3)
    assert slab.shape == (
        h,
        w,
        3,
    ), f"Slab shape mismatch. Got {slab.shape}, expected {(h, w, 3)}"
    print("    Tri-Slab generation logic verified.")

    # 2. Test Dataset Instantiation
    # We use the training metadata provided
    ds = OSICDataset(Config.META_TRAIN, mode="train", transform=None, load_cached=False)

    # Fetch one item
    # Note: If pydicom is missing, DataProcessor returns dummy zeros, which is handled gracefully.
    item = ds[0]

    # Verify keys
    required_keys = [
        "img_axial",
        "img_coronal",
        "meta",
        "week_diff",
        "baseline_fvc",
        "target",
    ]
    for k in required_keys:
        assert k in item, f"Missing key {k} in dataset item"

    # Verify Tensor Shapes
    # Images: (3, 224, 224)
    assert item["img_axial"].shape == (
        3,
        224,
        224,
    ), f"Axial image shape wrong: {item['img_axial'].shape}"
    assert item["img_coronal"].shape == (
        3,
        224,
        224,
    ), f"Coronal image shape wrong: {item['img_coronal'].shape}"
    # Meta: (8,)
    assert item["meta"].shape == (8,), f"Meta shape wrong: {item['meta'].shape}"

    print(f"    Dataset loaded successfully. Sample patient: {item['patient_id']}")


def demo_model_architecture():
    """
    Demonstrates Model instantiation and Forward pass.
    """
    print("\n[3] Verifying Model Architecture...")

    device = torch.device("cpu")  # Use CPU for quick logic check
    model = TSCPNet().to(device)
    model.eval()

    # Create dummy batch
    B = 2
    dummy_ax = torch.randn(B, 3, 224, 224).to(device)
    dummy_cor = torch.randn(B, 3, 224, 224).to(device)
    dummy_meta = torch.randn(B, 8).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_ax, dummy_cor, dummy_meta)

    # Output should be (B, 3) -> [alpha, sigma_base, sigma_growth]
    assert output.shape == (B, 3), f"Model output shape mismatch. Got {output.shape}"

    # Check constraints
    # sigma_base and sigma_growth should be positive (Softplus)
    # alpha can be negative
    sigmas = output[:, 1:]
    assert torch.all(sigmas >= 0), "Sigma values must be non-negative"

    print("    Model forward pass verified.")


def demo_training_pipeline():
    """
    Demonstrates the full training and inference pipeline using library functions.
    """
    print("\n[4] Running Training Pipeline Demo...")

    # 1. Configure for Speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.CACHE_DIR = "./working/demo_cache"  # Isolate cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 2. Run Training
    # This uses train_model from library.train
    best_score = train_model(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        device_name=Config.DEVICE,
        checkpoint_dir="./working/checkpoints_demo",
    )

    print(f"    Training finished. Best Score: {best_score}")

    # Verify checkpoint creation
    ckpt_path = os.path.join("./working/checkpoints_demo", "best_model.pth")
    assert os.path.exists(ckpt_path), "Checkpoint file was not created."

    # 3. Run Inference
    # This uses generate_predictions from library.train
    sub_path = "./working/submission_demo.csv"
    generate_predictions(
        model_path=ckpt_path,
        output_path=sub_path,
        batch_size=Config.BATCH_SIZE,
        device_name=Config.DEVICE,
    )

    # Verify Submission
    assert os.path.exists(sub_path), "Submission file was not created."
    df_sub = pd.read_csv(sub_path)

    # Check columns
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert all(
        col in df_sub.columns for col in expected_cols
    ), "Submission columns mismatch."

    # Check content
    assert len(df_sub) > 0, "Submission file is empty."
    assert not df_sub.isnull().values.any(), "Submission contains NaN values."

    print(f"    Inference finished. Submission rows: {len(df_sub)}")


if __name__ == "__main__":
    # Set global seed for reproducibility
    set_seed(42)

    print("=" * 40)
    print("LUNG DECLINE PREDICTION LIBRARY DEMO")
    print("=" * 40)

    try:
        demo_metric_verification()
        demo_data_processing()
        demo_model_architecture()
        demo_training_pipeline()

        print("\n" + "=" * 40)
        print("ALL DEMOS COMPLETED SUCCESSFULLY")
        print("=" * 40)

    except AssertionError as e:
        print(f"\n[FAIL] Assertion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
