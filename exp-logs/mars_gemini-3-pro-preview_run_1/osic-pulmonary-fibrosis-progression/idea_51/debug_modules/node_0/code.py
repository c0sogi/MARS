import os
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Import provided library components
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.image_processing import generate_tri_slab, normalize_hu
from library.dataset import LungDataset
from library.model import NSLHN
from library.loss import LaplaceLikelihoodLoss
from library.engine import run_training

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup Configuration for Speed and Reproducibility
    print("\n[1] Configuring Environment...")
    seed_everything(42)

    # Override Config for a fast demo run
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Only use 10 samples
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Use a specific cache directory for this demo to avoid polluting main working dir
    DEMO_CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    Config.CACHE_DIR = DEMO_CACHE_DIR
    if os.path.exists(DEMO_CACHE_DIR):
        shutil.rmtree(DEMO_CACHE_DIR)
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)

    print(f"Debug Mode: Enabled")
    print(f"Epochs: {Config.EPOCHS}, Batch Size: {Config.BATCH_SIZE}")
    print(f"Cache Dir: {Config.CACHE_DIR}")

    # 2. Verify Image Processing Logic (Unit Test)
    print("\n[2] Verifying Image Processing Logic...")
    # Simulate a 3D CT volume: Depth=100, Height=512, Width=512
    # Values in HU range roughly -1000 to 400 (after normalization 0 to 1)
    dummy_volume = np.random.rand(100, 512, 512).astype(np.float32)

    # Test Axial Slab Generation
    slab_axial = generate_tri_slab(dummy_volume, axis="axial")
    assert slab_axial.shape == (
        224,
        224,
        3,
    ), f"Axial slab shape mismatch. Expected (224, 224, 3), got {slab_axial.shape}"
    assert (
        slab_axial.min() >= 0 and slab_axial.max() <= 1
    ), "Axial slab values out of range [0, 1]"

    # Test Coronal Slab Generation
    slab_coronal = generate_tri_slab(dummy_volume, axis="coronal")
    assert slab_coronal.shape == (
        224,
        224,
        3,
    ), f"Coronal slab shape mismatch. Expected (224, 224, 3), got {slab_coronal.shape}"

    print("Image processing logic verified successfully.")

    # 3. Verify Model Architecture (Unit Test)
    print("\n[3] Verifying Model Architecture...")
    device = torch.device("cpu")  # Use CPU for simple logic check
    model = NSLHN().to(device)
    model.eval()

    # Create dummy inputs matching batch size 2
    # Images: (B, 3, 224, 224)
    dummy_img = torch.randn(2, 3, 224, 224).to(device)
    # Tabular: (B, 4) -> Age, Sex, Smoking, Percent
    dummy_tab = torch.randn(2, 4).to(device)

    with torch.no_grad():
        output = model(dummy_img, dummy_img, dummy_tab)

    # Output should be (B, 3) -> [alpha, sigma_base, sigma_growth]
    assert output.shape == (
        2,
        3,
    ), f"Model output shape mismatch. Expected (2, 3), got {output.shape}"

    # Verify constraints: sigma_base (idx 1) and sigma_growth (idx 2) must be positive
    # The model uses Softplus, so this should be guaranteed.
    sigmas = output[:, 1:]
    assert (
        sigmas > 0
    ).all(), "Model output constraints failed: Sigmas must be positive."

    print("Model architecture verified successfully.")

    # 4. Verify Loss Function (Unit Test)
    print("\n[4] Verifying Loss Function...")
    criterion = LaplaceLikelihoodLoss()

    # Dummy targets
    # Output: alpha=0, sigma_base=100, sigma_growth=10
    # Note: Model outputs raw values before softplus/constraints in some architectures,
    # but NSLHN applies Softplus internally.
    # Let's construct a tensor that represents the output of the model.
    # alpha = -5.0, sigma_base = 100, sigma_growth = 5
    model_out = torch.tensor([[-5.0, 100.0, 5.0], [-2.0, 80.0, 2.0]])

    # Targets
    target_fvc = torch.tensor([[2000.0], [2500.0]])
    baseline_fvc = torch.tensor([[2200.0], [2600.0]])
    time_delta = torch.tensor([[10.0], [5.0]])

    loss = criterion(model_out, target_fvc, baseline_fvc, time_delta)

    assert torch.isfinite(loss), "Loss returned non-finite value."
    assert loss.ndim == 0, "Loss should be a scalar."

    print(f"Loss calculation verified. Value: {loss.item():.4f}")

    # 5. Verify Metric Calculation (Unit Test)
    print("\n[5] Verifying Metric Calculation...")
    # Case: Perfect prediction
    # Delta = 0, Sigma = 70 (clipped min)
    # Metric = - (sqrt(2)*0)/70 - ln(sqrt(2)*70) = - ln(98.99) approx -4.595
    y_true = np.array([2000])
    y_pred = np.array([2000])
    sigma = np.array([50])  # Should clip to 70

    metric_val = calculate_metric(y_true, y_pred, sigma)

    expected_val = -np.log(np.sqrt(2) * 70)
    assert np.isclose(
        metric_val, expected_val, atol=1e-4
    ), f"Metric calculation failed. Got {metric_val}, expected {expected_val}"

    print(f"Metric calculation verified. Value: {metric_val:.4f}")

    # 6. Verify Dataset Loading (Integration Test)
    print("\n[6] Verifying Dataset Loading...")
    # Load metadata
    try:
        train_df = pd.read_csv(Config.TRAIN_CSV)
        # Take a tiny subset
        train_df_sub = train_df.head(4)

        ds = LungDataset(train_df_sub, mode="train", cache_dir=Config.CACHE_DIR)

        # Fetch one item
        sample = ds[0]

        # Check keys
        required_keys = [
            "img_axial",
            "img_coronal",
            "tabular",
            "time_delta",
            "baseline_fvc",
            "target",
            "patient_id",
        ]
        for k in required_keys:
            assert k in sample, f"Dataset sample missing key: {k}"

        # Check shapes
        assert sample["img_axial"].shape == (
            3,
            224,
            224,
        ), "Axial image tensor shape incorrect"
        assert sample["tabular"].shape == (4,), "Tabular feature vector shape incorrect"
        assert sample["target"].shape == (1,), "Target scalar shape incorrect"

        print(f"Dataset verified. Loaded patient: {sample['patient_id']}")

    except FileNotFoundError:
        print(
            "Skipping Dataset verification: Metadata file not found (expected in ./metadata/train.csv)"
        )
    except Exception as e:
        print(f"Dataset verification failed: {e}")
        raise e

    # 7. Run Training Loop (Integration Test)
    print("\n[7] Running Training Loop (Fast Debug Mode)...")
    try:
        best_score = run_training(debug=True)
        print(f"Training loop completed successfully. Best Score: {best_score}")
    except Exception as e:
        print(f"Training loop failed: {e}")
        raise e

    # 8. Cleanup
    print("\n[8] Cleaning up...")
    if os.path.exists(DEMO_CACHE_DIR):
        shutil.rmtree(DEMO_CACHE_DIR)
        print("Temporary cache directory removed.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
