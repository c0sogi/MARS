import os
import torch
import numpy as np
import pandas as pd
import shutil
from sklearn.metrics import roc_auc_score

# Import from the provided library
from library.config import Config
from library.utils import (
    seed_everything,
    compute_class_weights,
    calculate_metric,
    ModelEMA,
)
from library.dataset import load_data, get_loaders
from library.models import AppleNet, GeM
from library.train import run_training
from library.inference import run_inference


def demonstrate_and_verify():
    print("Starting demonstration and verification script...")

    # =========================================================================
    # 1. Configuration Override for Demo Speed & Safety
    # =========================================================================
    print("\n[1] Overriding Configuration for Fast Execution...")
    # Set up a specific working directory for this demo
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Patch Config attributes
    Config.WORKING_DIR = demo_working_dir
    Config.SUBMISSION_PATH = os.path.join(demo_working_dir, "submission.csv")
    Config.DEBUG = True  # Use small subset (100 samples)
    Config.DEBUG_SAMPLE_SIZE = 50  # Even smaller for extreme speed
    Config.EPOCHS = 1  # Only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.BACKBONES = ["resnet18"]  # Lightweight backbone
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script

    # We disable pretrained weights to ensure this runs without internet access
    # Note: In a real scenario, pretrained=True is preferred.
    # We need to monkeypatch the model creation in run_training/inference or just accept
    # that the library code uses pretrained=True.
    # Since we cannot modify library files, we rely on the environment having cached weights
    # or internet access. If strictly offline and no cache, this might fail.
    # However, standard models like resnet18 are often available.

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Backbone: {Config.BACKBONES}")

    # =========================================================================
    # 2. Verify Utilities
    # =========================================================================
    print("\n[2] Verifying Utilities...")

    # Test seed_everything
    seed_everything(Config.SEED)
    r1 = torch.rand(1).item()
    seed_everything(Config.SEED)
    r2 = torch.rand(1).item()
    assert r1 == r2, "seed_everything failed to ensure reproducibility"
    print("    seed_everything: Passed")

    # Test calculate_metric
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    # Expected: High AUC since predictions align with truth
    score = calculate_metric(y_true, y_pred)
    assert 0.9 <= score <= 1.0, f"calculate_metric returned unexpected score: {score}"
    print(f"    calculate_metric: Passed (Score: {score:.4f})")

    # Test compute_class_weights
    # Load data first to have a dataframe
    train_df, _, _ = load_data(load_cached_data=False)
    weights = compute_class_weights(train_df, load_cached_data=False)
    assert isinstance(weights, torch.Tensor), "Class weights should be a Tensor"
    assert weights.shape[0] == Config.NUM_CLASSES, "Class weights shape mismatch"
    print("    compute_class_weights: Passed")

    # =========================================================================
    # 3. Verify Dataset & Loaders
    # =========================================================================
    print("\n[3] Verifying Dataset and Loaders...")

    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    # Fetch one batch
    images, labels = next(iter(train_loader))

    # Check shapes
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), f"Image batch shape mismatch: {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Label batch shape mismatch: {labels.shape}"

    # Check value ranges (normalized images should generally be roughly in -3 to 3 range)
    assert (
        images.max() <= 5.0 and images.min() >= -5.0
    ), "Image normalization looks suspicious"
    print("    get_loaders: Passed")

    # =========================================================================
    # 4. Verify Model Architecture
    # =========================================================================
    print("\n[4] Verifying Model Architecture...")

    # Test GeM Layer
    gem = GeM(p=1.0)  # p=1 is Average Pooling
    dummy_feat = torch.ones(2, 64, 10, 10)  # B, C, H, W
    out_gem = gem(dummy_feat)  # Should be (2, 64, 1, 1)
    assert out_gem.shape == (2, 64, 1, 1), f"GeM output shape mismatch: {out_gem.shape}"
    assert torch.allclose(
        out_gem, torch.ones_like(out_gem)
    ), "GeM p=1 should act as AvgPool"
    print("    GeM Layer: Passed")

    # Test AppleNet
    # We use pretrained=False here just for the unit test speed/safety
    model = AppleNet(
        backbone_name="resnet18", num_classes=Config.NUM_CLASSES, pretrained=False
    )
    model.to(Config.DEVICE)
    model.eval()

    with torch.no_grad():
        # Use the batch from the loader check
        images = images.to(Config.DEVICE)
        logits = model(images)

    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch: {logits.shape}"
    print("    AppleNet Forward Pass: Passed")

    # =========================================================================
    # 5. Run Training Cycle (Integration Test)
    # =========================================================================
    print("\n[5] Running Training Cycle (1 Epoch, Debug Data)...")

    # This calls the provided training script logic
    # It will train 'resnet18' for 1 epoch on 50 samples
    try:
        run_training()
        print("    Training completed successfully.")
    except Exception as e:
        print(f"    Training failed with error: {e}")
        raise e

    # Verify artifacts
    expected_model_path = os.path.join(Config.WORKING_DIR, "resnet18_best.pth")
    assert os.path.exists(expected_model_path), "Best model file was not saved."
    print("    Model artifact verification: Passed")

    # =========================================================================
    # 6. Run Inference Cycle (Integration Test)
    # =========================================================================
    print("\n[6] Running Inference Cycle...")

    try:
        run_inference()
        print("    Inference completed successfully.")
    except Exception as e:
        print(f"    Inference failed with error: {e}")
        raise e

    # =========================================================================
    # 7. Final Submission Verification
    # =========================================================================
    print("\n[7] Verifying Submission File...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    expected_cols = ["image_id"] + Config.LABELS
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Found: {sub_df.columns}"

    # Check length (should match test set size, which is small in DEBUG mode)
    # In DEBUG mode, test_df is sampled to Config.DEBUG_SAMPLE_SIZE
    # We need to verify if run_inference respected the debug sampling for the dataframe construction
    _, _, test_df_full = load_data(load_cached_data=True)

    # Note: run_inference re-loads data. If Config.DEBUG is True, it samples.
    # So the submission length should match Config.DEBUG_SAMPLE_SIZE (or len(test_df) if smaller)
    expected_len = min(len(test_df_full), Config.DEBUG_SAMPLE_SIZE)
    assert (
        len(sub_df) == expected_len
    ), f"Submission length mismatch. Expected {expected_len}, got {len(sub_df)}"

    # Check values are probabilities
    pred_cols = Config.LABELS
    preds = sub_df[pred_cols].values
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Predictions contain values outside [0, 1]"

    print("    Submission Format: Passed")
    print("\nAll demonstrations and verifications passed successfully!")


if __name__ == "__main__":
    demonstrate_and_verify()
