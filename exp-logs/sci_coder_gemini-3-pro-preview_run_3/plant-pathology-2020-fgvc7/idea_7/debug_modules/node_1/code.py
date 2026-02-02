import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import timm

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc, get_class_weights
from library.data import prepare_folds, get_loaders, get_test_loader
from library.models import AppleEfficientNet, AppleSwin
from library.train import run_training
from library.inference import run_inference


def run_demo():
    print("==== Starting Apple Disease Detection Demo ====")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Purposes
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Use a separate working directory for the demo to avoid conflicts
    DEMO_WORKING_DIR = "./working/demo_test"
    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    # Override Config attributes
    Config.WORKING_DIR = DEMO_WORKING_DIR
    Config.SUBMISSION_DIR = os.path.join(DEMO_WORKING_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.CLASS_WEIGHTS_PATH = os.path.join(DEMO_WORKING_DIR, "class_weights.npy")

    # Optimization settings
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Very small subset for instant execution
    Config.EPOCHS = 1  # Single epoch
    Config.N_FOLDS = 2  # Run 2 folds to verify loop logic
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure directories exist
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")

    # -------------------------------------------------------------------------
    # 2. Monkeypatching to Disable Pretrained Weights Download
    # -------------------------------------------------------------------------
    # To make this demo fast and offline-capable, we force pretrained=False
    print("[2] Monkeypatching timm.create_model to skip weight downloads...")
    _original_create_model = timm.create_model

    def _mock_create_model(*args, **kwargs):
        if "pretrained" in kwargs:
            kwargs["pretrained"] = False
        return _original_create_model(*args, **kwargs)

    timm.create_model = _mock_create_model

    # -------------------------------------------------------------------------
    # 3. Component Verification: Utils
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Library Utils...")

    # Test Seeding
    seed_everything(Config.SEED)
    r1 = np.random.rand()
    seed_everything(Config.SEED)
    r2 = np.random.rand()
    assert r1 == r2, "Seed everything failed to produce reproducible numpy results."

    # Test ROC AUC
    # Create dummy one-hot targets and probabilities
    y_true = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    y_pred_perfect = np.array(
        [
            [0.9, 0.0, 0.05, 0.05],
            [0.1, 0.8, 0.05, 0.05],
            [0.05, 0.05, 0.8, 0.1],
            [0.05, 0.05, 0.1, 0.8],
        ]
    )
    auc_score = calculate_roc_auc(y_true, y_pred_perfect)
    assert auc_score > 0.9, f"ROC AUC calculation seems incorrect. Score: {auc_score}"
    print("Utils verification passed.")

    # -------------------------------------------------------------------------
    # 4. Component Verification: Data
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Data Pipeline...")

    # Test Fold Preparation
    # This creates the folds.parquet file
    df_folds = prepare_folds(load_cached_data=False)
    assert "fold" in df_folds.columns, "Fold column missing in prepared dataframe."
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "folds.parquet")
    ), "Folds parquet file not saved."

    # Test DataLoaders
    train_loader, val_loader = get_loaders(
        fold=0, img_size=224, batch_size=Config.BATCH_SIZE, debug=True
    )

    # Fetch one batch
    images, labels = next(iter(train_loader))

    # Check shapes
    # Images: (B, C, H, W) -> (4, 3, 224, 224)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), f"Incorrect image shape: {images.shape}"
    # Labels: (B, NumClasses) -> (4, 4)
    assert labels.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Incorrect label shape: {labels.shape}"

    print("Data pipeline verification passed.")

    # -------------------------------------------------------------------------
    # 5. Component Verification: Models
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Model Architectures...")

    device = torch.device("cpu")  # Test on CPU for simplicity in unit test
    dummy_input = torch.randn(2, 3, 224, 224).to(device)

    # Test EfficientNet
    model_eff = AppleEfficientNet(pretrained=False).to(device)
    model_eff.eval()
    with torch.no_grad():
        out_eff = model_eff(dummy_input)
    assert out_eff.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"EffNet output shape mismatch: {out_eff.shape}"

    # Test Swin
    model_swin = AppleSwin(pretrained=False).to(device)
    model_swin.eval()
    with torch.no_grad():
        out_swin = model_swin(dummy_input)
    assert out_swin.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Swin output shape mismatch: {out_swin.shape}"

    print("Model architecture verification passed.")

    # -------------------------------------------------------------------------
    # 6. Integration Test: Training Pipeline
    # -------------------------------------------------------------------------
    print("\n[6] Running Training Pipeline (Debug Mode)...")

    # This will train 2 folds x 2 models = 4 short runs
    # Models will be saved to DEMO_WORKING_DIR
    run_training(debug=True)

    # Verify outputs exist
    expected_files = [
        "effnet_fold_0_best.pth",
        "effnet_fold_1_best.pth",
        "swin_fold_0_best.pth",
        "swin_fold_1_best.pth",
    ]

    for fname in expected_files:
        fpath = os.path.join(Config.WORKING_DIR, fname)
        assert os.path.exists(fpath), f"Expected model checkpoint not found: {fname}"

    print("Training pipeline completed successfully.")

    # -------------------------------------------------------------------------
    # 7. Integration Test: Inference Pipeline
    # -------------------------------------------------------------------------
    print("\n[7] Running Inference Pipeline...")

    # This loads the models saved above and generates submission.csv
    run_inference()

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not generated."

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    test_df = pd.read_csv(Config.TEST_CSV)

    assert len(sub_df) == len(test_df), "Submission row count mismatch."
    assert all(
        col in sub_df.columns for col in ["image_id"] + Config.CLASS_LABELS
    ), "Submission columns mismatch."

    print("Inference pipeline completed successfully.")
    print(f"Submission saved to: {Config.SUBMISSION_FILE}")

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
