import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
# Note: We assume the library files are in the ./library directory as provided.
from library.config import Config
from library.utils import seed_everything, quadratic_weighted_kappa
from library.dataset import RetinopathyDataset, get_dataloaders
from library.model import EfficientNetV2Ordinal, run_training, generate_submission
import library.model  # Imported to facilitate monkeypatching


def main():
    print("=== Starting Diabetic Retinopathy Classification Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Purposes
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Set paths to a specific demo directory to avoid overwriting production runs
    DEMO_DIR = "./working/demo_run"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Update Config global state
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.BEST_MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Hyperparameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 16  # Use a tiny subset
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # Set Reproducibility
    seed_everything(Config.SEED)
    print("    Configuration updated.")

    # ---------------------------------------------------------
    # 2. Monkeypatch Model to Disable Pretraining
    # ---------------------------------------------------------
    # We force pretrained=False to avoid downloading weights during this demo.
    print("\n[2] Monkeypatching model to disable pretraining (offline mode)...")

    _original_init = library.model.EfficientNetV2Ordinal.__init__

    def patched_init(
        self, model_name=Config.MODEL_NAME, pretrained=True, drop_rate=Config.DROP_RATE
    ):
        # Force pretrained=False regardless of argument
        _original_init(
            self, model_name=model_name, pretrained=False, drop_rate=drop_rate
        )

    library.model.EfficientNetV2Ordinal.__init__ = patched_init
    print("    Model initialization patched.")

    # ---------------------------------------------------------
    # 3. Verify Metric Logic
    # ---------------------------------------------------------
    print("\n[3] Verifying Metric (Quadratic Weighted Kappa)...")

    # Case 1: Perfect agreement
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0.1, 1.1, 1.9, 3.0, 4.0])  # Values close to integers
    score = quadratic_weighted_kappa(y_true, y_pred)
    print(f"    Perfect Agreement Score: {score:.4f}")
    assert score > 0.95, "QWK should be near 1.0 for perfect agreement"

    # Case 2: Complete disagreement
    y_true_bad = np.array([0, 0, 0, 0, 0])
    y_pred_bad = np.array([4, 4, 4, 4, 4])
    score_bad = quadratic_weighted_kappa(y_true_bad, y_pred_bad)
    print(f"    Disagreement Score: {score_bad:.4f}")
    # QWK is 0 for random/constant mismatch usually, or negative.
    assert score_bad < 0.1, "QWK should be low for disagreement"
    print("    Metric logic verified.")

    # ---------------------------------------------------------
    # 4. Verify Dataset and DataLoaders
    # ---------------------------------------------------------
    print("\n[4] Verifying Dataset and DataLoaders...")

    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.DEBUG, subset_size=Config.DEBUG_SUBSET_SIZE
    )

    # Fetch a single batch
    images, targets = next(iter(train_loader))

    print(f"    Batch Image Shape: {images.shape}")
    print(f"    Batch Target Shape: {targets.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect image tensor shape"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_OUTPUTS,
    ), "Incorrect target tensor shape"

    # Verify Ordinal Encoding Logic
    # Example: Class 2 should be [1, 1, 0, 0]
    # We check that for every sample, the ones appear before zeros.
    for i in range(Config.BATCH_SIZE):
        t = targets[i].numpy()
        # Check if sorted descending (1s then 0s)
        is_consistent = np.all(t[:-1] >= t[1:])
        assert is_consistent, f"Invalid ordinal encoding found: {t}"

    print("    Dataset logic verified.")

    # ---------------------------------------------------------
    # 5. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[5] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = EfficientNetV2Ordinal()
    model.to(device)
    model.eval()

    with torch.no_grad():
        # Pass the batch fetched earlier
        output = model(images.to(device))

    print(f"    Model Output Shape: {output.shape}")
    assert output.shape == (
        Config.BATCH_SIZE,
        Config.NUM_OUTPUTS,
    ), "Model output shape mismatch"
    print("    Model architecture verified.")

    # ---------------------------------------------------------
    # 6. Run Training Loop (Integration Test)
    # ---------------------------------------------------------
    print("\n[6] Running Training Loop...")

    best_qwk = run_training(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.EPOCHS,
        lr=1e-3,
        save_path=Config.BEST_MODEL_PATH,
    )

    print(f"    Training finished. Best QWK: {best_qwk:.4f}")
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not saved."
    print("    Training integration verified.")

    # ---------------------------------------------------------
    # 7. Generate Submission (Inference Test)
    # ---------------------------------------------------------
    print("\n[7] Generating Submission...")

    generate_submission(
        test_loader=test_loader,
        model_path=Config.BEST_MODEL_PATH,
        output_path=Config.SUBMISSION_PATH,
    )

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Validate Submission File Content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission shape: {df_sub.shape}")
    print(f"    Submission columns: {df_sub.columns.tolist()}")

    assert (
        len(df_sub) == Config.DEBUG_SUBSET_SIZE
    ), f"Expected {Config.DEBUG_SUBSET_SIZE} rows in submission, got {len(df_sub)}"
    assert list(df_sub.columns) == [
        "id_code",
        "diagnosis",
    ], "Submission columns mismatch"
    assert (
        df_sub["diagnosis"].between(0, 4).all()
    ), "Predictions contain values outside range 0-4"

    print("    Submission logic verified.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
