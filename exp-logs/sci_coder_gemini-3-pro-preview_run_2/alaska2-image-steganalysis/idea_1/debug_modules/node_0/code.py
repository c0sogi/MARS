import os
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_srm_weights, weighted_auc_score
from library.dataset import StegoDataset, get_transforms
from library.model import SRMEfficientNet
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print("--- Starting Library Demonstration ---")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Override Config for speed
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.PRETRAINED = False  # Skip downloading weights
    Config.DEBUG_SAMPLE_SIZE = 50  # Use very small subset

    # Ensure reproducibility
    seed_everything(Config.SEED)
    print("Configuration updated: DEBUG=True, EPOCHS=1, PRETRAINED=False")

    # -------------------------------------------------------------------------
    # 2. Verify Utils
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test SRM Weights
    srm_weights = get_srm_weights()
    print(f"SRM Weights Shape: {srm_weights.shape}")
    assert srm_weights.shape == (30, 1, 5, 5), "SRM weights should be (30, 1, 5, 5)"
    assert srm_weights.dtype == torch.float32, "SRM weights should be float32"

    # Test Weighted AUC
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])
    # Thresholds [0.0, 0.4, 1.0], Weights [2, 1]
    auc = weighted_auc_score(y_true, y_score, Config.TPR_THRESHOLDS, Config.TPR_WEIGHTS)
    print(f"Calculated Weighted AUC: {auc:.4f}")
    assert 0.0 <= auc <= 1.0, "AUC must be between 0 and 1"

    # -------------------------------------------------------------------------
    # 3. Verify Dataset
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Dataset...")

    # Initialize Train Dataset
    train_ds = StegoDataset(
        csv_path=Config.TRAIN_CSV,
        mode="train",
        transform=get_transforms("train"),
        load_cached_data=False,  # Force reload to test logic
    )

    print(f"Train Dataset Length (Debug): {len(train_ds)}")
    assert len(train_ds) > 0, "Dataset should not be empty"

    # Fetch one sample
    img, label = train_ds[0]
    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Label: {label}")

    assert img.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Image shape mismatch. Expected (3, {Config.IMAGE_SIZE}, {Config.IMAGE_SIZE})"
    assert isinstance(label, torch.Tensor), "Label should be a tensor"
    assert label.ndim == 0, "Label should be a scalar tensor"

    # -------------------------------------------------------------------------
    # 4. Verify Model
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    model = SRMEfficientNet(model_name=Config.MODEL_NAME, pretrained=False)
    model.eval()

    # Check SRM Layer freezing
    srm_layer_params = list(model.srm_conv.parameters())
    assert len(srm_layer_params) > 0, "SRM layer should have parameters"
    assert not srm_layer_params[
        0
    ].requires_grad, "SRM layer parameters should be frozen (requires_grad=False)"

    # Check Forward Pass
    dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, 1), "Model output should be (Batch_Size, 1)"

    # -------------------------------------------------------------------------
    # 5. Verify Trainer (Training & Inference)
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Trainer Pipeline...")

    trainer = Trainer()

    # Run Training
    print("Running Trainer.fit()...")
    trainer.fit()

    # Check if checkpoint was saved (if score improved) or at least logic ran
    # Since we use random weights and random data, score might not improve,
    # but we check if the best_model_path is defined.
    print(f"Best model path: {trainer.best_model_path}")

    # Run Inference
    print("Running Trainer.predict()...")

    # Ensure we have a "best model" to load. If fit didn't save one (due to poor performance),
    # we manually save the current model to allow predict() to load it.
    if not os.path.exists(trainer.best_model_path):
        torch.save(trainer.model.state_dict(), trainer.best_model_path)

    trainer.predict()

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created"

    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission File Shape: {sub_df.shape}")
    print("Submission Head:")
    print(sub_df.head())

    assert list(sub_df.columns) == ["Id", "Label"], "Submission columns mismatch"
    assert len(sub_df) > 0, "Submission file is empty"

    print("\n--- Demonstration Completed Successfully ---")
