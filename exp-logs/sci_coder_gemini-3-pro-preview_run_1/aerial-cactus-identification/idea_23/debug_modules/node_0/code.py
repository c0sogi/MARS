import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, mixup_data, mixup_criterion
from library.dataset import CactusDataset
from library.model import QualityRepVGG
from library.trainer import Trainer


def run_demo():
    print("=== Starting Cactus Classification Demo ===")

    # 1. Override Config for Speed
    print("\n[1] Configuring environment for rapid demonstration...")
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.N_FOLDS = 2
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.EXPERIMENT_NAME = "demo_run_script"

    # Update derived paths based on new experiment name
    Config.WORK_DIR = os.path.join("./working", Config.EXPERIMENT_NAME)
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORK_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORK_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print("Configuration updated. Debug mode enabled.")

    # 2. Verify Utilities (Mixup)
    print("\n[2] Verifying Utility Functions (Mixup)...")
    batch_size = 4
    dummy_imgs = torch.randn(batch_size, 3, 32, 32).to(Config.DEVICE)
    dummy_labels = torch.randint(0, 2, (batch_size, 1)).float().to(Config.DEVICE)
    dummy_quality = torch.rand(batch_size, 1).to(Config.DEVICE)

    mixed_x, y_a, y_b, q_a, q_b, lam = mixup_data(
        dummy_imgs, dummy_labels, dummy_quality, alpha=0.2, device=Config.DEVICE
    )

    assert mixed_x.shape == dummy_imgs.shape, "Mixed images shape mismatch"
    assert y_a.shape == dummy_labels.shape, "Mixed labels shape mismatch"
    print("Mixup data generation successful.")

    # Test Mixup Criterion
    crit_cls = torch.nn.BCEWithLogitsLoss()
    crit_qual = torch.nn.MSELoss()
    pred_cls = torch.randn(batch_size, 1).to(Config.DEVICE)
    pred_qual = torch.randn(batch_size, 1).to(Config.DEVICE)

    loss = mixup_criterion(
        crit_cls,
        crit_qual,
        pred_cls,
        pred_qual,
        y_a,
        y_b,
        q_a,
        q_b,
        lam,
        aux_weight=0.1,
    )
    assert isinstance(loss, torch.Tensor), "Loss is not a tensor"
    assert not torch.isnan(loss), "Loss contains NaNs"
    print("Mixup criterion calculation successful.")

    # 3. Verify Dataset
    print("\n[3] Verifying Dataset Loading...")
    # We use the existing metadata files
    dataset = CactusDataset(
        metadata_path=Config.TRAIN_META_PATH,
        mode="train",
        load_cached_data=False,  # Force processing to test logic
    )

    # In Debug mode, dataset is limited to 100
    print(f"Dataset length: {len(dataset)}")
    assert (
        len(dataset) == 100
    ), f"Expected 100 samples in DEBUG mode, got {len(dataset)}"

    img, label, qual = dataset[0]
    print(
        f"Sample shapes - Image: {img.shape}, Label: {label.shape}, Quality: {qual.shape}"
    )

    assert img.shape == (3, 32, 32), "Incorrect image dimensions"
    assert label.shape == (1,), "Incorrect label dimensions"
    assert qual.shape == (1,), "Incorrect quality dimensions"
    assert img.max() <= 1.0 and img.min() >= 0.0, "Image not normalized to [0, 1]"
    print("Dataset verification successful.")

    # 4. Verify Model Architecture
    print("\n[4] Verifying Model Architecture (QualityRepVGG)...")
    model = QualityRepVGG(num_classes=1, deploy=False).to(Config.DEVICE)

    # Test Forward Pass (Training Mode)
    dummy_input = torch.randn(2, 3, 32, 32).to(Config.DEVICE)
    outputs = model(dummy_input)

    assert isinstance(
        outputs, tuple
    ), "Model in training mode should return tuple (cls, qual)"
    cls_out, qual_out = outputs
    assert cls_out.shape == (2, 1), "Classification output shape mismatch"
    assert qual_out.shape == (2, 1), "Quality output shape mismatch"
    print("Model training forward pass successful.")

    # Test Reparameterization (Deploy Mode)
    print("Testing reparameterization...")
    model.eval()
    model.reparameterize()

    assert model.deploy is True, "Model deploy flag not set to True"
    assert (
        not hasattr(model, "linear_qual") or model.linear_qual is None
    ), "Auxiliary head not removed"

    # Test Forward Pass (Deploy Mode)
    deploy_out = model(dummy_input)
    assert isinstance(
        deploy_out, torch.Tensor
    ), "Deployed model should return Tensor, not tuple"
    assert deploy_out.shape == (2, 1), "Deployed output shape mismatch"
    print("Model reparameterization successful.")

    # 5. Verify Trainer (Training Loop)
    print("\n[5] Verifying Training Loop...")
    trainer = Trainer()

    # Run Fold 0
    # This will use the DEBUG dataset (100 samples), batch size 8, for 1 epoch.
    # It should be very fast.
    auc_score = trainer.run_fold(fold_idx=0)

    print(f"Fold 0 completed with AUC: {auc_score}")

    # Check artifacts
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_fold0.pth")
    swa_model_path = os.path.join(Config.CHECKPOINT_DIR, "swa_fold0.pth")

    # Note: SWA starts at epoch 20 by default in config, but we set EPOCHS=1.
    # So SWA logic might not trigger fully, but the code handles it.
    # However, best_model_path should definitely exist.
    assert os.path.exists(
        best_model_path
    ), f"Best model checkpoint not found at {best_model_path}"
    print("Training loop verification successful.")

    # 6. Verify Inference
    print("\n[6] Verifying Inference on Test Set...")
    # We need to simulate SWA model existence if it wasn't created due to low epochs
    # just so the prediction loop works seamlessly for the demo logic.
    if not os.path.exists(swa_model_path):
        # Copy best model to swa path for the sake of the demo prediction loop
        # which prefers SWA but falls back to best.
        shutil.copy(best_model_path, swa_model_path)

    trainer.predict_test_set()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not generated"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Rows: {len(sub_df)}")
    print(sub_df.head())

    assert list(sub_df.columns) == ["id", "has_cactus"], "Submission columns mismatch"
    assert len(sub_df) > 0, "Submission file is empty"

    # In debug mode, test set is also limited?
    # The Dataset class limits based on Config.DEBUG.
    # The test metadata has 3325 rows.
    # If Config.DEBUG is True, Dataset will only load head(100).
    assert (
        len(sub_df) == 100
    ), f"Expected 100 predictions in DEBUG mode, got {len(sub_df)}"

    print("Inference verification successful.")

    print("\n=== All Demonstrations Passed Successfully ===")


if __name__ == "__main__":
    run_demo()
