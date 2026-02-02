import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, compute_auc, save_checkpoint, load_checkpoint
from library.dataset import get_dataloaders
from library.model_components import GatedTransformerResFunnelHybrid
from library.train import train_model, generate_submission


def run_demo():
    print("============================================================")
    print("       Manufacturing Control - Library Demo Script          ")
    print("============================================================")

    # ------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # ------------------------------------------------------------------
    # We override the Config paths to isolate the demo execution from
    # any existing production artifacts.
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print(f"[Setup] Working directory set to: {demo_dir}")

    # Monkey-patch Config to use the demo directory
    Config.WORKING_DIR = demo_dir
    Config.PROCESSED_DATA = os.path.join(demo_dir, "processed_data.npz")
    Config.MODEL_CHECKPOINT = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(demo_dir, "submission.csv")

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # ------------------------------------------------------------------
    # 2. Verify Utility Functions
    # ------------------------------------------------------------------
    print("\n[Utils] Verifying utility functions...")

    # Test AUC Computation
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.6, 0.9])
    auc_score = compute_auc(y_true, y_pred)
    print(f"   - Computed AUC: {auc_score:.4f}")
    assert auc_score == 1.0, "AUC calculation incorrect for perfect predictions"

    # Test Checkpointing
    dummy_model = nn.Linear(10, 1)
    dummy_optimizer = optim.SGD(dummy_model.parameters(), lr=0.01)
    dummy_scheduler = optim.lr_scheduler.StepLR(dummy_optimizer, step_size=1)
    ckpt_path = os.path.join(demo_dir, "test_ckpt.pth")

    save_checkpoint(
        dummy_model,
        dummy_optimizer,
        dummy_scheduler,
        epoch=1,
        score=0.85,
        filepath=ckpt_path,
    )
    assert os.path.exists(ckpt_path), "Checkpoint file was not created."

    loaded_ckpt = load_checkpoint(
        ckpt_path, dummy_model, dummy_optimizer, dummy_scheduler
    )
    assert loaded_ckpt["epoch"] == 1
    assert loaded_ckpt["score"] == 0.85
    print("   - Checkpoint save/load verified.")

    # ------------------------------------------------------------------
    # 3. Verify Dataset Loading & Processing
    # ------------------------------------------------------------------
    print("\n[Dataset] Verifying data loading and processing...")
    print("   - This step triggers data processing (normalization/tokenization).")

    # We force load_cached_data=False to demonstrate the processing logic works
    # and to populate our demo cache.
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=32, load_cached_data=False
    )

    # Fetch a single batch
    batch = next(iter(train_loader))
    cont_feats = batch["cont_features"]
    cat_feats = batch["cat_features"]
    targets = batch["target"]

    print(
        f"   - Batch Shapes -> Continuous: {cont_feats.shape}, Categorical: {cat_feats.shape}, Target: {targets.shape}"
    )

    # Assertions
    assert cont_feats.shape == (
        32,
        Config.NUM_CONT_FEATURES,
    ), "Continuous feature shape mismatch"
    assert cat_feats.shape == (32, Config.SEQ_LEN), "Categorical feature shape mismatch"
    assert targets.shape == (32,), "Target shape mismatch"
    assert not torch.isnan(cont_feats).any(), "NaNs found in continuous features"
    print("   - Dataset shapes and content verified.")

    # ------------------------------------------------------------------
    # 4. Verify Model Architecture
    # ------------------------------------------------------------------
    print("\n[Model] Verifying model architecture...")

    device = torch.device(Config.DEVICE)
    model = GatedTransformerResFunnelHybrid().to(device)

    # Move batch to device
    cont_feats = cont_feats.to(device)
    cat_feats = cat_feats.to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(cont_feats, cat_feats)

    print(f"   - Output Logits Shape: {logits.shape}")
    assert logits.shape == (32, 1), "Model output shape mismatch"
    assert not torch.isnan(logits).any(), "Model produced NaNs"
    print("   - Forward pass verified.")

    # ------------------------------------------------------------------
    # 5. Verify Training Loop (Integration Test)
    # ------------------------------------------------------------------
    print("\n[Training] Running short training loop (Debug Mode)...")

    # Train for 2 epochs on a tiny subset of data to ensure pipeline integrity
    # without waiting for full training.
    train_model(
        epochs=2,
        batch_size=32,
        debug=True,
        debug_samples=200,  # Use only 200 samples for speed
    )

    assert os.path.exists(
        Config.MODEL_CHECKPOINT
    ), "Best model checkpoint was not saved."
    print("   - Training loop completed and checkpoint saved.")

    # ------------------------------------------------------------------
    # 6. Verify Submission Generation
    # ------------------------------------------------------------------
    print("\n[Inference] Generating submission...")

    # Use larger batch size for inference speed
    generate_submission(batch_size=1024)

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not generated."

    # Validate submission content
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"   - Submission File Shape: {df_sub.shape}")

    # Expected: 100,000 rows + header.
    # Note: test.csv has 100,001 lines (header + 100k rows).
    assert len(df_sub) == 100000, f"Expected 100,000 predictions, got {len(df_sub)}"
    assert list(df_sub.columns) == ["id", "target"], "Submission columns mismatch"

    # Check value range (probabilities)
    preds = df_sub["target"].values
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    print("   - Submission file verified.")

    print("\n============================================================")
    print("       Demo Completed Successfully!                         ")
    print("============================================================")


if __name__ == "__main__":
    run_demo()
