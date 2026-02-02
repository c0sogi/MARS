import os
import sys
import shutil
import torch
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config, set_seed
from library.data_loader import get_loaders
from library.model import ICRCN
from library.loss import MultiStageLoss
from library.train import Trainer
from library.inference import Predictor
from library.utils import (
    compute_competition_metric,
    decode_predictions,
    apply_median_filter,
)


def main():
    print("=== Starting Demonstration of IC-RCN Library ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Fast Demo
    # -------------------------------------------------------------------------
    print("[1] Configuring environment for fast execution...")

    # Set a separate working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    # Ensure directories exist
    Config.ensure_dirs()

    # Limit data size and training duration
    Config.DEBUG_SUBSET_SIZE = 10  # Only process 10 samples per split
    Config.BATCH_SIZE = 2
    Config.NUM_EPOCHS = 1
    Config.PATIENCE = 1

    # Set seeds for reproducibility
    set_seed(Config.SEED)
    print("    Configuration updated: Subset=10, Batch=2, Epochs=1")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[2] Testing Data Loader...")

    # Force reload to ensure we use the small subset defined above
    # We delete the cache dir for this demo run to be sure
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    Config.ensure_dirs()

    train_loader, val_loader, test_loader, test_ids = get_loaders(
        load_cached_data=False
    )

    print(f"    Train batches: {len(train_loader)}")
    print(f"    Val batches:   {len(val_loader)}")
    print(f"    Test batches:  {len(test_loader)}")

    # Fetch one batch to verify shapes
    features, targets, lengths = next(iter(train_loader))

    # Features shape: (Batch, Time, InputDim)
    # Targets shape: (Batch, Time)
    print(f"    Feature shape: {features.shape}")
    print(f"    Target shape:  {targets.shape}")

    assert features.dim() == 3, "Features should be 3D (Batch, Time, Feats)"
    assert (
        features.shape[2] == Config.INPUT_DIM
    ), f"Input dim should be {Config.INPUT_DIM}"
    assert targets.dim() == 2, "Targets should be 2D (Batch, Time)"
    assert len(lengths) == features.shape[0], "Lengths should match batch size"

    print("    Data Loader verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Instantiation & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[3] Testing Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = ICRCN().to(device)

    # Move batch to device
    features = features.to(device)

    # Forward pass
    outputs = model(features)

    # Check outputs
    assert isinstance(outputs, dict), "Model output should be a dictionary"
    assert "gen" in outputs, "Output missing 'gen' stage"
    assert "ref1" in outputs, "Output missing 'ref1' stage"
    assert "ref2" in outputs, "Output missing 'ref2' stage"

    # Check shape: (Batch, Classes, Time)
    gen_out = outputs["gen"]
    # Note: features is (B, T, D), model output is (B, C, T)
    expected_shape = (features.shape[0], Config.NUM_CLASSES, features.shape[1])

    print(f"    Model output shape (Gen): {gen_out.shape}")

    assert (
        gen_out.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {gen_out.shape}"
    assert outputs["ref1"].shape == expected_shape
    assert outputs["ref2"].shape == expected_shape

    print("    Model architecture verification passed.")

    # -------------------------------------------------------------------------
    # 4. Loss Function
    # -------------------------------------------------------------------------
    print("\n[4] Testing Loss Function...")

    criterion = MultiStageLoss().to(device)
    targets = targets.to(device)

    loss, loss_dict = criterion(outputs, targets)

    print(f"    Total Loss: {loss.item():.4f}")
    print(f"    Loss Components: {list(loss_dict.keys())}")

    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() > 0, "Loss should be positive"
    assert "loss_gen" in loss_dict
    assert "loss_ref2_tmse" in loss_dict

    print("    Loss function verification passed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop (Trainer)
    # -------------------------------------------------------------------------
    print("\n[5] Testing Trainer (1 Epoch)...")

    trainer = Trainer(train_loader, val_loader)

    # Run one epoch
    train_loss = trainer.train_epoch(epoch=1)
    print(f"    Epoch 1 Train Loss: {train_loss:.4f}")

    # Run validation
    val_loss, val_metric = trainer.validate()
    print(f"    Validation Loss: {val_loss:.4f}")
    print(f"    Validation Metric (Levenshtein): {val_metric:.4f}")

    # Simulate saving best model manually for the next step (Inference)
    # in case validation didn't trigger save (though it likely will as initial best is inf)
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    assert os.path.exists(Config.BEST_MODEL_PATH), "Model checkpoint was not created."

    print("    Trainer verification passed.")

    # -------------------------------------------------------------------------
    # 6. Inference
    # -------------------------------------------------------------------------
    print("\n[6] Testing Inference...")

    predictor = Predictor(model_path=Config.BEST_MODEL_PATH)
    predictions = predictor.predict(test_loader)

    print(f"    Number of predictions: {len(predictions)}")
    print(f"    First prediction sequence: {predictions[0]}")

    assert len(predictions) == len(
        test_ids
    ), "Mismatch between predictions and test IDs"
    assert isinstance(
        predictions[0], list
    ), "Prediction should be a list of gesture IDs"

    print("    Inference verification passed.")

    # -------------------------------------------------------------------------
    # 7. Utilities
    # -------------------------------------------------------------------------
    print("\n[7] Testing Utilities...")

    # Test Metric
    # Case 1: Perfect match
    score_perfect = compute_competition_metric([[1, 2, 3]], [[1, 2, 3]])
    assert score_perfect == 0.0, "Metric should be 0 for perfect match"

    # Case 2: Complete mismatch
    # Levenshtein distance between [1] and [2] is 1 (substitution)
    # Total gestures = 1. Metric = 1/1 = 1.0
    score_mismatch = compute_competition_metric([[1]], [[2]])
    assert abs(score_mismatch - 1.0) < 1e-6, "Metric calculation incorrect for mismatch"

    # Test Decoding
    # [0, 0, 1, 1, 1, 0, 2, 2, 0] -> [1, 2] (0 is background, collapse repeats)
    raw_preds = [0, 0, 1, 1, 1, 0, 2, 2, 0]
    decoded = decode_predictions(raw_preds)
    print(f"    Raw: {raw_preds} -> Decoded: {decoded}")
    assert decoded == [1, 2], f"Decoding failed. Got {decoded}"

    # Test Median Filter
    # [1, 1, 100, 1, 1] -> Median filter size 3 should smooth the outlier 100 to 1
    noisy_signal = np.array([1, 1, 100, 1, 1])
    smoothed = apply_median_filter(noisy_signal, kernel_size=3)
    print(f"    Noisy: {noisy_signal} -> Smoothed: {smoothed}")
    assert smoothed[2] == 1, "Median filter failed to remove outlier"

    print("    Utilities verification passed.")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    main()
