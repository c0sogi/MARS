import os
import gc
import torch
import numpy as np
import pandas as pd
from library import config, utils, model, dataset, train, predict


def set_seed(seed=42):
    """Sets seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def verify_utils():
    print("\n=== Verifying library.utils ===")

    # 1. Test RLE Encoding
    # Create a simple 2x2 mask: [[0, 1], [1, 0]]
    # Flattened (row-major): 0, 1, 1, 0
    # Indices (1-based): 2, 3
    # Expected RLE: "2 2" (Start at 2, length 2)
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle_out = utils.rle_encode(mask)
    print(f"RLE Input:\n{mask}\nRLE Output: '{rle_out}'")
    assert rle_out == "2 2", f"RLE encoding mismatch. Expected '2 2', got '{rle_out}'"

    # 2. Test F-beta Score
    # Preds: [High, Low, High, Low] -> Binary(0.5): [1, 0, 1, 0]
    # Targets: [1, 0, 0, 1]
    # TP=1 (Index 0), FP=1 (Index 2), FN=1 (Index 3)
    # Beta=0.5 -> Beta^2=0.25
    # Score = (1.25 * TP) / (1.25*TP + 0.25*FN + FP)
    # Score = 1.25 / (1.25 + 0.25 + 1) = 1.25 / 2.5 = 0.5
    preds = torch.tensor([0.9, 0.1, 0.8, 0.2])
    targets = torch.tensor([1, 0, 0, 1])
    score = utils.fbeta_score(preds, targets, beta=0.5, threshold=0.5)
    print(f"Calculated F0.5 Score: {score}")
    assert abs(score - 0.5) < 1e-6, f"F-beta score mismatch. Expected 0.5, got {score}"

    # 3. Test Threshold Optimization
    # Perfect predictions should yield threshold between 0.2 and 0.8 and score 1.0
    p_perf = torch.tensor([0.2, 0.8])
    t_perf = torch.tensor([0, 1])
    best_th, best_sc = utils.optimize_threshold(p_perf, t_perf)
    print(f"Optimized Threshold: {best_th:.4f}, Score: {best_sc:.4f}")
    assert best_sc == 1.0, "Threshold optimization failed to find perfect score."


def verify_model():
    print("\n=== Verifying library.model ===")

    device = config.DEVICE
    net = model.SFRPNet().to(device)

    # Create dummy input: (Batch=2, Z=65, H=128, W=128)
    # Using smaller spatial dims for speed, valid for fully convolutional architecture
    B, Z, H, W = 2, config.Z_DIM, 128, 128
    dummy_input = torch.randn(B, Z, H, W).to(device)

    print(f"Input Tensor Shape: {dummy_input.shape}")

    with torch.no_grad():
        output = net(dummy_input)

    print(f"Output Tensor Shape: {output.shape}")

    # Expected output: (B, 1, H, W)
    assert output.shape == (
        B,
        1,
        H,
        W,
    ), f"Model output shape mismatch. Expected {(B, 1, H, W)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Model produced NaN values."


def verify_dataset():
    print("\n=== Verifying library.dataset ===")

    # Use a small limit to test loading without processing entire dataset
    limit = 4
    batch_size = 2
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        batch_size=batch_size, limit=limit
    )

    print(f"Train Loader Batches: {len(train_loader)}")

    # Fetch one batch
    batch = next(iter(train_loader))

    required_keys = ["volume", "label", "sample_id", "fragment_id", "x", "y"]
    for k in required_keys:
        assert k in batch, f"Batch missing key: {k}"

    vol = batch["volume"]
    lbl = batch["label"]

    print(f"Batch Volume Shape: {vol.shape}")
    print(f"Batch Label Shape: {lbl.shape}")

    # Verify shapes match config
    # Volume: (Batch, Z_DIM, H, W)
    # Label: (Batch, 1, H, W)
    assert (
        vol.shape[1] == config.Z_DIM
    ), f"Volume depth mismatch. Expected {config.Z_DIM}, got {vol.shape[1]}"
    assert lbl.shape[1] == 1, "Label channel dimension mismatch."


def verify_training_pipeline():
    print("\n=== Verifying library.train ===")

    # Run a minimal training loop
    # limit=8 ensures we have a few batches (batch_size=32 in config, but we override in function call if possible,
    # or rely on the fact that get_dataloaders uses config.BATCH_SIZE.
    # To ensure it runs, we rely on the script provided.
    # Note: config.BATCH_SIZE is 32. If limit=8, DataLoader drop_last=True might result in 0 batches.
    # We must temporarily adjust batch size or limit.
    # Since we cannot modify config.py, we pass batch_size to run_training if supported.
    # library.train.run_training accepts batch_size.

    limit = 8
    batch_size = 2
    epochs = 1

    print(
        f"Running training with limit={limit}, batch_size={batch_size}, epochs={epochs}..."
    )
    train.run_training(epochs=epochs, batch_size=batch_size, limit=limit)

    # Verify Artifacts
    best_model = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    best_thresh = os.path.join(config.WORKING_DIR, "best_threshold.txt")
    submission = config.SUBMISSION_FILE

    assert os.path.exists(best_model), "Checkpoint best_model.pth not found."
    assert os.path.exists(best_thresh), "Threshold file best_threshold.txt not found."
    assert os.path.exists(submission), "Submission file not found after training."

    # Verify Submission Content
    df = pd.read_csv(submission)
    print("Submission Head:")
    print(df.head())
    assert (
        "Id" in df.columns and "Predicted" in df.columns
    ), "Submission columns invalid."


def verify_inference_pipeline():
    print("\n=== Verifying library.predict ===")

    # Rename previous submission to ensure new one is generated
    if os.path.exists(config.SUBMISSION_FILE):
        os.rename(config.SUBMISSION_FILE, config.SUBMISSION_FILE + ".bak")

    limit = 4
    print(f"Running inference with limit={limit}...")

    predict.run_inference(limit=limit)

    assert os.path.exists(
        config.SUBMISSION_FILE
    ), "Inference failed to generate submission.csv"
    print("Inference completed successfully.")


if __name__ == "__main__":
    # Force garbage collection and clear CUDA cache to release zombie memory from previous runs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Initialize environment
    config.setup_directories()
    set_seed(config.SEED)

    # Execute Verifications
    verify_utils()
    verify_model()
    verify_dataset()
    verify_training_pipeline()
    verify_inference_pipeline()

    print("\nAll verifications passed successfully.")
