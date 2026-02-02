import os
import sys
import numpy as np
import torch
import pandas as pd
import cv2
import warnings

# Import from the provided library files
from library.utils import (
    rle_encode,
    rle_decode,
    pad_image,
    unpad_image,
    do_kaggle_metric,
)
from library.dataset import get_loaders
from library.model import ResNet34WideLinkNet
from library.losses import CombinedLoss, LovaszHingeLoss, StableBCELoss
from library.engine import (
    set_seed,
    train_one_epoch,
    evaluate,
    optimize_threshold,
    train_model,
    predict_test,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def test_utils():
    print("\n--- Testing Utils ---")

    # 1. RLE Encode/Decode
    # Create a synthetic mask (101x101) with a square of 1s
    mask = np.zeros((101, 101), dtype=np.uint8)
    mask[10:20, 10:20] = 1

    encoded = rle_encode(mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    assert isinstance(encoded, str), "RLE encode should return a string"
    assert np.array_equal(mask, decoded), "Decoded mask does not match original"
    print("RLE Encode/Decode: PASSED")

    # 2. Pad/Unpad
    # Create random image 101x101
    img = np.random.randint(0, 255, (101, 101), dtype=np.uint8)
    padded = pad_image(img, target_size=(128, 128))

    assert padded.shape == (128, 128), f"Padded shape mismatch: {padded.shape}"

    unpadded = unpad_image(padded, original_size=(101, 101))
    assert unpadded.shape == (101, 101), f"Unpadded shape mismatch: {unpadded.shape}"
    assert np.array_equal(
        img, unpadded
    ), "Unpadded image content mismatch (reflection padding might affect edges if not careful, but center crop should be exact for valid padding)"
    print("Pad/Unpad: PASSED")

    # 3. Metric
    # Perfect match
    pred_perfect = np.zeros((1, 101, 101))
    pred_perfect[0, 10:20, 10:20] = 1
    truth = np.zeros((1, 101, 101))
    truth[0, 10:20, 10:20] = 1

    score = do_kaggle_metric(pred_perfect, truth, threshold=0.5)
    assert np.isclose(score, 1.0), f"Perfect match should have score 1.0, got {score}"

    # No overlap
    pred_bad = np.zeros((1, 101, 101))
    pred_bad[0, 50:60, 50:60] = 1  # Disjoint
    score_bad = do_kaggle_metric(pred_bad, truth, threshold=0.5)
    assert np.isclose(
        score_bad, 0.0
    ), f"No overlap should have score 0.0, got {score_bad}"

    print("Metric Calculation: PASSED")


def test_data_pipeline():
    print("\n--- Testing Data Pipeline ---")

    # Use debug=True to load a small subset (100 images)
    # This uses the metadata files provided in ./metadata/
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=4,
        num_workers=0,  # Use 0 workers for simple debugging to avoid multiprocessing overhead
        load_cached_data=False,  # Force reload to verify processing logic
        debug=True,
    )

    # Fetch one batch
    images, masks, depths, ids = next(iter(train_loader))

    # Assert Shapes
    # Batch size 4, 1 channel (grayscale), 128x128 (padded)
    assert images.shape == (
        4,
        1,
        128,
        128,
    ), f"Image batch shape incorrect: {images.shape}"
    assert masks.shape == (4, 1, 128, 128), f"Mask batch shape incorrect: {masks.shape}"
    assert depths.shape == (4, 1), f"Depth batch shape incorrect: {depths.shape}"
    assert len(ids) == 4, "IDs length mismatch"

    # Check Data Types
    assert images.dtype == torch.float32, "Images should be float tensors"
    assert masks.dtype == torch.float32, "Masks should be float tensors (for BCE)"

    print(f"Data Loader: PASSED (Batch shape: {images.shape})")
    return train_loader, val_loader, test_loader


def test_model_and_loss(device):
    print("\n--- Testing Model and Loss ---")

    # Instantiate Model
    # pretrained=False to avoid downloading weights during this quick test
    model = ResNet34WideLinkNet(pretrained=False).to(device)

    # Create dummy input
    dummy_img = torch.randn(2, 1, 128, 128).to(device)
    dummy_depth = torch.randn(2, 1).to(device)
    dummy_target = torch.randint(0, 2, (2, 1, 128, 128)).float().to(device)

    # Forward Pass
    logits = model(dummy_img, dummy_depth)
    assert logits.shape == (2, 1, 128, 128), f"Output shape mismatch: {logits.shape}"

    # Loss Calculation
    loss_fn = CombinedLoss()
    loss = loss_fn(logits, dummy_target)

    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() > 0, "Loss should be positive"

    # Backward Pass check
    loss.backward()
    assert model.conv1.weight.grad is not None, "Gradients not computed"

    print("Model Forward/Backward & Loss: PASSED")
    return model


def run_training_demo(model, train_loader, val_loader, device):
    print("\n--- Running Training Demo ---")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    save_path = "./working/demo_run/best_model.pth"

    # Train for 1 epoch, max 2 batches to be fast
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epochs=1,
        patience=1,
        save_path=save_path,
        max_batches=2,  # Limit batches for speed
    )

    assert os.path.exists(save_path), "Model checkpoint was not saved"
    print("Training Loop: PASSED")
    return trained_model


def run_inference_demo(model, test_loader, device):
    print("\n--- Running Inference Demo ---")

    submission_path = "./working/demo_run/submission.csv"

    # Predict on test set (limit to 2 batches implicitly by loader size or logic,
    # but predict_test iterates full loader. Since we used debug=True, loader is small.)
    # However, predict_test doesn't have max_batches arg, so we rely on debug loader being small (100 items).
    # 100 items / batch 4 = 25 batches. This is acceptable for 1 hour limit.

    predict_test(model, test_loader, device, threshold=0.5, output_path=submission_path)

    assert os.path.exists(submission_path), "Submission file not created"

    df = pd.read_csv(submission_path)
    assert "id" in df.columns and "rle_mask" in df.columns, "Submission columns missing"
    assert len(df) > 0, "Submission file is empty"

    print("Inference: PASSED")


if __name__ == "__main__":
    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Verify Utils
    test_utils()

    # 3. Verify Data Pipeline
    train_loader, val_loader, test_loader = test_data_pipeline()

    # 4. Verify Model & Loss
    model = test_model_and_loss(device)

    # 5. Run Training Demo
    model = run_training_demo(model, train_loader, val_loader, device)

    # 6. Run Inference Demo
    run_inference_demo(model, test_loader, device)

    print("\nAll demonstrations completed successfully.")
