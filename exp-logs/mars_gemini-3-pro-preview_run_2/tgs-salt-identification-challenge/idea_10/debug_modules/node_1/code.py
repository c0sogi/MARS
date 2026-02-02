import os
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import warnings
import random

# Import from the provided library
from library.config import SEED, DEVICE, IMG_SIZE, ORIG_SIZE, WORKING_DIR
from library.utils import rle_encode, rle_decode, do_kaggle_metric, save_checkpoint
from library.dataset import SaltDataset
from library.model import DepthRobustLinkNet
from library.losses import CombinedLoss
from library.train import train_one_epoch, validate

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_utils():
    print("\n=== Testing Utilities ===")

    # 1. Test RLE Encoding/Decoding
    # Create a dummy mask (101x101) with a square of 1s
    mask = np.zeros((ORIG_SIZE, ORIG_SIZE), dtype=np.uint8)
    mask[10:20, 10:20] = 1

    encoded = rle_encode(mask)
    decoded = rle_decode(encoded, shape=(ORIG_SIZE, ORIG_SIZE))

    assert isinstance(encoded, str), "RLE encode should return a string"
    assert np.array_equal(mask, decoded), "Decoded mask does not match original mask"
    print("RLE Encode/Decode: PASSED")

    # 2. Test Kaggle Metric (IoU mAP)
    # Case A: Perfect match
    pred_perfect = mask.copy()
    score_perfect = do_kaggle_metric(pred_perfect, mask, threshold=0.5)
    # Note: If pred matches truth exactly, IoU is 1.0 for all thresholds, so mAP is 1.0
    assert np.isclose(
        score_perfect, 1.0
    ), f"Perfect match score should be 1.0, got {score_perfect}"

    # Case B: No overlap
    pred_empty = np.zeros_like(mask)
    # If truth has object but pred is empty -> IoU is 0. Score should be 0.
    score_empty = do_kaggle_metric(pred_empty, mask, threshold=0.5)
    assert np.isclose(
        score_empty, 0.0
    ), f"No overlap score should be 0.0, got {score_empty}"

    print("Kaggle Metric Calculation: PASSED")


def test_dataset():
    print("\n=== Testing Dataset ===")

    # Initialize dataset in train mode
    # This triggers caching, which might take a moment on first run
    ds = SaltDataset(mode="train", load_cached_data=True)

    assert len(ds) > 0, "Dataset should not be empty"

    # Fetch one sample
    idx = 0
    image, mask, depth, img_id = ds[idx]

    # Verify Shapes
    # Image: (1, 128, 128) - Channel first, padded size
    assert image.shape == (
        1,
        IMG_SIZE,
        IMG_SIZE,
    ), f"Image shape mismatch: {image.shape}"
    # Mask: (1, 128, 128)
    assert mask.shape == (1, IMG_SIZE, IMG_SIZE), f"Mask shape mismatch: {mask.shape}"
    # Depth: (1,)
    assert depth.shape == (1,), f"Depth shape mismatch: {depth.shape}"

    # Verify Types
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert isinstance(depth, torch.Tensor)
    assert isinstance(img_id, str) or isinstance(img_id, np.str_), "ID should be string"

    # Verify Normalization/Values
    # Mask should be float 0.0 or 1.0 (or in between if interpolation happened, but mostly binary logic)
    assert mask.max() <= 1.0 and mask.min() >= 0.0, "Mask values out of range [0, 1]"

    print(f"Dataset Load & Preprocessing: PASSED (Sample ID: {img_id})")
    return ds


def test_model_and_loss(dataset):
    print("\n=== Testing Model and Loss ===")

    # Create a small DataLoader
    batch_size = 4
    # Subset dataset for speed
    subset_indices = list(range(batch_size))
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=batch_size)

    # Get a batch
    images, masks, depths, _ = next(iter(loader))
    images = images.to(DEVICE)
    masks = masks.to(DEVICE)
    depths = depths.to(DEVICE)

    # Instantiate Model
    model = DepthRobustLinkNet(in_channels=1, n_classes=1).to(DEVICE)

    # 1. Forward Pass
    logits = model(images, depths)

    # Output shape should be (B, 1, 128, 128)
    expected_shape = (batch_size, 1, IMG_SIZE, IMG_SIZE)
    assert (
        logits.shape == expected_shape
    ), f"Model output shape mismatch. Got {logits.shape}, expected {expected_shape}"
    print("Model Forward Pass: PASSED")

    # 2. Loss Calculation
    criterion = CombinedLoss(bce_weight=0.5, lovasz_weight=0.5)
    loss = criterion(logits, masks)

    assert torch.is_tensor(loss), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    print("Loss Calculation: PASSED")

    # 3. Backward Pass (Gradient Check)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()

    # Check if gradients are populated
    # Check the first layer's weights
    first_layer_grad = model.initial[0].weight.grad
    assert first_layer_grad is not None, "Gradients not computed for first layer"
    assert torch.norm(first_layer_grad) > 0, "Gradients are zero"

    optimizer.step()
    print("Backward Pass & Optimization Step: PASSED")

    return model, loader, criterion, optimizer


def test_training_pipeline(model, loader, criterion, optimizer):
    print("\n=== Testing Training Pipeline ===")

    # 1. Train One Epoch
    # We use the small subset loader created previously
    avg_train_loss = train_one_epoch(model, loader, criterion, optimizer, DEVICE)

    assert isinstance(avg_train_loss, float), "Train loss should be a float"
    print(f"Train One Epoch: PASSED (Loss: {avg_train_loss:.4f})")

    # 2. Validate
    # Using the same loader for validation just to test the function mechanics
    avg_val_loss, val_score, best_thresh = validate(model, loader, criterion, DEVICE)

    assert isinstance(avg_val_loss, float), "Val loss should be a float"
    assert 0.0 <= val_score <= 1.0, "Validation score (mAP) out of range"
    assert 0.3 <= best_thresh <= 0.75, "Best threshold out of search range"

    print(
        f"Validation Loop: PASSED (Loss: {avg_val_loss:.4f}, Score: {val_score:.4f}, Threshold: {best_thresh})"
    )

    # 3. Checkpoint Saving
    ckpt_path = os.path.join(WORKING_DIR, "test_checkpoint.pth")
    save_checkpoint(model, optimizer, epoch=1, score=val_score, path=ckpt_path)

    assert os.path.exists(ckpt_path), "Checkpoint file was not created"

    # Verify content
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert ckpt["score"] == val_score
    print("Checkpoint Saving: PASSED")


if __name__ == "__main__":
    set_seed(SEED)

    # Ensure working directory exists (though config usually does this)
    os.makedirs(WORKING_DIR, exist_ok=True)

    try:
        # Run verification steps
        test_utils()

        # Load dataset
        dataset = test_dataset()

        # Test Model & Loss
        model, loader, criterion, optimizer = test_model_and_loss(dataset)

        # Test Training Loop Functions
        test_training_pipeline(model, loader, criterion, optimizer)

        print("\nAll verification steps completed successfully.")

    except AssertionError as e:
        print(f"\nVERIFICATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
