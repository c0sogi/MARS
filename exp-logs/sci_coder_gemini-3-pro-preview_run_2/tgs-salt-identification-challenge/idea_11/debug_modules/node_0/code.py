import os
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset

# Import provided library modules
import library.utils as utils
from library.dataset import SaltDataset
from library.model import FiLMResNet34
from library.loss import BCELovaszLoss
from library.trainer import SaltTrainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_utils():
    print("Testing Utils...")
    # Test RLE Encode
    # Create a 2x2 mask:
    # [[0, 1],
    #  [0, 0]]
    # Flattened Fortran-style (col-major): [0, 0, 1, 0]
    # Run should start at index 3 (1-based) with length 1.
    mask = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    rle = utils.rle_encode(mask)
    assert rle == "3 1", f"Expected '3 1', got '{rle}'"

    # Test IoU Batch
    # Pred: [1, 1], True: [1, 0] -> Intersection 1, Union 2 -> IoU 0.5
    y_pred = torch.tensor([[[1.0, 1.0]]])  # Shape (1, 1, 2)
    y_true = torch.tensor([[[1.0, 0.0]]])  # Shape (1, 1, 2)
    iou = utils.calculate_iou_batch(y_pred, y_true, threshold=0.5)
    assert np.isclose(iou[0], 0.5), f"Expected IoU 0.5, got {iou[0]}"
    print("Utils verified.")


def test_dataset():
    print("Testing Dataset...")
    # Initialize dataset
    ds = SaltDataset(mode="train", load_cached=True)

    # Get one item
    image, mask, depth, id_code = ds[0]

    # Verify shapes
    # Image: (1, 128, 128) - due to padding in transforms
    assert image.shape == (1, 128, 128), f"Image shape mismatch: {image.shape}"
    # Mask: (1, 128, 128)
    assert mask.shape == (1, 128, 128), f"Mask shape mismatch: {mask.shape}"
    # Depth: (1,)
    assert depth.shape == (1,), f"Depth shape mismatch: {depth.shape}"

    # Verify values
    assert isinstance(id_code, str)
    assert mask.max() <= 1.0 and mask.min() >= 0.0
    print("Dataset verified.")


def test_model_and_loss():
    print("Testing Model and Loss...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate Model
    model = FiLMResNet34(num_classes=1, pretrained=False).to(device)

    # Create dummy input
    B, C, H, W = 2, 1, 128, 128
    dummy_img = torch.randn(B, C, H, W).to(device)
    dummy_depth = torch.randn(B, 1).to(device)

    # Forward pass
    logits = model(dummy_img, dummy_depth)
    assert logits.shape == (B, 1, H, W), f"Output shape mismatch: {logits.shape}"

    # Instantiate Loss
    criterion = BCELovaszLoss()
    dummy_targets = torch.randint(0, 2, (B, 1, H, W)).float().to(device)

    # Calculate Loss
    loss = criterion(logits, dummy_targets)
    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    print("Model and Loss verified.")


def run_training_demo():
    print("Running Training Demo...")

    # Initialize Trainer with minimal epochs
    trainer = SaltTrainer(epochs=1, batch_size=4, patience=1)

    # OPTIMIZATION FOR SPEED:
    # Replace full datasets with tiny subsets to finish quickly
    subset_indices = list(range(10))  # Use first 10 samples

    trainer.train_dataset = Subset(trainer.train_dataset, subset_indices)
    trainer.val_dataset = Subset(trainer.val_dataset, subset_indices)
    trainer.test_dataset = Subset(trainer.test_dataset, list(range(5)))

    # Re-initialize loaders with subsets
    trainer.train_loader = DataLoader(
        trainer.train_dataset, batch_size=4, shuffle=True, num_workers=0
    )
    trainer.val_loader = DataLoader(
        trainer.val_dataset, batch_size=4, shuffle=False, num_workers=0
    )
    trainer.test_loader = DataLoader(
        trainer.test_dataset, batch_size=4, shuffle=False, num_workers=0
    )

    # Run Training
    best_thresh = trainer.train()

    # Check if model checkpoint exists
    assert os.path.exists(trainer.checkpoint_path), "Model checkpoint was not created."

    # Run Prediction
    trainer.predict_test(best_thresh)

    # Check if submission exists
    sub_path = os.path.join(trainer.submission_dir, "submission.csv")
    assert os.path.exists(sub_path), "Submission file was not created."

    # Validate submission format
    df = pd.read_csv(sub_path)
    assert "id" in df.columns and "rle_mask" in df.columns
    assert len(df) == 5, f"Expected 5 predictions, got {len(df)}"

    print("Training demo completed successfully.")


if __name__ == "__main__":
    set_seed(42)

    # 1. Verify utility functions
    test_utils()

    # 2. Verify dataset loading and transforms
    test_dataset()

    # 3. Verify model architecture and loss calculation
    test_model_and_loss()

    # 4. Run a fast training and inference loop
    run_training_demo()
