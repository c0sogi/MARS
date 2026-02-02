import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

# Import library modules
from library.config import Config
from library.utils import set_seed, rle_encode, do_kaggle_metric
from library.dataset import get_dataloaders
from library.model import SaltLinkNet
from library.losses import BCEDiceLoss
from library.trainer import SaltTrainer


def verify_rle_encoding():
    """Verifies the Run-Length Encoding logic with a simple manual case."""
    print("Verifying RLE Encoding...")
    # Create a 3x3 mask
    # 0 1 0
    # 0 1 0
    # 0 0 0
    # Column-major flatten (Transposed):
    # Col 1: 0, 0, 0
    # Col 2: 1, 1, 0
    # Col 3: 0, 0, 0
    # Sequence: 0 0 0 1 1 0 0 0 0
    # Indices (1-based): 4, 5 are 1s.
    # Expected RLE: Start 4, Length 2 -> "4 2"
    mask = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)

    encoded = rle_encode(mask)
    expected = "4 2"

    assert (
        encoded == expected
    ), f"RLE Encoding failed. Expected '{expected}', got '{encoded}'"
    print("RLE Encoding verified.")


def verify_metric():
    """Verifies the Kaggle Metric (mAP @ IoU thresholds) logic."""
    print("Verifying Kaggle Metric...")
    # Case 1: Perfect match
    pred = np.ones((2, 10, 10), dtype=np.uint8)
    truth = np.ones((2, 10, 10), dtype=np.uint8)
    # IoU is 1.0 for all thresholds -> Precision 1.0
    score = do_kaggle_metric(pred, truth)
    assert np.isclose(score, 1.0), f"Metric failed for perfect match. Got {score}"

    # Case 2: No overlap
    pred = np.zeros((2, 10, 10), dtype=np.uint8)
    truth = np.ones((2, 10, 10), dtype=np.uint8)
    # IoU is 0.0 -> Precision 0.0
    score = do_kaggle_metric(pred, truth)
    assert np.isclose(score, 0.0), f"Metric failed for no match. Got {score}"

    print("Kaggle Metric verified.")


def verify_model_and_loss(device):
    """Verifies Model Forward Pass and Loss Calculation."""
    print("Verifying Model and Loss...")
    model = SaltLinkNet().to(device)
    criterion = BCEDiceLoss()

    # Create dummy batch: Batch Size 2, 1 Channel, 128x128 Image
    dummy_img = torch.randn(2, 1, 128, 128).to(device)
    # Dummy depth: Batch Size 2, 1 Value
    dummy_depth = torch.randn(2, 1).to(device)
    # Dummy mask: Batch Size 2, 128x128
    dummy_mask = torch.randint(0, 2, (2, 128, 128)).float().to(device)

    # Forward Pass
    logits = model(dummy_img, dummy_depth)

    # Check Output Shape: (B, 1, H, W)
    assert logits.shape == (
        2,
        1,
        128,
        128,
    ), f"Model output shape mismatch. Got {logits.shape}"

    # Calculate Loss
    loss = criterion(logits, dummy_mask)

    # Check Loss is scalar
    assert loss.dim() == 0, "Loss should be a scalar tensor."
    assert not torch.isnan(loss), "Loss returned NaN."

    print("Model and Loss verified.")


def main():
    # 1. Setup
    print("Initializing Configuration...")
    # Override Config for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8  # Smaller batch size for demo
    Config.setup()
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Verify Utils
    verify_rle_encoding()
    verify_metric()
    verify_model_and_loss(device)

    # 3. Data Loading
    print("\nLoading Data...")
    # Load full datasets
    train_loader_full, val_loader_full, test_loader_full = get_dataloaders(
        load_cached_data=True
    )

    # Create Subsets for fast demonstration (e.g., 2 batches worth of data)
    subset_size = Config.BATCH_SIZE * 2

    train_subset = Subset(train_loader_full.dataset, range(subset_size))
    val_subset = Subset(val_loader_full.dataset, range(subset_size))
    test_subset = Subset(test_loader_full.dataset, range(subset_size))

    print(
        f"Created subsets: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_subset)}"
    )

    # Create new loaders for subsets
    train_loader = DataLoader(
        train_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for small demo
    )
    val_loader = DataLoader(
        val_subset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_subset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # 4. Training Pipeline
    print("\nStarting Training Pipeline Demonstration...")
    trainer = SaltTrainer()

    # A. Fit (Train + Val loop)
    # We expect this to run for 1 epoch on the subset
    trainer.fit(train_loader, val_loader)

    # Check if model saved
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        # If validation score didn't improve (unlikely with random init vs random data, but possible),
        # force save for the sake of the demo prediction step.
        print(
            "Model not saved by EarlyStopping (score didn't improve). Saving manually for demo."
        )
        torch.save(trainer.model.state_dict(), Config.MODEL_SAVE_PATH)
    else:
        print("Model checkpoint verified.")

    # B. Optimize Threshold
    print("\nOptimizing Threshold...")
    best_threshold = trainer.optimize_threshold(val_loader)
    assert 0.0 < best_threshold < 1.0, "Threshold optimization returned invalid value."

    # C. Predict
    print("\nGenerating Predictions...")
    trainer.predict(test_loader, threshold=best_threshold)

    # 5. Verify Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission generated with {len(sub_df)} rows.")

        # Check columns
        assert (
            "id" in sub_df.columns and "rle_mask" in sub_df.columns
        ), "Submission columns mismatch."

        # Check if we have rows corresponding to our subset
        # Note: The trainer.predict iterates the loader. The loader yields subset data.
        # However, the trainer uses `idx_counter` to map to `test_ids` loaded from metadata.
        # Since we are using a subset of the loader (first N images), and the loader is sequential (shuffle=False),
        # the predictions correspond to the first N IDs in the metadata.
        # The submission file should contain N rows.
        assert len(sub_df) == len(
            test_subset
        ), f"Submission row count mismatch. Expected {len(test_subset)}, got {len(sub_df)}"

        print("Submission format verified.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
