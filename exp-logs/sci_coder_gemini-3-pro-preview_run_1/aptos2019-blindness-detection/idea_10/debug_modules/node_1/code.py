import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import library components
from library.config import Config, seed_everything
from library.utils import quadratic_weighted_kappa, ModelEMA
from library.dataset import get_loaders, mixup_data, RetinaDataset
from library.model import MultiScaleConvNeXt
from library.train import run_training, get_ordinal_targets, train_fn


def demo_utils():
    print("=== Demo: Utils ===")

    # 1. Verify Quadratic Weighted Kappa
    # Case A: Perfect Agreement
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4])
    score = quadratic_weighted_kappa(y_true, y_pred)
    print(f"QWK (Perfect Agreement): {score:.4f}")
    assert np.isclose(score, 1.0), "QWK should be 1.0 for perfect agreement"

    # Case B: Complete Disagreement (Inverse)
    y_pred_bad = np.array([4, 3, 2, 1, 0])
    score_bad = quadratic_weighted_kappa(y_true, y_pred_bad)
    print(f"QWK (Inverse Agreement): {score_bad:.4f}")
    assert score_bad < 1.0, "QWK should be less than 1.0 for disagreement"

    print("Utils verification passed.\n")


def demo_dataset():
    print("=== Demo: Dataset ===")

    # Override Config for speed and debugging
    Config.debug = True
    Config.image_size = 256  # Smaller size for faster processing
    Config.batch_size = 4
    Config.num_workers = 0  # Avoid multiprocessing overhead in demo

    print("Configured for Debug Mode: Image Size 256, Batch Size 4")

    # 1. Get DataLoaders
    # load_cached_data=False forces the dataset to process images from disk,
    # verifying the reading and resizing logic.
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches:   {len(val_loader)}")
    print(f"Test Loader Batches:  {len(test_loader)}")

    # 2. Fetch a single batch
    images, targets = next(iter(train_loader))

    print(f"Images Shape: {images.shape}")  # Expected: [4, 3, 256, 256]
    print(f"Targets Shape: {targets.shape}")  # Expected: [4]
    print(f"Targets Dtype: {targets.dtype}")  # Expected: torch.int64

    # Assertions
    assert images.shape == (4, 3, 256, 256), f"Unexpected image shape: {images.shape}"
    assert targets.shape == (4,), f"Unexpected target shape: {targets.shape}"
    assert targets.dtype == torch.long, "Targets must be of type torch.long"

    # 3. Test Mixup
    mixed_x, y_a, y_b, lam = mixup_data(images, targets, alpha=1.0, device="cpu")
    assert mixed_x.shape == images.shape, "Mixup altered image shape incorrectly"
    assert y_a.shape == targets.shape, "Mixup altered target shape incorrectly"

    print("Dataset verification passed.\n")
    return train_loader


def demo_model(loader):
    print("=== Demo: Model ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Config.device = device

    # 1. Instantiate Model
    # We use pretrained=False to avoid downloading weights during this quick demo,
    # but the architecture logic remains the same.
    model = MultiScaleConvNeXt(pretrained=False)
    model.to(device)

    # 2. Forward Pass
    images, _ = next(iter(loader))
    images = images.to(device)

    logits = model(images)
    print(f"Logits Shape: {logits.shape}")

    # Assertions
    # Output should be [Batch Size, Num Ordinal Heads] -> [4, 4]
    assert logits.shape == (4, 4), f"Expected logits shape (4, 4), got {logits.shape}"

    # 3. Test EMA Initialization
    ema = ModelEMA(model)
    ema.update(model)
    print("ModelEMA initialized and updated successfully.")

    print("Model verification passed.\n")
    return model


def demo_training_components(model, loader):
    print("=== Demo: Training Components ===")
    device = Config.device

    # 1. Test Ordinal Target Conversion
    # Targets: [0, 2, 4] -> Ordinal vectors (thresholds 0, 1, 2, 3)
    # 0 -> [0, 0, 0, 0] ( >0:F, >1:F, >2:F, >3:F )
    # 2 -> [1, 1, 0, 0] ( >0:T, >1:T, >2:F, >3:F )
    # 4 -> [1, 1, 1, 1] ( >0:T, >1:T, >2:T, >3:T )
    dummy_targets = torch.tensor([0, 2, 4], device=device)
    ordinal_targets = get_ordinal_targets(dummy_targets, num_classes=5, device=device)

    expected_ordinal = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
        device=device,
    )

    assert torch.allclose(
        ordinal_targets, expected_ordinal
    ), f"Ordinal target conversion failed.\nGot:\n{ordinal_targets}\nExpected:\n{expected_ordinal}"
    print("Ordinal target conversion verified.")

    # 2. Run a single training step manually
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    print("Executing single training epoch (subset)...")
    loss = train_fn(model, None, loader, criterion, optimizer, device, epoch=0)
    print(f"Training Step Loss: {loss:.4f}")
    assert not np.isnan(loss), "Training loss resulted in NaN"

    print("Training components verification passed.\n")


def demo_full_pipeline():
    print("=== Demo: Full Pipeline Integration ===")

    # Configure for a minimal full run
    Config.epochs = 1
    Config.use_ema = False  # Disable EMA to save memory/time for this check

    # run_training handles the entire loop: Train -> Val -> Save Model -> Inference -> Submission
    # We pass load_cached_data=True to use the cache generated in demo_dataset()
    try:
        run_training(epochs=1, load_cached_data=True)
    except Exception as e:
        print(f"Training run failed with error: {e}")
        raise e

    # Verify Submission File
    submission_path = "./submission/submission.csv"
    if os.path.exists(submission_path):
        df = pd.read_csv(submission_path)
        print(f"Submission generated at {submission_path}")
        print(df.head())
        assert len(df) > 0, "Submission file is empty"
        assert (
            "id_code" in df.columns and "diagnosis" in df.columns
        ), "Invalid submission columns"
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("Full pipeline verification passed.\n")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)

    # Run demonstrations
    demo_utils()
    train_loader = demo_dataset()
    model = demo_model(train_loader)
    demo_training_components(model, train_loader)
    demo_full_pipeline()

    print("All demonstrations completed successfully.")
