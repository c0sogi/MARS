import os
import shutil
import torch
import pandas as pd
import numpy as np
from library.utils import (
    set_seed,
    AverageMeter,
    calculate_map5,
    save_checkpoint,
    load_checkpoint,
)
from library.data_loader import get_dataloaders, LabelEncoder, WhaleDataset
from library.model import WhaleClassifier
from library.train import train_model
from library.inference import generate_submission

# Configuration
INPUT_DIR = "./input"
ORIGINAL_METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
DEMO_METADATA_DIR = os.path.join(WORKING_DIR, "demo_metadata")
DEMO_CACHE_DIR = os.path.join(WORKING_DIR, "demo_cache")
DEMO_CHECKPOINT_DIR = os.path.join(WORKING_DIR, "demo_checkpoints")
SUBMISSION_FILE = os.path.join(WORKING_DIR, "submission", "demo_submission.csv")
INFERENCE_SUBMISSION_FILE = os.path.join(
    WORKING_DIR, "submission", "inference_submission.csv"
)


def setup_demo_environment():
    """Creates a mini subset of metadata for fast execution."""
    print("\n[Setup] Creating mini metadata for demonstration...")
    os.makedirs(DEMO_METADATA_DIR, exist_ok=True)
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)
    os.makedirs(DEMO_CHECKPOINT_DIR, exist_ok=True)

    # Read original metadata
    train_df = pd.read_csv(os.path.join(ORIGINAL_METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(ORIGINAL_METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(ORIGINAL_METADATA_DIR, "test.csv"))

    # Create subsets (ensure enough samples for batch_size=4)
    # Strategy: Ensure validation classes exist in training subset to avoid KeyError

    # 1. Ensure we select at least 5 distinct classes to satisfy MAP@5 topk(5) requirement (Cite debug_lesson_1)
    # We select 5 classes from the training set.
    target_classes = train_df["Id"].unique()[:5]

    # 2. Create training subset with these classes
    # Take up to 2 samples per class to ensure representation
    train_forced = train_df[train_df["Id"].isin(target_classes)].groupby("Id").head(2)

    # Fill the rest of the subset with other samples to reach ~20 total
    train_others = train_df[~train_df["Id"].isin(target_classes)]
    train_subset = pd.concat([train_forced, train_others]).head(20)

    # 3. Create validation subset
    # Ensure validation classes are a subset of training classes
    final_train_classes = set(train_subset["Id"].unique())
    val_subset = val_df[val_df["Id"].isin(final_train_classes)].head(10)

    # Fallback: If validation set ends up empty (e.g. classes are singletons), borrow from train
    if len(val_subset) == 0:
        val_subset = train_subset.iloc[:5].copy()

    test_subset = test_df.head(10)

    # Save to demo directory
    train_subset.to_csv(os.path.join(DEMO_METADATA_DIR, "train.csv"), index=False)
    val_subset.to_csv(os.path.join(DEMO_METADATA_DIR, "val.csv"), index=False)
    test_subset.to_csv(os.path.join(DEMO_METADATA_DIR, "test.csv"), index=False)
    print("[Setup] Mini metadata created.")


def test_utils():
    print("\n[Test] Verifying Utility Functions...")

    # 1. Test AverageMeter
    meter = AverageMeter()
    meter.update(10, n=1)
    meter.update(20, n=1)
    assert meter.avg == 15.0, f"AverageMeter failed: expected 15.0, got {meter.avg}"
    assert meter.count == 2, f"AverageMeter count failed: expected 2, got {meter.count}"
    print("  -> AverageMeter passed.")

    # 2. Test calculate_map5
    # Case 1: Target is the first prediction (Rank 0) -> Score 1.0
    preds = torch.tensor([[0, 1, 2, 3, 4]])
    targets = torch.tensor([0])
    score = calculate_map5(preds, targets)
    assert np.isclose(score, 1.0), f"MAP@5 failed for rank 0: got {score}"

    # Case 2: Target is the second prediction (Rank 1) -> Score 1/2 = 0.5
    preds = torch.tensor([[0, 1, 2, 3, 4]])
    targets = torch.tensor([1])
    score = calculate_map5(preds, targets)
    assert np.isclose(score, 0.5), f"MAP@5 failed for rank 1: got {score}"

    # Case 3: Target not in predictions -> Score 0.0
    preds = torch.tensor([[0, 1, 2, 3, 4]])
    targets = torch.tensor([99])
    score = calculate_map5(preds, targets)
    assert np.isclose(score, 0.0), f"MAP@5 failed for miss: got {score}"

    print("  -> calculate_map5 passed.")


def test_data_loader():
    print("\n[Test] Verifying Data Loading...")

    # Use the mini metadata created in setup
    batch_size = 4
    image_size = 64  # Small size for speed

    train_loader, val_loader, test_loader, label_encoder = get_dataloaders(
        data_dir=INPUT_DIR,
        metadata_dir=DEMO_METADATA_DIR,
        batch_size=batch_size,
        num_workers=0,  # 0 for main thread debugging
        load_cached_data=False,
        cache_dir=DEMO_CACHE_DIR,
        image_size=image_size,
    )

    # Verify LabelEncoder
    num_classes = label_encoder.num_classes()
    assert num_classes > 0, "LabelEncoder found 0 classes."
    print(f"  -> LabelEncoder found {num_classes} classes.")

    # Verify Train Loader Batch
    images, labels = next(iter(train_loader))
    assert images.shape == (
        batch_size,
        3,
        image_size,
        image_size,
    ), f"Train image shape mismatch. Expected {(batch_size, 3, image_size, image_size)}, got {images.shape}"
    assert labels.shape == (batch_size,), f"Train label shape mismatch."
    print("  -> Train DataLoader batch shape verified.")

    # Verify Test Loader Batch
    test_images, filenames = next(iter(test_loader))
    assert test_images.shape == (
        batch_size,
        3,
        image_size,
        image_size,
    ), "Test image shape mismatch."
    assert len(filenames) == batch_size, "Test filename list length mismatch."
    print("  -> Test DataLoader batch shape verified.")


def test_model_components():
    print("\n[Test] Verifying Model Components...")

    num_classes = 5
    model = WhaleClassifier(num_classes=num_classes)

    # 1. Forward Pass
    dummy_input = torch.randn(2, 3, 224, 224)  # Batch size 2
    output = model(dummy_input)
    assert output.shape == (
        2,
        num_classes,
    ), f"Model output shape mismatch. Expected (2, {num_classes}), got {output.shape}"
    print("  -> Model forward pass verified.")

    # 2. Checkpoint Save/Load
    checkpoint_path = os.path.join(DEMO_CHECKPOINT_DIR, "test_ckpt.pth.tar")
    save_checkpoint(
        {"state_dict": model.state_dict(), "epoch": 1, "best_score": 0.5},
        is_best=True,
        checkpoint_dir=DEMO_CHECKPOINT_DIR,
        filename="test_ckpt.pth.tar",
    )

    assert os.path.exists(checkpoint_path), "Checkpoint file not created."

    # Load back
    model_new = WhaleClassifier(num_classes=num_classes)
    epoch, best_score = load_checkpoint(checkpoint_path, model_new)
    assert epoch == 1, "Checkpoint loading (epoch) failed."
    assert best_score == 0.5, "Checkpoint loading (best_score) failed."

    # Verify weights match
    for p1, p2 in zip(model.parameters(), model_new.parameters()):
        assert torch.equal(p1, p2), "Model weights do not match after loading."

    print("  -> Checkpoint save/load verified.")


def test_full_pipeline():
    print("\n[Test] Verifying Full Training & Inference Pipeline...")

    # Clean up previous artifacts to prevent loading stale checkpoints
    if os.path.exists(DEMO_CHECKPOINT_DIR):
        shutil.rmtree(DEMO_CHECKPOINT_DIR)
    os.makedirs(DEMO_CHECKPOINT_DIR, exist_ok=True)

    # 1. Train Model
    # We use very few epochs and limit batches per epoch for speed
    print("  -> Starting training (1 epoch, max 2 batches)...")
    train_model(
        data_dir=INPUT_DIR,
        metadata_dir=DEMO_METADATA_DIR,
        working_dir=DEMO_CHECKPOINT_DIR,
        submission_file=SUBMISSION_FILE,
        epochs=1,
        batch_size=4,
        lr=1e-4,
        patience=1,
        image_size=128,
        num_workers=0,
        load_cached_data=False,  # Force re-encoding for the mini dataset
        max_batches_per_epoch=2,
    )

    assert os.path.exists(
        SUBMISSION_FILE
    ), "Training pipeline failed to generate submission file."
    df_sub = pd.read_csv(SUBMISSION_FILE)
    assert not df_sub.empty, "Generated submission file is empty."
    assert (
        "Image" in df_sub.columns and "Id" in df_sub.columns
    ), "Submission file missing required columns."
    print("  -> Training pipeline completed successfully.")

    # 2. Inference from Checkpoint
    # Use the best model saved during the training step above
    best_model_path = os.path.join(DEMO_CHECKPOINT_DIR, "model_best.pth.tar")

    print("  -> Starting standalone inference...")
    generate_submission(
        checkpoint_path=best_model_path,
        metadata_dir=DEMO_METADATA_DIR,
        data_dir=INPUT_DIR,
        output_file=INFERENCE_SUBMISSION_FILE,
        batch_size=4,
        num_workers=0,
        image_size=128,
        device="cpu",  # Force CPU for simple testing if needed, or let it auto-detect
        cache_dir=DEMO_CHECKPOINT_DIR,  # train_model saves cache to working_dir
        max_test_samples=5,
    )

    assert os.path.exists(
        INFERENCE_SUBMISSION_FILE
    ), "Inference pipeline failed to generate submission file."
    df_inf = pd.read_csv(INFERENCE_SUBMISSION_FILE)
    assert len(df_inf) <= 5, "Inference max_test_samples limit failed."
    print("  -> Inference pipeline completed successfully.")


if __name__ == "__main__":
    set_seed(42)

    try:
        setup_demo_environment()
        test_utils()
        test_data_loader()
        test_model_components()
        test_full_pipeline()
        print("\nAll demonstrations and verifications passed successfully!")
    except Exception as e:
        print(f"\nFAILED: {e}")
        raise e
