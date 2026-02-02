import os
import torch
import pandas as pd
import numpy as np
import shutil
from library.config import Config, seed_everything
from library.utils import AverageMeter, apk, mapk
from library.dataset import HotelDataset, get_transforms
from library.model import HotelIdModel
from library.trainer import Trainer


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Override
    # -------------------------------------------------------------------------
    print("Initializing Demo...")
    seed_everything(42)

    # Define paths for demo data
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    demo_train_path = os.path.join(demo_dir, "train_subset.csv")
    demo_val_path = os.path.join(demo_dir, "val_subset.csv")
    demo_test_path = os.path.join(demo_dir, "test_subset.csv")

    # -------------------------------------------------------------------------
    # 2. Create Data Subsets (Optimization for Speed)
    # -------------------------------------------------------------------------
    print("Creating data subsets...")

    # Load original metadata
    full_train = pd.read_csv(Config.TRAIN_META)
    full_val = pd.read_csv(Config.VAL_META)
    full_test = pd.read_csv(Config.TEST_META)

    # Select top 5 classes with enough samples
    top_hotels = full_train["hotel_id"].value_counts().head(5).index.tolist()

    # Subset Train: 5 images per class
    train_subset = (
        full_train[full_train["hotel_id"].isin(top_hotels)]
        .groupby("hotel_id")
        .head(5)
        .reset_index(drop=True)
    )

    # Subset Val: 2 images per class
    val_subset = (
        full_val[full_val["hotel_id"].isin(top_hotels)]
        .groupby("hotel_id")
        .head(2)
        .reset_index(drop=True)
    )

    # Subset Test: First 10 images
    test_subset = full_test.head(10).reset_index(drop=True)

    # Save subsets
    train_subset.to_csv(demo_train_path, index=False)
    val_subset.to_csv(demo_val_path, index=False)
    test_subset.to_csv(demo_test_path, index=False)

    print(
        f"Train subset: {len(train_subset)} samples, {train_subset['hotel_id'].nunique()} classes"
    )
    print(f"Val subset: {len(val_subset)} samples")
    print(f"Test subset: {len(test_subset)} samples")

    # Override Config
    Config.TRAIN_META = demo_train_path
    Config.VAL_META = demo_val_path
    Config.TEST_META = demo_test_path
    Config.WORKING_DIR = demo_dir
    Config.MODEL_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")
    Config.SUBMISSION_DIR = demo_dir

    # Reduce compute requirements for demo
    Config.TOTAL_EPOCHS = 2
    Config.WARMUP_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead
    Config.NUM_CLASSES = len(
        top_hotels
    )  # Technically Trainer recalculates this, but good to note

    # -------------------------------------------------------------------------
    # 3. Verify Library: Utils
    # -------------------------------------------------------------------------
    print("\nVerifying Utils...")
    # Test AverageMeter
    meter = AverageMeter()
    meter.update(10, n=2)
    meter.update(20, n=2)
    assert meter.avg == 15.0, f"AverageMeter failed: expected 15.0, got {meter.avg}"

    # Test MAPK
    actual = [1, 2, 3]
    predicted = [
        [1, 0, 0],
        [2, 1, 0],
        [0, 0, 0],
    ]  # 1st correct@1, 2nd correct@1, 3rd wrong
    # AP for 1: 1.0
    # AP for 2: 1.0
    # AP for 3: 0.0
    # MAP = 2/3 = 0.666...
    score = mapk(actual, predicted, k=3)
    assert abs(score - 0.666) < 0.01, f"MAPK failed: expected ~0.666, got {score}"
    print("Utils validation passed.")

    # -------------------------------------------------------------------------
    # 4. Verify Library: Dataset
    # -------------------------------------------------------------------------
    print("\nVerifying Dataset...")
    # We need a class mapping for the dataset
    # We simulate what Trainer does:
    unique_hotels = sorted(train_subset["hotel_id"].unique())
    class_to_idx = {h: i for i, h in enumerate(unique_hotels)}

    ds = HotelDataset(
        csv_path=Config.TRAIN_META,
        transform=get_transforms("train"),
        class_to_idx=class_to_idx,
        mode="train",
    )

    sample = ds[0]
    assert "image" in sample, "Dataset sample missing 'image'"
    assert "label" in sample, "Dataset sample missing 'label'"
    assert isinstance(sample["image"], torch.Tensor), "Image is not a tensor"
    assert sample["image"].shape == (
        3,
        Config.CROP_SIZE,
        Config.CROP_SIZE,
    ), f"Incorrect image shape: {sample['image'].shape}"
    print("Dataset validation passed.")

    # -------------------------------------------------------------------------
    # 5. Verify Library: Model
    # -------------------------------------------------------------------------
    print("\nVerifying Model...")
    model = HotelIdModel(
        n_classes=len(unique_hotels),
        backbone_name="efficientnet_b0",  # Use small backbone
        embedding_dim=128,  # Small dim for speed
        pretrained=False,  # No need to download weights for logic check
    )
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)

    # Test Inference (Embedding)
    with torch.no_grad():
        embeddings = model(dummy_input)
        assert embeddings.shape == (
            2,
            128,
        ), f"Embedding shape mismatch: {embeddings.shape}"
        # Check normalization
        norms = torch.norm(embeddings, p=2, dim=1)
        assert torch.allclose(
            norms, torch.ones_like(norms), atol=1e-5
        ), "Embeddings not normalized"

    # Test Training (Logits)
    dummy_labels = torch.tensor([0, 1])
    logits = model(dummy_input, dummy_labels)
    assert logits.shape == (
        2,
        len(unique_hotels),
    ), f"Logits shape mismatch: {logits.shape}"
    print("Model validation passed.")

    # -------------------------------------------------------------------------
    # 6. Verify Library: Trainer (Integration)
    # -------------------------------------------------------------------------
    print("\nStarting Trainer Execution...")
    trainer = Trainer()

    # Verify mapping loaded correctly by Trainer
    assert len(trainer.class_to_idx) == len(
        unique_hotels
    ), f"Trainer loaded {len(trainer.class_to_idx)} classes, expected {len(unique_hotels)}"

    # Run Training
    trainer.fit()

    # -------------------------------------------------------------------------
    # 7. Verify Outputs
    # -------------------------------------------------------------------------
    print("\nVerifying Outputs...")

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError("best_model.pth was not created.")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("submission.csv was not created.")

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(sub_df) == len(
        test_subset
    ), f"Submission has {len(sub_df)} rows, expected {len(test_subset)}"
    assert (
        "image" in sub_df.columns and "hotel_id" in sub_df.columns
    ), "Submission columns incorrect"

    # Check format of prediction string (space delimited)
    example_pred = sub_df.iloc[0]["hotel_id"]
    assert isinstance(example_pred, str), "Prediction is not a string"
    assert len(example_pred.split(" ")) <= 5, "Prediction contains more than 5 items"

    print("Output verification passed.")
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
