import os
import torch
import pandas as pd
import warnings
import shutil
import sys

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, collate_fn
from library.dataset import ChestXrayDataset
from library.model import build_model
from library.loss import build_criterion
from library.engine import Engine
from torch.utils.data import DataLoader


def main():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    print("--- Setting up Demo Configuration ---")

    # 1. Override Config for a fast, self-contained demo run
    # We redirect outputs to a specific demo folder in working directory
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.LOG_PATH = os.path.join(Config.WORKING_DIR, "training_log.csv")

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Initialize environment (creates directories, sets seeds)
    # This also sets Config.DEBUG = True and Config.EPOCHS = 2
    Config.setup(debug=True)

    # Further optimize for speed
    Config.EPOCHS = 1  # Reduce to 1 epoch
    Config.BATCH_SIZE = 2
    Config.DEBUG_SAMPLE_SIZE = 10  # Only use 10 images
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.NUM_QUERIES = 20  # Reduce object queries for lighter computation

    print(
        f"Config configured: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}, Debug={Config.DEBUG}"
    )

    print("\n--- 1. Demonstrating Dataset Loading ---")
    # Instantiate training dataset
    # load_cached_data=False forces processing from metadata CSVs
    train_ds = ChestXrayDataset(split="train", load_cached_data=False)

    print(f"Dataset initialized. Length: {len(train_ds)}")
    assert (
        len(train_ds) == Config.DEBUG_SAMPLE_SIZE
    ), "Dataset did not sample correctly in debug mode."

    # Fetch a single item
    img, target, img_id = train_ds[0]
    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Target Keys: {list(target.keys())}")

    # Verify Data Structure
    # Image should be (3, H, W)
    assert img.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Unexpected image shape: {img.shape}"
    # Target should contain boxes, labels, study_label
    assert "boxes" in target
    assert "labels" in target
    assert "study_label" in target
    assert isinstance(target["boxes"], torch.Tensor)

    print("\n--- 2. Demonstrating Model Forward Pass ---")
    # Create DataLoader for batch processing
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn, shuffle=False
    )

    # Get a batch
    images, targets, ids = next(iter(train_loader))
    images = images.to(Config.DEVICE)
    targets = [{k: v.to(Config.DEVICE) for k, v in t.items()} for t in targets]

    # Build Model
    model = build_model(Config)
    model.to(Config.DEVICE)
    model.train()

    # Forward Pass
    outputs = model(images)
    print("Forward pass successful.")

    # Verify Output Structure
    # Co-DETR outputs: pred_logits, pred_boxes, pred_study, enc_outputs
    required_keys = ["pred_logits", "pred_boxes", "pred_study", "enc_outputs"]
    for key in required_keys:
        assert key in outputs, f"Missing key in model output: {key}"

    # Check shapes
    # pred_logits: [Batch, Queries, Classes + 1]
    assert outputs["pred_logits"].shape == (
        Config.BATCH_SIZE,
        Config.NUM_QUERIES,
        Config.NUM_CLASSES + 1,
    )
    # pred_boxes: [Batch, Queries, 4]
    assert outputs["pred_boxes"].shape == (Config.BATCH_SIZE, Config.NUM_QUERIES, 4)
    # pred_study: [Batch, Num_Study_Classes]
    assert outputs["pred_study"].shape == (Config.BATCH_SIZE, Config.NUM_STUDY_CLASSES)

    print("\n--- 3. Demonstrating Loss Calculation ---")
    # Build Criterion
    criterion = build_criterion(Config)
    criterion.to(Config.DEVICE)
    criterion.train()

    # Calculate Loss
    loss_dict = criterion(outputs, targets)
    print(f"Loss components calculated: {list(loss_dict.keys())}")

    # Verify we have classification and box losses
    assert "loss_ce" in loss_dict
    assert "loss_bbox" in loss_dict
    assert "loss_study" in loss_dict

    # Aggregate weighted loss (simulating what happens in Engine)
    total_loss = sum(
        loss_dict[k] * criterion.weight_dict.get(k, 0.0)
        for k in loss_dict.keys()
        if k in criterion.weight_dict
    )
    print(f"Total Weighted Loss: {total_loss.item():.4f}")

    # Ensure gradients can be computed
    assert total_loss.requires_grad, "Total loss does not require gradients!"

    print("\n--- 4. Demonstrating Engine Execution (Train & Inference) ---")
    # Initialize Engine
    engine = Engine()

    # Run Training
    # This will run for 1 epoch on the small debug dataset
    print("Starting Training Loop...")
    engine.run_training()

    # Verify Model Checkpoint
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Success: Best model saved to {Config.BEST_MODEL_PATH}")
    else:
        raise FileNotFoundError("Training completed but best_model.pth was not found.")

    # Run Inference
    # This will load the 'test' split (also sampled to 10 items due to Config.DEBUG)
    # and generate a submission file.
    print("Starting Inference Loop...")
    engine.run_inference()

    # Verify Submission File
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Success: Submission saved to {Config.SUBMISSION_PATH}")

        # Validate content format
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print("Submission Head:")
        print(sub_df.head())

        assert list(sub_df.columns) == [
            "id",
            "PredictionString",
        ], "Incorrect submission columns"
        assert len(sub_df) > 0, "Submission file is empty"

        # Check if we have both study and image rows
        study_rows = sub_df[sub_df["id"].str.contains("_study")]
        image_rows = sub_df[sub_df["id"].str.contains("_image")]

        assert len(study_rows) > 0, "No study-level predictions found"
        assert len(image_rows) > 0, "No image-level predictions found"

    else:
        raise FileNotFoundError("Inference completed but submission.csv was not found.")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    main()
