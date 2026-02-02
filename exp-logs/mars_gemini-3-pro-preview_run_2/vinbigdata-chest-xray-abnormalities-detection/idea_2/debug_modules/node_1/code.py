import os
import pandas as pd
import torch
import shutil
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dicom_converter import convert_and_cache_data
from library.dataset import VinBigDataset, get_transforms, collate_fn
from library.model import get_model
from library.engine import Trainer


def main():
    # --- 1. Setup ---
    seed_everything(42)
    logger = get_logger("demo_script")
    logger.info("Starting demo execution...")

    # --- 2. Create Subset Metadata for Speed ---
    # We create a small subset of the metadata to demonstrate the pipeline quickly.
    # Processing the full 15,000+ DICOMs would take too long for this demo.

    subset_dir = "./working/demo_metadata"
    os.makedirs(subset_dir, exist_ok=True)

    # Read original metadata
    logger.info("Loading original metadata to create subsets...")
    train_full = pd.read_csv("./metadata/train_meta.csv")
    val_full = pd.read_csv("./metadata/val_meta.csv")
    test_full = pd.read_csv("./metadata/test_meta.csv")

    # Select a small number of images
    # We include some images with findings (class_id != 14) and some without.
    train_findings = train_full[train_full["class_id"] != 14]["image_id"].unique()[:10]
    train_nofindings = train_full[train_full["class_id"] == 14]["image_id"].unique()[:5]
    train_ids = list(train_findings) + list(train_nofindings)

    # Filter DataFrames
    train_subset = train_full[train_full["image_id"].isin(train_ids)].copy()
    val_subset = val_full[
        val_full["image_id"].isin(val_full["image_id"].unique()[:5])
    ].copy()
    test_subset = test_full[
        test_full["image_id"].isin(test_full["image_id"].unique()[:5])
    ].copy()

    # Save subset CSVs
    train_sub_path = os.path.join(subset_dir, "train.csv")
    val_sub_path = os.path.join(subset_dir, "val.csv")
    test_sub_path = os.path.join(subset_dir, "test.csv")

    train_subset.to_csv(train_sub_path, index=False)
    val_subset.to_csv(val_sub_path, index=False)
    test_subset.to_csv(test_sub_path, index=False)

    logger.info(
        f"Subsets created: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_subset)}"
    )

    # --- 3. Patch Configuration ---
    # Modify Config attributes to point to our subsets and temporary working directories.
    Config.TRAIN_META_PATH = train_sub_path
    Config.VAL_META_PATH = val_sub_path
    Config.TEST_META_PATH = test_sub_path

    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_OUTPUT_DIR = os.path.join(Config.WORKING_DIR, "models")
    Config.LOG_DIR = os.path.join(Config.WORKING_DIR, "logs")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Reduce hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2

    # Setup directories based on new Config
    Config.setup()

    # --- 4. Run Preprocessing (DICOM -> PNG) ---
    logger.info("Running preprocessing on subset...")
    # load_cached_data=False ensures we process our new subset fresh
    train_df, val_df, test_df = convert_and_cache_data(load_cached_data=False)

    # Verify preprocessing results
    assert len(train_df) == len(train_subset), "Processed train DataFrame size mismatch"
    assert os.path.exists(
        train_df.iloc[0]["file_path"]
    ), "Processed PNG file does not exist"
    assert train_df.iloc[0]["file_path"].endswith(
        ".png"
    ), "File extension should be .png"
    logger.info("Preprocessing completed and verified.")

    # --- 5. Create Datasets and DataLoaders ---
    logger.info("Instantiating Datasets and DataLoaders...")

    train_dataset = VinBigDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = VinBigDataset(val_df, transforms=get_transforms("val"), mode="val")
    test_dataset = VinBigDataset(
        test_df, transforms=get_transforms("test"), mode="test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Verify Data Loading logic
    try:
        images, targets = next(iter(train_loader))
        assert len(images) <= Config.BATCH_SIZE, "Batch size mismatch"
        assert isinstance(images, torch.Tensor), "Images should be a torch.Tensor"
        assert len(targets) == len(images), "Targets list length must match batch size"
        assert "boxes" in targets[0], "Target dict missing 'boxes'"
        assert "labels" in targets[0], "Target dict missing 'labels'"
        logger.info("Data loading verified.")
    except StopIteration:
        logger.error("DataLoader is empty!")
        raise

    # --- 6. Initialize Model ---
    logger.info("Initializing Faster R-CNN model...")
    model = get_model(num_classes=Config.NUM_CLASSES)
    model.to(Config.DEVICE)

    # Setup Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # --- 7. Training Loop ---
    logger.info("Starting training (1 epoch)...")
    trainer = Trainer(model, optimizer, Config.DEVICE)

    save_path = os.path.join(Config.MODEL_OUTPUT_DIR, "best_model.pth")

    # Run training
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS, save_path=save_path)

    # Verify model checkpoint
    assert os.path.exists(save_path), "Model checkpoint was not saved."
    logger.info(f"Training completed. Model saved to {save_path}")

    # --- 8. Inference ---
    logger.info("Starting inference on test subset...")

    # Reload best model to ensure saving/loading works
    model.load_state_dict(torch.load(save_path, map_location=Config.DEVICE))

    # Run prediction
    trainer.predict(test_loader, Config.SUBMISSION_PATH)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        len(sub_df) == test_subset["image_id"].nunique()
    ), "Submission row count mismatch"
    assert (
        "image_id" in sub_df.columns and "PredictionString" in sub_df.columns
    ), "Submission columns missing"

    # Check format of one prediction string
    pred_string = sub_df.iloc[0]["PredictionString"]
    assert isinstance(pred_string, str), "PredictionString must be a string"
    # Even "No finding" should be "14 1 0 0 1 1"
    assert (
        len(pred_string.split()) % 6 == 0
    ), "PredictionString format invalid (must be multiples of 6)"

    logger.info(f"Inference verified. Submission saved to {Config.SUBMISSION_PATH}")
    logger.info("Demo script completed successfully.")


if __name__ == "__main__":
    main()
