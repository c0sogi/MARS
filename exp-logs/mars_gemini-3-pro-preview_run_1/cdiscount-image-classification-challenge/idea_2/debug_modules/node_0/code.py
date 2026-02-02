import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim

# Import library modules
from library.config import Config
from library.utils import CategoryHierarchy
from library.dataset import BSONProductDataset, collate_fn
from library.model import HierarchicalResNet50
from library.train import Trainer
from library.evaluate import Evaluator


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    print("==== Starting Hierarchical ResNet-50 Demonstration ====")

    # 1. Setup Configuration for Speed
    # We override Config attributes to run a fast demonstration on a small subset.
    print("\n[Step 1] Configuring for fast demonstration...")
    Config.DEBUG = True  # Limits dataset to 1000 samples via dataset.py logic
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 16  # Small batch size for demonstration
    Config.NUM_WORKERS = 2  # Reduce worker overhead

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Demonstrate CategoryHierarchy
    print("\n[Step 2] Testing CategoryHierarchy...")
    # Force creation (load_cached_data=False) to verify mapping logic
    hierarchy = CategoryHierarchy(load_cached_data=False)

    # Validation: Check if mappings exist
    assert len(hierarchy.id_to_indices) > 0, "Hierarchy mapping is empty"
    assert len(hierarchy.l3_idx_to_id) > 0, "L3 to ID mapping is empty"

    # Test a specific lookup (using the first available category)
    test_cat_id = list(hierarchy.id_to_indices.keys())[0]
    l1, l2, l3 = hierarchy.get_hierarchy_indices(test_cat_id)
    print(f"Category ID {test_cat_id} maps to indices: L1={l1}, L2={l2}, L3={l3}")

    # Verify reverse mapping
    recovered_id = hierarchy.get_category_id_from_l3(l3)
    assert recovered_id == test_cat_id, "Reverse mapping failed"
    print("CategoryHierarchy logic verified.")

    # 3. Demonstrate Dataset and DataLoader
    print("\n[Step 3] Testing Dataset and DataLoader...")
    # Initialize Train Dataset
    train_dataset = BSONProductDataset(mode="train")
    print(f"Train Dataset Size (DEBUG mode): {len(train_dataset)}")
    assert len(train_dataset) > 0, "Train dataset is empty"

    # Test single item retrieval
    sample = train_dataset[0]
    assert "id" in sample
    assert "images" in sample
    assert "labels" in sample
    assert isinstance(sample["images"], list)
    assert len(sample["images"]) > 0

    # Check image tensor shape (C, H, W) -> (3, 180, 180)
    img_shape = sample["images"][0].shape
    assert img_shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {img_shape}"
    print("Single item retrieval verified.")

    # Initialize DataLoader with custom collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Test Batch Retrieval
    batch = next(iter(train_loader))
    # Expected shapes:
    # images: (B, 4, 3, 180, 180) - 4 is MAX_IMAGES fixed in collate_fn
    # labels['target']: (B,)
    assert batch["images"].dim() == 5, "Batch images should be 5D (B, N, C, H, W)"
    assert batch["images"].size(1) == 4, "Batch images should have sequence length 4"
    assert (
        batch["labels"]["target"].size(0) == Config.BATCH_SIZE
    ), "Incorrect batch size in labels"
    print("DataLoader batch retrieval verified.")

    # 4. Demonstrate Model
    print("\n[Step 4] Testing HierarchicalResNet50 Model...")
    model = HierarchicalResNet50()
    model.to(device)

    # Forward pass with the batch retrieved earlier
    images = batch["images"].to(device)
    outputs = model(images)

    # Verify output shapes
    # L1: (B, 49), L2: (B, 483), Target: (B, 5270)
    assert outputs["l1"].shape == (Config.BATCH_SIZE, Config.NUM_CLASSES_L1)
    assert outputs["l2"].shape == (Config.BATCH_SIZE, Config.NUM_CLASSES_L2)
    assert outputs["target"].shape == (Config.BATCH_SIZE, Config.NUM_CLASSES_L3)
    print("Model forward pass verified.")

    # 5. Demonstrate Training Loop
    print("\n[Step 5] Testing Training Loop...")
    # Setup Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=Config.EPOCHS, steps_per_epoch=len(train_loader)
    )

    # Initialize Validation Loader for the Trainer
    val_dataset = BSONProductDataset(mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    trainer = Trainer(model, train_loader, val_loader, device)

    # Run one epoch
    print("Running 1 epoch of training...")
    loss, acc = trainer.train_epoch(optimizer, scheduler, epoch_idx=1)

    assert not np.isnan(loss), "Training loss is NaN"
    assert 0.0 <= acc <= 1.0, "Accuracy out of bounds"
    print(f"Training Epoch Completed. Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    # Run validation
    print("Running validation...")
    val_loss, val_acc = trainer.validate()
    print(f"Validation Completed. Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    # Save a checkpoint (simulating the training loop saving the best model)
    torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)
    print("Model checkpoint saved.")

    # 6. Demonstrate Evaluation and Submission
    print("\n[Step 6] Testing Evaluation and Submission...")
    evaluator = Evaluator(model, device)

    # Test Submission Generation
    # We use the test dataset (also limited by DEBUG mode)
    test_dataset = BSONProductDataset(mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    print(f"Test Dataset Size (DEBUG mode): {len(test_dataset)}")

    # Generate submission
    output_csv = Config.SUBMISSION_PATH
    # Clean up previous run if exists
    if os.path.exists(output_csv):
        os.remove(output_csv)

    evaluator.generate_submission(test_loader, output_csv)

    # Verify Submission File
    assert os.path.exists(output_csv), "Submission file was not created"

    df_sub = pd.read_csv(output_csv)
    print(f"Submission file loaded. Rows: {len(df_sub)}")

    # Verify columns
    assert "_id" in df_sub.columns, "Missing _id column"
    assert "category_id" in df_sub.columns, "Missing category_id column"

    # Verify content (category_id should be integer)
    assert pd.api.types.is_integer_dtype(
        df_sub["category_id"]
    ), "category_id should be integer"

    print("Submission generation verified.")
    print("\n==== Demonstration Complete ====")


if __name__ == "__main__":
    main()
