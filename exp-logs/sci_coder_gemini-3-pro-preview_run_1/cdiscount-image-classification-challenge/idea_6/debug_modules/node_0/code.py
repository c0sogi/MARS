import os
import sys
import torch
import torch.optim as optim
import numpy as np
import warnings
import pandas as pd
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import HierarchyManager
from library.dataset import BSONDataset
from library.model import MultiLevelResNet
from library.train import HierarchicalLoss, train_one_epoch, validate

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets fixed seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_demo():
    print("==== Starting Library Demo ====")

    # 1. Setup and Configuration Overrides
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Override Config for speed
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Define a small subset size for the demo
    DEMO_LIMIT = 16

    # 2. Hierarchy Manager Demonstration
    print("\n[Demo] Initializing HierarchyManager...")
    hm = HierarchyManager(load_cached_data=True)

    # Validation: Check if mappings are loaded
    assert len(hm.cat_id_to_class_idx) > 0, "Hierarchy mappings should not be empty."

    # Test conversion logic
    # Get a real category_id from the mapping dataframe
    sample_cat_id = hm.mapping_df.iloc[0]["category_id"]
    class_idx = hm.category_id_to_class_idx(sample_cat_id)
    retrieved_cat_id = hm.class_idx_to_category_id(class_idx)

    assert (
        retrieved_cat_id == sample_cat_id
    ), "Category ID round-trip conversion failed."
    print(
        f"Hierarchy Manager verified. Total L3 classes: {len(hm.cat_id_to_class_idx)}"
    )

    # 3. Dataset Demonstration
    print(f"\n[Demo] Initializing BSONDataset (Limit: {DEMO_LIMIT} samples)...")
    # We use the training metadata but limit the rows read
    dataset = BSONDataset(
        metadata_path=Config.TRAIN_METADATA,
        bson_path=Config.TRAIN_BSON,
        split="train",
        limit_size=DEMO_LIMIT,
    )

    assert (
        len(dataset) == DEMO_LIMIT
    ), f"Dataset length mismatch. Expected {DEMO_LIMIT}, got {len(dataset)}"

    # Fetch one sample to verify structure
    sample = dataset[0]
    required_keys = ["images", "mask", "sample_id", "target"]
    for key in required_keys:
        assert key in sample, f"Sample missing key: {key}"

    # Verify shapes
    # Images: (4, 3, 180, 180) -> Fixed 4 views (padded if necessary)
    assert sample["images"].shape == (
        4,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image tensor shape mismatch: {sample['images'].shape}"
    assert sample["mask"].shape == (4,), f"Mask shape mismatch: {sample['mask'].shape}"
    assert isinstance(sample["target"].item(), int), "Target should be an integer."

    print("Dataset sample verification passed.")

    # 4. Model Demonstration
    print("\n[Demo] Initializing MultiLevelResNet...")
    model = MultiLevelResNet()
    model.to(device)

    # Create a dummy batch from the dataset sample
    # Add batch dimension: (1, 4, 3, 180, 180)
    dummy_imgs = sample["images"].unsqueeze(0).to(device)
    dummy_mask = sample["mask"].unsqueeze(0).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits_l3, logits_l2, logits_l1 = model(dummy_imgs, dummy_mask)

    # Verify output shapes
    # L3: Fine-grained (Target)
    assert logits_l3.shape == (
        1,
        Config.NUM_CLASSES_L3,
    ), f"L3 Logits shape mismatch. Expected (1, {Config.NUM_CLASSES_L3}), got {logits_l3.shape}"
    # L2: Intermediate
    assert logits_l2.shape == (
        1,
        Config.NUM_CLASSES_L2,
    ), f"L2 Logits shape mismatch. Expected (1, {Config.NUM_CLASSES_L2}), got {logits_l2.shape}"
    # L1: Coarse
    assert logits_l1.shape == (
        1,
        Config.NUM_CLASSES_L1,
    ), f"L1 Logits shape mismatch. Expected (1, {Config.NUM_CLASSES_L1}), got {logits_l1.shape}"

    print("Model forward pass verification passed.")

    # 5. Training Loop Demonstration
    print("\n[Demo] Simulating Training Loop...")

    # Create DataLoader
    loader = DataLoader(
        dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True
    )

    # Setup Training Components
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=1, steps_per_epoch=len(loader)
    )
    criterion = HierarchicalLoss(hm, device)
    scaler = GradScaler()

    # Run one epoch of training
    print("Running train_one_epoch...")
    train_loss, train_acc = train_one_epoch(
        model, loader, criterion, optimizer, scheduler, scaler, device, epoch=1
    )

    assert not np.isnan(train_loss), "Training loss returned NaN."
    assert 0.0 <= train_acc <= 1.0, "Training accuracy out of bounds."

    # Run validation
    print("Running validate...")
    val_loss, val_acc = validate(model, loader, criterion, device)

    assert not np.isnan(val_loss), "Validation loss returned NaN."

    print(f"Demo Results -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Demo Results -> Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
