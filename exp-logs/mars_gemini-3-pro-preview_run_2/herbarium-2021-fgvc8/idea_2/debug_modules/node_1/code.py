import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    calculate_f1,
    save_checkpoint,
    load_checkpoint,
)
from library.taxonomy import TaxonomyManager
from library.dataset import HerbariumDataset, get_transforms, CutMixCollator
from library.model import HierarchicalEfficientNet
from library.loss import HierarchicalLoss
from library.engine import train_one_epoch, evaluate


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # -------------------------------------------------------------------------
    # We override Config parameters to ensure the demo runs quickly and uses minimal resources.
    print("\n[1] Configuring environment...")
    seed_everything(42)

    # Override Config for Demo
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Use only 20 images for this demo
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.EPOCHS = 1
    Config.BACKBONE = "resnet18"  # Use a lightweight backbone for speed

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Taxonomy Manager Demo
    # -------------------------------------------------------------------------
    print("\n[2] Testing TaxonomyManager...")
    tm = TaxonomyManager()

    # Generate or load mappings
    # This reads metadata.json and creates the species -> family/order mapping
    mapping_df = tm.process_taxonomy(
        load_cached_data=False
    )  # Force regeneration for demo

    # Get counts
    num_families, num_orders = tm.get_counts()
    print(f"    Num Families: {num_families}")
    print(f"    Num Orders: {num_orders}")

    # Verify mappings
    mapping_dict = tm.get_mappings()
    assert isinstance(mapping_dict, dict), "Mapping should be a dictionary"
    assert len(mapping_dict) > 0, "Mapping dictionary is empty"

    # Check a specific category if available (e.g., category_id 0)
    if 0 in mapping_dict:
        assert "family_id" in mapping_dict[0]
        assert "order_id" in mapping_dict[0]

    print("    TaxonomyManager verification passed.")

    # -------------------------------------------------------------------------
    # 3. Dataset & Transforms Demo
    # -------------------------------------------------------------------------
    print("\n[3] Testing HerbariumDataset & Transforms...")

    # Initialize Transforms
    train_transform = get_transforms(mode="train", image_size=Config.IMAGE_SIZE)
    val_transform = get_transforms(mode="val", image_size=Config.IMAGE_SIZE)

    # Initialize Dataset (Train)
    train_dataset = HerbariumDataset(
        mode="train", transform=train_transform, debug=True
    )
    print(f"    Train Dataset Length: {len(train_dataset)}")

    # Verify __getitem__
    img, species_id, family_id, order_id = train_dataset[0]
    print(f"    Sample 0 - Image Shape: {img.shape}")
    print(
        f"    Sample 0 - IDs: Species={species_id}, Family={family_id}, Order={order_id}"
    )

    assert img.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect image shape"
    assert isinstance(species_id, int), "Species ID should be int"
    assert isinstance(family_id, int), "Family ID should be int"
    assert isinstance(order_id, int), "Order ID should be int"

    print("    Dataset verification passed.")

    # -------------------------------------------------------------------------
    # 4. DataLoader & CutMix Collator Demo
    # -------------------------------------------------------------------------
    print("\n[4] Testing DataLoader & CutMixCollator...")

    # Initialize Collator (force p=1.0 to ensure CutMix happens for verification)
    collator = CutMixCollator(alpha=1.0, p=1.0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True,
    )

    # Fetch one batch
    images, targets = next(iter(train_loader))

    print(f"    Batch Image Shape: {images.shape}")
    print(f"    Targets Keys: {list(targets.keys())}")

    # Verify CutMix targets structure
    assert "species" in targets
    assert "family" in targets
    assert "order" in targets
    assert "lam" in targets
    assert isinstance(
        targets["species"], (tuple, list)
    ), "Species target should be a tuple or list (target_a, target_b)"
    assert targets["lam"] <= 1.0, "Lambda should be <= 1.0"

    print("    DataLoader & Collator verification passed.")

    # -------------------------------------------------------------------------
    # 5. Model Initialization Demo
    # -------------------------------------------------------------------------
    print("\n[5] Testing HierarchicalEfficientNet...")

    model = HierarchicalEfficientNet(
        backbone_name=Config.BACKBONE,
        pretrained=False,  # False for speed in demo
        num_classes=Config.NUM_CLASSES,
        num_families=num_families,
        num_orders=num_orders,
    )
    model.to(device)

    # Verify forward pass with dummy input
    dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(device)
    # Pass dummy labels for ArcFace margin calculation
    dummy_labels = torch.randint(0, Config.NUM_CLASSES, (2,)).to(device)

    species_logits, family_logits, order_logits = model(
        dummy_input, species_label=dummy_labels
    )

    print(f"    Species Logits Shape: {species_logits.shape}")
    print(f"    Family Logits Shape: {family_logits.shape}")
    print(f"    Order Logits Shape: {order_logits.shape}")

    assert species_logits.shape == (2, Config.NUM_CLASSES)
    assert family_logits.shape == (2, num_families)
    assert order_logits.shape == (2, num_orders)

    print("    Model verification passed.")

    # -------------------------------------------------------------------------
    # 6. Loss Function Demo
    # -------------------------------------------------------------------------
    print("\n[6] Testing HierarchicalLoss...")

    criterion = HierarchicalLoss().to(device)

    # Prepare targets for loss function (move to device)
    loss_targets = {
        "species": (targets["species"][0].to(device), targets["species"][1].to(device)),
        "family": (targets["family"][0].to(device), targets["family"][1].to(device)),
        "order": (targets["order"][0].to(device), targets["order"][1].to(device)),
        "lam": targets["lam"],
    }

    # Use real outputs from the model (using the batch from DataLoader)
    images = images.to(device)
    # Note: For training, we pass target_a (primary label) to the model for ArcFace margin
    outputs = model(images, species_label=loss_targets["species"][0])

    loss = criterion(outputs, loss_targets)
    print(f"    Calculated Loss: {loss.item():.4f}")

    assert torch.is_tensor(loss), "Loss should be a tensor"
    assert loss.item() > 0, "Loss should be positive"

    print("    Loss function verification passed.")

    # -------------------------------------------------------------------------
    # 7. Engine: Train & Evaluate Demo
    # -------------------------------------------------------------------------
    print("\n[7] Testing Training & Evaluation Loop...")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = None  # Skip scheduler for simple demo

    # --- Train One Epoch ---
    print("    Running train_one_epoch...")
    avg_train_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        epoch=0,
    )
    print(f"    Average Train Loss: {avg_train_loss:.4f}")

    # --- Evaluate ---
    print("    Running evaluation...")
    # Setup Validation Loader (No CutMix, simple batching)
    val_dataset = HerbariumDataset(mode="val", transform=val_transform, debug=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    f1_score, avg_val_loss = evaluate(
        model=model, loader=val_loader, criterion=criterion, device=device
    )

    print(f"    Validation F1 Score: {f1_score:.4f}")
    print(f"    Validation Loss: {avg_val_loss:.4f}")

    # -------------------------------------------------------------------------
    # 8. Utils: Checkpointing Demo
    # -------------------------------------------------------------------------
    print("\n[8] Testing Utils (Checkpointing)...")

    checkpoint_path = os.path.join(Config.WORKING_DIR, "demo_checkpoint.pth")

    # Save
    save_state = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": 0,
        "score": f1_score,
    }
    save_checkpoint(save_state, filename=checkpoint_path)
    assert os.path.exists(checkpoint_path), "Checkpoint file not created"

    # Load
    loaded_checkpoint = load_checkpoint(
        model, optimizer, filename=checkpoint_path, device=device
    )
    assert loaded_checkpoint is not None, "Failed to load checkpoint"
    assert loaded_checkpoint["score"] == f1_score, "Loaded score mismatch"

    print("    Checkpointing verification passed.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
