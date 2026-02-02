import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.taxonomy import TaxonomyManager
from library.dataset import get_dataloaders
from library.model import HierarchicalConvNeXt
from library.trainer import Trainer


def main():
    print("=== Starting Library Usage Demonstration ===\n")

    # ------------------------------------------------------------------------
    # 1. Configuration Override for Demo
    # ------------------------------------------------------------------------
    print("[1] Configuring environment for fast demonstration...")

    # Set a fixed seed for reproducibility
    set_seed(42)

    # Override Config defaults to run a quick test instead of full training
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 images for speed
    Config.BATCH_SIZE = 4  # Small batch size
    Config.STAGE1_EPOCHS = 1  # Only 1 epoch for demo
    Config.STAGE2_EPOCHS = 1  # Only 1 epoch for demo

    # Use a specific working directory for this demo
    Config.WORK_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_execution/submission"

    # Define paths for outputs
    Config.STAGE1_CHECKPOINT = os.path.join(Config.WORK_DIR, "stage1_checkpoint.pth")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORK_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Leverage existing taxonomy cache from the environment to save parsing time
    # (Checking if idea_1 cache exists, otherwise fallback to default which might take longer)
    existing_cache = "./working/idea_1/taxonomy_mappings.parquet"
    if os.path.exists(existing_cache):
        print(f"    Using existing taxonomy cache at: {existing_cache}")
        Config.TAXONOMY_MAP_PATH = existing_cache
    else:
        Config.TAXONOMY_MAP_PATH = os.path.join(
            Config.WORK_DIR, "taxonomy_mappings.parquet"
        )

    # Ensure directories exist
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print("    Configuration updated successfully.")

    # ------------------------------------------------------------------------
    # 2. Taxonomy Manager Demo
    # ------------------------------------------------------------------------
    print("\n[2] Testing TaxonomyManager...")
    tax_mgr = TaxonomyManager()

    # Load mappings
    tax_mgr.load(load_cached_data=True)

    # Verify counts
    num_families, num_orders = tax_mgr.get_counts()
    print(f"    Loaded Taxonomy: {num_families} families, {num_orders} orders.")

    assert num_families > 0, "Taxonomy failed to load families."
    assert num_orders > 0, "Taxonomy failed to load orders."

    # Verify mappings
    s2f, s2o = tax_mgr.get_mappings()
    assert len(s2f) > 0, "Species-to-Family mapping is empty."
    assert len(s2o) > 0, "Species-to-Order mapping is empty."
    print("    Taxonomy verification passed.")

    # ------------------------------------------------------------------------
    # 3. Data Loading Demo
    # ------------------------------------------------------------------------
    print("\n[3] Testing DataLoaders...")

    # Get dataloaders for Stage 1
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        stage=1, debug_sample_size=Config.DEBUG_SAMPLE_SIZE
    )

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")
    print(f"    Test Batches:  {len(test_loader)}")

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    images, species, families, orders = batch

    print(
        f"    Batch Shapes -> Images: {images.shape}, Species: {species.shape}, Families: {families.shape}, Orders: {orders.shape}"
    )

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect image tensor shape."
    assert species.shape == (Config.BATCH_SIZE,), "Incorrect species label shape."
    assert families.shape == (Config.BATCH_SIZE,), "Incorrect family label shape."
    assert orders.shape == (Config.BATCH_SIZE,), "Incorrect order label shape."
    print("    DataLoader verification passed.")

    # ------------------------------------------------------------------------
    # 4. Model Demo
    # ------------------------------------------------------------------------
    print("\n[4] Testing HierarchicalConvNeXt Model...")

    # Instantiate model
    model = HierarchicalConvNeXt(
        num_species=Config.NUM_CLASSES,
        num_families=num_families,
        num_orders=num_orders,
        pretrained=False,  # False for speed in demo, usually True
    )

    # Move to device
    device = Config.DEVICE
    model.to(device)

    # Test Forward Pass
    dummy_input = images.to(device)
    outputs = model(dummy_input)

    print("    Forward pass output keys:", outputs.keys())
    assert "species" in outputs and "family" in outputs and "order" in outputs
    assert outputs["species"].shape == (Config.BATCH_SIZE, Config.NUM_CLASSES)
    assert outputs["family"].shape == (Config.BATCH_SIZE, num_families)
    assert outputs["order"].shape == (Config.BATCH_SIZE, num_orders)

    # Test Freezing Logic
    print("    Testing backbone freezing...")
    model.freeze_backbone(True)
    for param in model.backbone.parameters():
        assert param.requires_grad is False, "Backbone parameter should be frozen."

    model.freeze_backbone(False)
    # Check one param to ensure unfreezing worked
    for param in model.backbone.parameters():
        if param.requires_grad:
            break
    else:
        assert False, "Backbone should be unfrozen."

    print("    Model verification passed.")

    # ------------------------------------------------------------------------
    # 5. Trainer Demo (Training & Inference)
    # ------------------------------------------------------------------------
    print("\n[5] Testing Trainer (Training & Inference Loop)...")

    trainer = Trainer(model, device=device)

    # --- Stage 1: Representation Learning ---
    print("    Running Stage 1 (Representation Learning)...")
    trainer.fit_stage1(train_loader, val_loader)

    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Stage 1 did not save the best model."

    # --- Stage 2: Classifier Re-balancing ---
    print("    Running Stage 2 (Classifier Re-balancing)...")
    # Note: fit_stage2 loads the best model from disk, freezes backbone, and trains head
    trainer.fit_stage2(train_loader, val_loader)

    # Verify freezing in Stage 2 (Trainer handles this internally, but we can check model state)
    # After fit_stage2, backbone should be frozen
    for param in trainer.model.backbone.parameters():
        assert (
            param.requires_grad is False
        ), "Backbone should be frozen after Stage 2 setup."

    # --- Inference ---
    print("    Running Inference on Test Set...")
    trainer.predict(test_loader, output_path=Config.SUBMISSION_PATH)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission generated with {len(df_sub)} rows.")
    assert (
        "Id" in df_sub.columns and "Predicted" in df_sub.columns
    ), "Submission columns missing."
    assert len(df_sub) > 0, "Submission file is empty."

    print("    Trainer verification passed.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
