import os
import torch
import pandas as pd
import numpy as np
import shutil
from library.config import Config
from library.taxonomy import TaxonomyManager
from library.dataset import HerbariumDataset, get_dataloaders
from library.model import HierarchicalConvNeXt
from library.train import Trainer
from library.inference import InferenceEngine


def run_demo():
    print("=== Starting Library Usage Demo ===\n")

    # ------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Purposes
    # ------------------------------------------------------------------------
    print("[1] Configuring environment for fast demonstration...")

    # Enable Debug mode to subsample the dataset (e.g., only use 50 images)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50

    # Reduce batch size and workers for the demo environment
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2

    # Disable downloading pretrained weights to ensure offline execution/speed
    Config.PRETRAINED = False

    # Reduce training epochs for the demo
    Config.STAGE1_EPOCHS = 1
    Config.STAGE2_EPOCHS = 1

    # Use an existing taxonomy cache if available to avoid parsing the large JSON
    # The prompt indicated 'working/idea_1/taxonomy_mappings.parquet' exists.
    existing_cache = "./working/idea_1/taxonomy_mappings.parquet"
    if os.path.exists(existing_cache):
        print(f"    Using existing taxonomy cache at {existing_cache}")
        Config.TAXONOMY_MAP_PATH = existing_cache
    else:
        print("    Existing cache not found, will rebuild (this may take a moment)...")

    # Ensure working directory exists
    Config.setup()

    print("    Configuration updated.")

    # ------------------------------------------------------------------------
    # 2. Taxonomy Manager
    # ------------------------------------------------------------------------
    print("\n[2] Testing TaxonomyManager...")
    tax_manager = TaxonomyManager()

    # Load mappings (Species -> Family -> Order)
    df_map, num_families, num_orders = tax_manager.build_mappings(load_cached_data=True)

    # Validation
    assert isinstance(df_map, pd.DataFrame), "Mapping should be a DataFrame"
    assert not df_map.empty, "Mapping DataFrame should not be empty"
    assert "category_id" in df_map.columns
    assert "family_id" in df_map.columns
    assert "order_id" in df_map.columns

    print(
        f"    Loaded taxonomy: {len(df_map)} species, {num_families} families, {num_orders} orders."
    )
    print("    TaxonomyManager verification successful.")

    # ------------------------------------------------------------------------
    # 3. Data Loading
    # ------------------------------------------------------------------------
    print("\n[3] Testing Data Loading...")

    # Get dataloaders for Stage 1 (Instance Balanced)
    loaders, n_fam, n_ord = get_dataloaders(stage=1)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    print(f"    Train loader size: {len(train_loader)} batches")

    # Fetch a single batch to verify shapes
    images, species_targets, family_targets, order_targets = next(iter(train_loader))

    # Validation
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Image batch shape mismatch: {images.shape}"
    assert species_targets.shape == (
        Config.BATCH_SIZE,
    ), "Species target shape mismatch"
    assert family_targets.shape == (Config.BATCH_SIZE,), "Family target shape mismatch"
    assert order_targets.shape == (Config.BATCH_SIZE,), "Order target shape mismatch"

    print(f"    Batch shapes verified. Images: {images.shape}")

    # ------------------------------------------------------------------------
    # 4. Model Instantiation
    # ------------------------------------------------------------------------
    print("\n[4] Testing HierarchicalConvNeXt Model...")

    device = Config.DEVICE
    model = HierarchicalConvNeXt(
        num_families=num_families, num_orders=num_orders, pretrained=Config.PRETRAINED
    ).to(device)

    # Run a forward pass with the batch fetched earlier
    images = images.to(device)
    species_logits, family_logits, order_logits = model(images)

    # Validation
    assert species_logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Species logits shape mismatch"
    assert family_logits.shape == (
        Config.BATCH_SIZE,
        num_families,
    ), "Family logits shape mismatch"
    assert order_logits.shape == (
        Config.BATCH_SIZE,
        num_orders,
    ), "Order logits shape mismatch"

    print("    Forward pass successful. Logit shapes verified.")

    # ------------------------------------------------------------------------
    # 5. Training Loop (Trainer)
    # ------------------------------------------------------------------------
    print("\n[5] Testing Trainer Class...")

    trainer = Trainer()
    trainer.initialize_model(num_families, num_orders, pretrained=Config.PRETRAINED)

    # Setup optimizer for the test
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)

    # Run one training epoch
    print("    Running 1 training epoch (subsampled)...")
    loss = trainer.train_epoch(train_loader, epoch_idx=0, total_epochs=1)

    assert loss > 0, "Training loss should be positive"
    print(f"    Training epoch finished. Loss: {loss:.4f}")

    # Run validation
    print("    Running validation...")
    f1_score = trainer.validate(val_loader)

    assert 0.0 <= f1_score <= 1.0, "F1 score must be between 0 and 1"
    print(f"    Validation finished. F1 Score: {f1_score:.4f}")

    # Save this 'trained' model for the inference step
    trainer.save_checkpoint(Config.MODEL_CHECKPOINT)
    assert os.path.exists(Config.MODEL_CHECKPOINT), "Model checkpoint was not saved"
    print("    Model checkpoint saved.")

    # ------------------------------------------------------------------------
    # 6. Inference and Submission
    # ------------------------------------------------------------------------
    print("\n[6] Testing InferenceEngine...")

    inference_engine = InferenceEngine()

    # Generate submission using the checkpoint we just saved
    # Note: Config.DEBUG is True, so this will only predict on a small subset of test data
    inference_engine.generate_submission(
        checkpoint_path=Config.MODEL_CHECKPOINT, output_path=Config.SUBMISSION_FILE
    )

    # Validation
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    assert list(df_sub.columns) == ["Id", "Predicted"], "Submission columns mismatch"
    assert not df_sub.empty, "Submission file is empty"
    assert (
        df_sub["Id"].dtype == "int64" or df_sub["Id"].dtype == "int"
    ), "Id column should be integer"

    print(f"    Submission generated at {Config.SUBMISSION_FILE}")
    print(f"    First few rows:\n{df_sub.head()}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Set fixed seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    run_demo()
