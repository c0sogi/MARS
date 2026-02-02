import sys
import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config, seed_everything
from library.dataset import get_dataloaders
from library.model import CascadedTaxonomicNetwork
from library.loss import HierarchicalLoss
from library.trainer import Trainer
from library.inference import InferenceRunner


def run_demo():
    print("Step 1: Configuration and Setup")
    # Set seed for reproducibility
    seed_everything(42)

    # Monkey-patch Config for speed and demonstration purposes
    print("Modifying Config for fast demonstration...")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4  # Small batch size for speed and low memory usage
    Config.DEBUG = True  # Use subsets of data via the debug flag in get_dataloaders
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in this simple script

    # Ensure working directory exists
    if not os.path.exists(Config.WORK_DIR):
        os.makedirs(Config.WORK_DIR)

    print("-" * 40)

    print("Step 2: Data Loading Verification")
    # Load dataloaders with debug=True to use small subsets
    train_loader, val_loader, test_loader, meta_counts = get_dataloaders(
        debug=Config.DEBUG
    )

    # Verify Train Loader structure
    try:
        images, (sp_labels, gn_labels, fm_labels) = next(iter(train_loader))
        print(f"Train Batch - Images: {images.shape}")
        print(
            f"Train Batch - Labels: Species {sp_labels.shape}, Genus {gn_labels.shape}, Family {fm_labels.shape}"
        )

        # Assertions to verify data shapes
        assert images.shape[0] == Config.BATCH_SIZE
        assert images.shape[1] == 3
        assert images.shape[2] == Config.IMG_SIZE
        assert images.shape[3] == Config.IMG_SIZE
        assert sp_labels.shape[0] == Config.BATCH_SIZE
        assert gn_labels.shape[0] == Config.BATCH_SIZE
        assert fm_labels.shape[0] == Config.BATCH_SIZE
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # Verify Test Loader structure
    try:
        test_images, test_ids = next(iter(test_loader))
        print(f"Test Batch - Images: {test_images.shape}")
        print(f"Test Batch - IDs: {len(test_ids)}")
        assert len(test_ids) == Config.BATCH_SIZE
    except StopIteration:
        raise AssertionError("Test loader is empty!")

    print("Data Loading Verified.")
    print("-" * 40)

    print("Step 3: Model and Loss Verification")
    num_species = Config.NUM_CLASSES
    num_families = meta_counts["num_families"]
    num_genera = meta_counts["num_genera"]

    print(
        f"Initializing Model with {num_species} species, {num_genera} genera, {num_families} families..."
    )
    # Initialize model
    # Note: In a real run, pretrained=True is used. Here we test the architecture.
    model = CascadedTaxonomicNetwork(
        num_species=num_species,
        num_families=num_families,
        num_genera=num_genera,
        pretrained=False,
    ).to(Config.DEVICE)

    # Prepare inputs for forward pass
    images = images.to(Config.DEVICE)
    sp_labels = sp_labels.to(Config.DEVICE)
    gn_labels = gn_labels.to(Config.DEVICE)
    fm_labels = fm_labels.to(Config.DEVICE)

    # Forward pass (Training mode with labels for ArcFace)
    outputs = model(images, labels=(sp_labels, gn_labels, fm_labels))
    sp_logits, gn_logits, fm_logits = outputs

    print(
        f"Output Shapes - Species: {sp_logits.shape}, Genus: {gn_logits.shape}, Family: {fm_logits.shape}"
    )

    # Verify output shapes
    assert sp_logits.shape == (Config.BATCH_SIZE, num_species)
    assert gn_logits.shape == (Config.BATCH_SIZE, num_genera)
    assert fm_logits.shape == (Config.BATCH_SIZE, num_families)

    # Loss Calculation
    criterion = HierarchicalLoss()
    loss = criterion(outputs, (sp_labels, gn_labels, fm_labels))

    print(f"Calculated Loss: {loss.item()}")
    assert not torch.isnan(loss)
    assert loss.item() > 0

    print("Model and Loss Verified.")
    print("-" * 40)

    print("Step 4: Trainer Execution (Training Loop)")
    # Initialize Trainer
    trainer = Trainer(meta_counts)

    print("Starting training (1 Epoch)...")
    # Execute training loop
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS, patience=1)

    # Verify checkpoint creation
    best_model_path = os.path.join(Config.WORK_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Training successful. Model saved at {best_model_path}")
    else:
        raise AssertionError("Best model file was not created!")

    print("Trainer Execution Verified.")
    print("-" * 40)

    print("Step 5: Inference Execution")
    # Initialize Inference Runner
    inference_runner = InferenceRunner()

    # Run inference pipeline
    # This loads the best model trained in Step 4 and generates predictions
    inference_runner.run()

    # Verify Submission File
    submission_path = Config.SUBMISSION_PATH
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"Submission generated with {len(df_sub)} rows.")
        print(df_sub.head())

        # Basic validation of submission format
        assert "Id" in df_sub.columns
        assert "Predicted" in df_sub.columns
        assert len(df_sub) > 0
        assert not df_sub.isnull().values.any()
    else:
        raise AssertionError("Submission file not found!")

    print("Inference Verified.")
    print("-" * 40)
    print("Success: All components demonstrated and verified.")


if __name__ == "__main__":
    run_demo()
