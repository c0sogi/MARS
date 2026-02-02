import os
import torch
import shutil
import pandas as pd
import numpy as np
from library.utils import seed_everything, get_logger
from library.dataset import get_dataloaders
from library.model import HierarchicalEfficientNet
from library.loss import HierarchicalLoss
from library.trainer import Trainer


def run_demo():
    # ------------------------------------------------------------------
    # 1. Setup and Configuration
    # ------------------------------------------------------------------
    # Define working directories
    WORKING_DIR = "./working/demo_run"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Clean up previous run if exists
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR)

    # Set random seeds for reproducibility
    seed_everything(42)

    logger = get_logger("Demo")
    logger.info("Starting library usage demonstration...")

    # Configuration for the run
    # We use a very small sample size and minimal epochs for demonstration speed
    config = {
        "num_species": 15501,  # As per dataset description
        "epochs": 1,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 1,
        "checkpoint_dir": CHECKPOINT_DIR,
        "genus_weight": 0.1,
        "family_weight": 0.1,
        "label_smoothing": 0.1,
        "batch_size": 8,
        "image_size": 128,  # Small image size for speed
        "num_workers": 2,
        "sample_size": 50,  # Only use 50 images for this demo
    }

    # ------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # ------------------------------------------------------------------
    logger.info("Initializing DataLoaders...")

    # get_dataloaders handles loading metadata, hierarchy mapping, and creating datasets
    train_loader, val_loader, test_loader, hierarchy_info = get_dataloaders(
        train_batch_size=config["batch_size"],
        val_batch_size=config["batch_size"],
        image_size=config["image_size"],
        num_workers=config["num_workers"],
        sample_size=config["sample_size"],
        cache_dir=CACHE_DIR,
    )

    # Validation: Check hierarchy info
    assert "num_genera" in hierarchy_info
    assert "num_families" in hierarchy_info
    assert hierarchy_info["num_genera"] > 0
    assert hierarchy_info["num_families"] > 0
    logger.info(
        f"Hierarchy loaded: {hierarchy_info['num_genera']} Genera, {hierarchy_info['num_families']} Families"
    )

    # Validation: Check DataLoader output shapes
    # Fetch one batch from train_loader
    images, species_ids, genus_ids, family_ids = next(iter(train_loader))

    # Expected shapes:
    # Images: [Batch, 3, H, W]
    # Labels: [Batch]
    assert images.dim() == 4
    assert images.shape[1] == 3
    assert images.shape[2] == config["image_size"]
    assert species_ids.shape[0] == config["batch_size"]
    assert genus_ids.shape[0] == config["batch_size"]
    assert family_ids.shape[0] == config["batch_size"]

    logger.info("DataLoader shapes verified.")

    # ------------------------------------------------------------------
    # 3. Model Initialization Demonstration
    # ------------------------------------------------------------------
    logger.info("Initializing Model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HierarchicalEfficientNet(
        num_species=config["num_species"],
        num_genera=hierarchy_info["num_genera"],
        num_families=hierarchy_info["num_families"],
        pretrained=True,  # Using pretrained weights
    ).to(device)

    # Validation: Forward pass with dummy data
    dummy_input = torch.randn(2, 3, config["image_size"], config["image_size"]).to(
        device
    )
    with torch.no_grad():
        outputs = model(dummy_input)

    # Check output structure
    assert "species" in outputs
    assert "genus" in outputs
    assert "family" in outputs
    assert outputs["species"].shape == (2, config["num_species"])
    assert outputs["genus"].shape == (2, hierarchy_info["num_genera"])
    assert outputs["family"].shape == (2, hierarchy_info["num_families"])

    logger.info("Model forward pass verified.")

    # ------------------------------------------------------------------
    # 4. Loss Function Demonstration
    # ------------------------------------------------------------------
    logger.info("Testing Loss Function...")

    criterion = HierarchicalLoss(
        genus_weight=config["genus_weight"],
        family_weight=config["family_weight"],
        label_smoothing=config["label_smoothing"],
    )

    # Create dummy targets on device
    dummy_species = torch.randint(0, config["num_species"], (2,)).to(device)
    dummy_genus = torch.randint(0, hierarchy_info["num_genera"], (2,)).to(device)
    dummy_family = torch.randint(0, hierarchy_info["num_families"], (2,)).to(device)

    loss = criterion(outputs, (dummy_species, dummy_genus, dummy_family))

    # Check loss is a scalar
    assert loss.dim() == 0
    assert loss.item() > 0

    logger.info(f"Loss computation verified. Loss value: {loss.item():.4f}")

    # ------------------------------------------------------------------
    # 5. Training Loop Demonstration (Trainer)
    # ------------------------------------------------------------------
    logger.info("Starting Training Loop (1 Epoch)...")

    trainer = Trainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        hierarchy_info=hierarchy_info,
    )

    # Run training
    # This will run for 1 epoch on the small subset of data
    trainer.fit()

    # Validation: Check if checkpoint was saved
    expected_checkpoint = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(expected_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {expected_checkpoint}")

    logger.info(f"Training complete. Checkpoint saved at {expected_checkpoint}")

    # ------------------------------------------------------------------
    # 6. Inference Demonstration
    # ------------------------------------------------------------------
    logger.info("Running Inference on Test Set...")

    # Load best model
    model.load_state_dict(torch.load(expected_checkpoint, map_location=device))
    model.eval()

    predictions = []
    image_ids = []

    # Iterate through test loader (subset)
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)
            outputs = model(images)

            # We only care about species prediction for submission
            preds = torch.argmax(outputs["species"], dim=1).cpu().numpy()

            predictions.extend(preds)
            image_ids.extend(ids)

    # Create submission dataframe
    submission_df = pd.DataFrame({"Id": image_ids, "Predicted": predictions})

    # Validation: Check submission format
    assert len(submission_df) == len(test_loader.dataset)
    assert "Id" in submission_df.columns
    assert "Predicted" in submission_df.columns

    submission_path = os.path.join(WORKING_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    logger.info(f"Inference complete. Submission saved to {submission_path}")
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
