import os
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import set_seed
from library.taxonomy import get_taxonomy_mappings
from library.dataset import HerbariumDataset
from library.model import HierarchicalEfficientNet
from library.loss import HierarchicalLoss
from library.engine import train_one_epoch, validate, predict


def run_demo():
    # --------------------------------------------------------------------------
    # 1. Configuration Override for Fast Demonstration
    # --------------------------------------------------------------------------
    print(">>> Setting up configuration for demo...")
    # Override Config for speed and debugging
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 images for this demo
    Config.PHASE1_BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(f"Device: {Config.DEVICE}")
    print(f"Debug Mode: {Config.DEBUG}")

    # --------------------------------------------------------------------------
    # 2. Taxonomy Mapping
    # --------------------------------------------------------------------------
    print("\n>>> Loading Taxonomy Mappings...")
    # This generates or loads the mapping from species -> genus -> family
    taxonomy_df = get_taxonomy_mappings(load_cached_data=False)

    # Verification
    assert not taxonomy_df.empty, "Taxonomy DataFrame should not be empty"
    expected_cols = ["category_id", "family", "genus", "family_id", "genus_id"]
    assert all(
        col in taxonomy_df.columns for col in expected_cols
    ), "Missing columns in taxonomy mapping"

    n_families = taxonomy_df["family_id"].nunique()
    n_genera = taxonomy_df["genus_id"].nunique()
    n_species = taxonomy_df["category_id"].nunique()

    print(
        f"Taxonomy loaded: {n_families} Families, {n_genera} Genera, {n_species} Species"
    )

    # --------------------------------------------------------------------------
    # 3. Dataset and DataLoader
    # --------------------------------------------------------------------------
    print("\n>>> Initializing Datasets...")
    # Initialize datasets for Train, Val, and Test
    # Using image_size=224 for the demo (Phase 1 size)
    train_dataset = HerbariumDataset(split="train", image_size=224)
    val_dataset = HerbariumDataset(split="val", image_size=224)
    test_dataset = HerbariumDataset(split="test", image_size=224)

    # Verification of Dataset
    assert (
        len(train_dataset) == Config.DEBUG_SUBSET_SIZE
    ), f"Train dataset size mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {len(train_dataset)}"

    sample = train_dataset[0]
    assert "image" in sample
    assert "species_id" in sample
    assert sample["image"].shape == (
        3,
        224,
        224,
    ), f"Unexpected image shape: {sample['image'].shape}"
    assert isinstance(sample["species_id"], torch.Tensor)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.PHASE1_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.PHASE1_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.PHASE1_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    print("DataLoaders initialized successfully.")

    # --------------------------------------------------------------------------
    # 4. Model Instantiation
    # --------------------------------------------------------------------------
    print("\n>>> Instantiating Model...")
    # We pass pretrained=False to avoid downloading weights during this time-constrained demo
    model = HierarchicalEfficientNet(
        n_families=n_families, n_genera=n_genera, n_species=n_species, pretrained=False
    )
    model.to(Config.DEVICE)

    # Verification of Model Output Shapes
    dummy_input = torch.randn(2, 3, 224, 224).to(Config.DEVICE)
    with torch.no_grad():
        outputs = model(dummy_input)

    assert "species" in outputs
    assert "genus" in outputs
    assert "family" in outputs
    assert outputs["species"].shape == (2, n_species)
    assert outputs["genus"].shape == (2, n_genera)
    assert outputs["family"].shape == (2, n_families)
    print("Model instantiated and forward pass verified.")

    # --------------------------------------------------------------------------
    # 5. Loss Function
    # --------------------------------------------------------------------------
    print("\n>>> Setting up Loss Function...")
    criterion = HierarchicalLoss()

    # Verification of Loss Calculation
    # Create dummy targets
    dummy_targets = {
        "species_id": torch.randint(0, n_species, (2,)).to(Config.DEVICE),
        "genus_id": torch.randint(0, n_genera, (2,)).to(Config.DEVICE),
        "family_id": torch.randint(0, n_families, (2,)).to(Config.DEVICE),
    }

    loss, metrics = criterion(outputs, dummy_targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
    assert "loss_total" in metrics
    print(f"Loss function verified. Initial dummy loss: {loss.item():.4f}")

    # --------------------------------------------------------------------------
    # 6. Training Loop (Single Epoch)
    # --------------------------------------------------------------------------
    print("\n>>> Running Training Loop (1 Epoch)...")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train for one epoch
    avg_train_loss = train_one_epoch(
        model, train_loader, optimizer, Config.DEVICE, criterion
    )
    print(f"Epoch 1 Training Completed. Average Loss: {avg_train_loss:.4f}")

    assert avg_train_loss > 0, "Training loss should be positive"

    # --------------------------------------------------------------------------
    # 7. Validation Loop
    # --------------------------------------------------------------------------
    print("\n>>> Running Validation...")
    val_loss, val_f1 = validate(model, val_loader, Config.DEVICE, criterion)
    print(f"Validation Completed. Loss: {val_loss:.4f}, Macro F1: {val_f1:.4f}")

    # Since model is random and untrained (pretrained=False), F1 will be near 0, but code should run.
    assert val_loss >= 0
    assert 0.0 <= val_f1 <= 1.0

    # --------------------------------------------------------------------------
    # 8. Inference / Prediction
    # --------------------------------------------------------------------------
    print("\n>>> Running Prediction on Test Set...")
    output_csv = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    predict(model, test_loader, Config.DEVICE, output_path=output_csv)

    # Verification of Submission File
    assert os.path.exists(output_csv), "Submission file was not created."
    submission_df = pd.read_csv(output_csv)
    assert list(submission_df.columns) == [
        "Id",
        "Predicted",
    ], "Incorrect submission columns."
    assert (
        len(submission_df) == Config.DEBUG_SUBSET_SIZE
    ), f"Submission rows mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {len(submission_df)}"

    print(f"Prediction completed. Submission saved to {output_csv}")
    print("\n>>> Demo Completed Successfully!")


if __name__ == "__main__":
    run_demo()
