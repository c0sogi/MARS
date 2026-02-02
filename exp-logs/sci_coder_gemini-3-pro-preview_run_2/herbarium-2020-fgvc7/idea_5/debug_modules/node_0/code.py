import os
import pandas as pd
import numpy as np
import torch
from library import config
from library.taxonomy import TaxonomyManager
from library.dataset import HerbariumDataset, get_dataloaders
from library.model import HierarchicalEfficientNet
from library.loss import HierarchicalLoss
from library.engine import train_model


def run_demo():
    print("=== Starting Herbarium Classification Demo ===")

    # Ensure reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # -------------------------------------------------------------------------
    # 1. Prepare Mock Data for Speed
    # -------------------------------------------------------------------------
    print("\n[1] Preparing subset data and mock taxonomy for fast execution...")

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Load original train CSV to get valid file paths and IDs
    full_train_df = pd.read_csv(config.TRAIN_CSV)

    # Create a small subset (50 samples for train, 50 for val)
    subset_size = 50
    demo_train_df = full_train_df.head(subset_size).copy()
    demo_val_df = full_train_df.iloc[subset_size : subset_size * 2].copy()

    # Save subsets to working directory
    demo_train_csv_path = os.path.join(config.WORKING_DIR, "train_demo.csv")
    demo_val_csv_path = os.path.join(config.WORKING_DIR, "val_demo.csv")

    demo_train_df.to_csv(demo_train_csv_path, index=False)
    demo_val_df.to_csv(demo_val_csv_path, index=False)
    print(f"    Created demo CSVs with {subset_size} samples each.")

    # Mock Taxonomy Mapping
    # The real TaxonomyManager parses a 10M line JSON. We mock the parquet cache
    # to skip that and use random mappings for the subset categories.
    unique_cats = pd.concat(
        [demo_train_df["category_id"], demo_val_df["category_id"]]
    ).unique()

    # Generate random genus and family IDs (small range for demo)
    mock_genus_ids = np.random.randint(0, 10, size=len(unique_cats))
    mock_family_ids = np.random.randint(0, 5, size=len(unique_cats))

    mapping_df = pd.DataFrame(
        {"id": unique_cats, "genus_id": mock_genus_ids, "family_id": mock_family_ids}
    ).set_index("id")

    # Save to the path expected by TaxonomyManager
    mapping_df.to_parquet(config.TAXONOMY_MAPPING_PATH)
    print(f"    Mocked taxonomy mapping saved to {config.TAXONOMY_MAPPING_PATH}")

    # -------------------------------------------------------------------------
    # 2. Test TaxonomyManager
    # -------------------------------------------------------------------------
    print("\n[2] Testing TaxonomyManager...")
    taxonomy = TaxonomyManager(load_cached_data=True)

    # Verification
    sample_cat_id = unique_cats[0]
    genus_id = taxonomy.get_genus_id(sample_cat_id)
    family_id = taxonomy.get_family_id(sample_cat_id)

    assert genus_id is not None, "Genus ID should not be None"
    assert family_id is not None, "Family ID should not be None"
    print(
        f"    Taxonomy lookup verified: Species {sample_cat_id} -> Genus {genus_id}, Family {family_id}"
    )

    # -------------------------------------------------------------------------
    # 3. Test Dataset and DataLoader
    # -------------------------------------------------------------------------
    print("\n[3] Testing Dataset and DataLoader...")

    # Use the demo CSVs with a small batch size
    train_loader, val_loader, _ = get_dataloaders(
        train_csv=demo_train_csv_path,
        val_csv=demo_val_csv_path,
        test_csv=demo_val_csv_path,  # Dummy usage
        batch_size=8,
        num_workers=2,  # Reduce workers for demo
    )

    # Fetch one batch
    images, targets = next(iter(train_loader))
    species_targets, genus_targets, family_targets = targets

    # Verification
    assert images.dim() == 4, "Images should be 4D tensor (B, C, H, W)"
    assert images.shape[1] == 3, "Images should have 3 channels"
    assert len(species_targets) == 8, "Batch size should match"
    print(f"    Batch loaded. Image shape: {images.shape}")
    print(f"    Targets retrieved: Species, Genus, Family")

    # -------------------------------------------------------------------------
    # 4. Test Model
    # -------------------------------------------------------------------------
    print("\n[4] Testing HierarchicalEfficientNet Model...")

    # Instantiate model
    # It will automatically use the TaxonomyManager we mocked to determine head sizes
    model = HierarchicalEfficientNet()
    model.to(config.DEVICE)

    # Forward pass
    images = images.to(config.DEVICE)
    outputs = model(images)

    # Verification
    assert "species" in outputs
    assert "genus" in outputs
    assert "family" in outputs
    assert outputs["species"].shape == (8, config.NUM_SPECIES_CLASSES)

    # Check genus/family output sizes match our mocked taxonomy counts
    expected_genus_classes = mapping_df["genus_id"].max() + 1
    expected_family_classes = mapping_df["family_id"].max() + 1

    assert outputs["genus"].shape[1] == expected_genus_classes
    assert outputs["family"].shape[1] == expected_family_classes

    print("    Model forward pass successful. Output shapes verified.")

    # -------------------------------------------------------------------------
    # 5. Test Loss Function
    # -------------------------------------------------------------------------
    print("\n[5] Testing HierarchicalLoss...")

    criterion = HierarchicalLoss()

    # Move targets to device
    target_tuple = (
        species_targets.to(config.DEVICE),
        genus_targets.to(config.DEVICE),
        family_targets.to(config.DEVICE),
    )

    loss, metrics = criterion(outputs, target_tuple)

    # Verification
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss).any()
    print(f"    Loss computed: {loss.item():.4f}")
    print(f"    Metrics: {metrics}")

    # -------------------------------------------------------------------------
    # 6. Test Training Loop
    # -------------------------------------------------------------------------
    print("\n[6] Testing Training Loop (1 Epoch)...")

    # Run training for 1 epoch using the engine
    trained_model = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader, num_epochs=1
    )

    print("    Training loop completed successfully.")

    # Check if best model was saved (it might not if val loss doesn't improve, but file check is good practice)
    if os.path.exists(config.BEST_MODEL_PATH):
        print(f"    Best model found at {config.BEST_MODEL_PATH}")
    else:
        print(
            "    (Note: Best model file not created, possibly due to short training duration)"
        )

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
