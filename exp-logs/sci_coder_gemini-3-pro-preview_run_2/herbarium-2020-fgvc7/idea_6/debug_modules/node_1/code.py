import os
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader
import torch.optim as optim

# Import provided library modules
from library.taxonomy_utils import build_taxonomy_mapping, get_taxonomy_stats, set_seed
from library.dataset import HerbariumDataset, get_transforms
from library.model import get_model
from library.losses import HierarchicalLoss
from library.train_utils import train_model


def main():
    # 1. Setup and Configuration
    print("Setting up demonstration...")
    set_seed(42)

    # Define paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/demo_usage"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Demonstrate Taxonomy Mapping (taxonomy_utils.py)
    print("\n--- 1. Taxonomy Mapping ---")
    # We use the provided metadata file to build the mapping
    metadata_json_path = os.path.join(INPUT_DIR, "nybg2020/train/metadata.json")

    # Build mapping (caches to working dir)
    df_taxonomy = build_taxonomy_mapping(
        metadata_path=metadata_json_path, cache_dir=WORKING_DIR, load_cached_data=True
    )

    # Verify mapping
    assert isinstance(df_taxonomy, pd.DataFrame)
    assert "genus_id" in df_taxonomy.columns
    assert "family_id" in df_taxonomy.columns

    stats = get_taxonomy_stats(df_taxonomy)
    print(f"Taxonomy Stats: {stats}")

    num_species = stats["num_species"]
    num_genus = stats["num_genera"]
    num_family = stats["num_families"]

    # 3. Demonstrate Dataset Loading (dataset.py)
    print("\n--- 2. Dataset and Transforms ---")

    # Create a small subset of the training data for demonstration purposes
    # This ensures the code runs quickly without loading the full dataset
    full_train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    subset_size = 32  # Small batch for demo
    subset_train_path = os.path.join(WORKING_DIR, "train_subset.csv")
    subset_val_path = os.path.join(WORKING_DIR, "val_subset.csv")

    # Save subsets
    full_train_df.head(subset_size).to_csv(subset_train_path, index=False)
    full_train_df.iloc[subset_size : subset_size * 2].to_csv(
        subset_val_path, index=False
    )

    # Define transforms
    train_transform = get_transforms(
        split="train", image_size=128
    )  # Small size for speed
    val_transform = get_transforms(split="val", image_size=128)

    # Instantiate Datasets
    train_dataset = HerbariumDataset(
        csv_path=subset_train_path,
        taxonomy_map=df_taxonomy,
        transform=train_transform,
        is_test=False,
        input_root=INPUT_DIR,
    )

    val_dataset = HerbariumDataset(
        csv_path=subset_val_path,
        taxonomy_map=df_taxonomy,
        transform=val_transform,
        is_test=False,
        input_root=INPUT_DIR,
    )

    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size: {len(val_dataset)}")

    # Verify __getitem__
    img, target = train_dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 128, 128)  # Channels, Height, Width
    assert isinstance(target, dict)
    assert "species" in target and "genus" in target and "family" in target
    print("Dataset verification successful: Image shape and target keys correct.")

    # Create DataLoaders
    batch_size = 8
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # 4. Demonstrate Model Instantiation (model.py)
    print("\n--- 3. Model Instantiation ---")
    # Initialize model with correct number of classes
    # Using pretrained=False to avoid downloading weights during this demo run
    model = get_model(
        num_species=num_species,
        num_genus=num_genus,
        num_family=num_family,
        pretrained=False,
    )
    model.to(device)

    # Verify Forward Pass
    dummy_input = torch.randn(2, 3, 128, 128).to(device)
    with torch.no_grad():
        outputs = model(dummy_input)

    assert "species" in outputs
    assert outputs["species"].shape == (2, num_species)
    assert outputs["genus"].shape == (2, num_genus)
    assert outputs["family"].shape == (2, num_family)
    print("Model forward pass successful. Output shapes correct.")

    # 5. Demonstrate Loss Function (losses.py)
    print("\n--- 4. Loss Function ---")
    criterion = HierarchicalLoss(
        weights={"species": 1.0, "genus": 0.5, "family": 0.5}, focal_gamma=2.0
    )

    # Create dummy targets matching the dummy input batch size
    dummy_targets = {
        "species": torch.randint(0, num_species, (2,)).to(device),
        "genus": torch.randint(0, num_genus, (2,)).to(device),
        "family": torch.randint(0, num_family, (2,)).to(device),
    }

    loss, metrics = criterion(outputs, dummy_targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert "loss_species" in metrics
    print(f"Loss calculation successful. Total Loss: {loss.item():.4f}")

    # 6. Demonstrate Training Loop (train_utils.py)
    print("\n--- 5. Training Loop Execution ---")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Run training for 1 epoch on the subset
    # This validates the integration of all components
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=1,
        device=device,
        save_dir=WORKING_DIR,
        patience=1,
    )

    assert len(history["train_loss"]) == 1
    assert len(history["val_f1"]) == 1
    print("Training loop completed successfully.")
    print(f"Final Train Loss: {history['train_loss'][0]:.4f}")
    print(f"Final Val F1: {history['val_f1'][0]:.4f}")

    # Cleanup
    print("\nCleaning up...")
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    print("Demonstration finished.")


if __name__ == "__main__":
    main()
