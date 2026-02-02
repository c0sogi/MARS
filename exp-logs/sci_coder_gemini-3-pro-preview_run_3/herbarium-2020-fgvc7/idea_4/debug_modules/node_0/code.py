import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from provided library files
from library.utils import seed_everything, get_device
from library.data_mappings import get_species_to_genus_mapping
from library.dataset import get_dataloaders, HerbariumDataset, get_transforms
from library.model import HierarchicalResNet
from library.engine import train_model, generate_submission


def run_demo():
    # 1. Setup and Configuration
    print("--- 1. Setup ---")
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    # Define working directories
    work_dir = "./working/demo_test"
    os.makedirs(work_dir, exist_ok=True)

    # 2. Data Mappings
    print("\n--- 2. Testing Data Mappings ---")
    # We use the default path for metadata.json.
    # Note: Parsing the large JSON might take a few seconds.
    # We use a specific cache dir to avoid conflicts or use existing ones if available.
    mapping_cache_dir = os.path.join(work_dir, "mapping_cache")

    species_to_genus, num_genera = get_species_to_genus_mapping(
        json_path="./input/nybg2020/train/metadata.json",
        cache_dir=mapping_cache_dir,
        load_cached_data=True,
    )

    print(f"Number of genera found: {num_genera}")
    print(f"Number of species mapped: {len(species_to_genus)}")

    # Assertions
    assert isinstance(species_to_genus, dict), "species_to_genus should be a dictionary"
    assert num_genera > 0, "num_genera should be greater than 0"

    # 3. Data Loading (Subset)
    print("\n--- 3. Testing Data Loading (Subset) ---")
    # We limit the sample size to 100 to ensure the script runs quickly.
    batch_size = 8
    image_size = 128  # Small size for speed

    train_loader, val_loader, num_classes, num_genera_loader = get_dataloaders(
        image_size=image_size,
        batch_size=batch_size,
        num_workers=2,
        load_cached_data=True,  # Will reuse the mapping we just verified if cache dir matches, but get_dataloaders uses hardcoded CACHE_DIR in dataset.py.
        # dataset.py uses "./working/idea_4/". We let it handle its own caching.
        sample_limit=100,
    )

    print(f"Num classes: {num_classes}")
    print(f"Num genera (from loader): {num_genera_loader}")

    # Fetch one batch to verify structure
    images, species_ids, genus_ids = next(iter(train_loader))

    print(
        f"Batch shapes - Images: {images.shape}, Species: {species_ids.shape}, Genus: {genus_ids.shape}"
    )

    # Assertions
    assert images.shape == (
        batch_size,
        3,
        image_size,
        image_size,
    ), "Incorrect image tensor shape"
    assert species_ids.shape == (batch_size,), "Incorrect species_ids shape"
    assert genus_ids.shape == (batch_size,), "Incorrect genus_ids shape"
    assert (
        num_genera_loader == num_genera
    ), "Mismatch in genus count between mapping and loader"

    # 4. Model Initialization
    print("\n--- 4. Testing Model Initialization ---")
    model = HierarchicalResNet(
        num_species=num_classes,
        num_genera=num_genera,
        backbone_name="resnet50",
        pretrained=True,
    )
    model = model.to(device)

    # Test forward pass
    species_logits, genus_logits = model(
        images.to(device), species_label=species_ids.to(device)
    )

    print(
        f"Output shapes - Species Logits: {species_logits.shape}, Genus Logits: {genus_logits.shape}"
    )

    # Assertions
    assert species_logits.shape == (
        batch_size,
        num_classes,
    ), "Incorrect species logits shape"
    assert genus_logits.shape == (
        batch_size,
        num_genera,
    ), "Incorrect genus logits shape"

    # 5. Training Loop (1 Epoch)
    print("\n--- 5. Testing Training Loop ---")
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Train for just 1 epoch on the subset
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=1,
        scheduler=scheduler,
        genus_weight=0.5,
        patience=1,
        checkpoint_dir=os.path.join(work_dir, "checkpoints"),
    )
    print("Training finished.")

    # 6. Inference and Submission
    print("\n--- 6. Testing Inference and Submission ---")

    # Create a small test dataloader manually to avoid processing the full test set
    test_csv_path = "./metadata/test.csv"
    test_df = pd.read_csv(test_csv_path).head(20)  # Only 20 samples

    test_dataset = HerbariumDataset(
        df=test_df,
        transform=get_transforms(image_size, is_training=False),
        species_to_genus_map=None,
        is_test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    output_dir = os.path.join(work_dir, "submission")
    generate_submission(
        model=trained_model,
        dataloader=test_loader,
        device=device,
        output_dir=output_dir,
    )

    submission_file = os.path.join(output_dir, "submission.csv")
    assert os.path.exists(submission_file), "Submission file was not created"

    # Verify submission content
    sub_df = pd.read_csv(submission_file)
    print(f"Submission shape: {sub_df.shape}")
    assert list(sub_df.columns) == ["Id", "Predicted"], "Incorrect submission columns"
    assert len(sub_df) == 20, "Submission length mismatch"
    assert not sub_df.isnull().values.any(), "Submission contains NaNs"

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
