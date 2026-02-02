import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.utils import seed_everything, get_taxonomy_mappings
from library.dataset import (
    get_species_mapping,
    get_transforms,
    HierarchicalPlantDataset,
)
from library.model import HierarchicalConvNeXt
from library.trainer import HierarchicalTrainer


def main():
    print("==== Starting Library Usage Demonstration ====")

    # 1. Setup
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Define paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    TRAIN_META_JSON = os.path.join(INPUT_DIR, "train_metadata.json")

    # 2. Load and Sample Data (Optimization for Speed)
    # We load the full metadata but only use a small sample for this demo
    print("Loading and sampling metadata...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Sample subsets to ensure the script finishes quickly
    # Ensure we pick samples that exist on disk (which they should based on metadata generation)
    train_subset = train_df.sample(n=64, random_state=42).reset_index(drop=True)
    val_subset = val_df.sample(n=32, random_state=42).reset_index(drop=True)
    test_subset = test_df.sample(n=32, random_state=42).reset_index(drop=True)

    print(f"Train subset size: {len(train_subset)}")
    print(f"Val subset size: {len(val_subset)}")
    print(f"Test subset size: {len(test_subset)}")

    # 3. Generate Mappings
    print("Generating taxonomy mappings...")
    # Get Family and Genus mappings
    cat_to_fam, cat_to_gen, num_families, num_genera = get_taxonomy_mappings(
        metadata_json_path=TRAIN_META_JSON,
        load_cached_data=False,  # Force recompute for demo purposes
    )

    # Get Species mapping (0..N-1)
    cat_to_label, num_species = get_species_mapping(
        train_csv_path=TRAIN_CSV, load_cached_data=False
    )

    print(
        f"Taxonomy Stats: Families={num_families}, Genera={num_genera}, Species={num_species}"
    )

    # 4. Instantiate Datasets and Loaders
    print("Creating Datasets and DataLoaders...")
    batch_size = 16
    image_size = 128  # Reduced size for speed

    # Training Data
    train_ds = HierarchicalPlantDataset(
        df=train_subset,
        transform=get_transforms(mode="train", image_size=image_size),
        cat_to_fam=cat_to_fam,
        cat_to_gen=cat_to_gen,
        cat_to_label=cat_to_label,
        mode="train",
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    # Validation Data
    val_ds = HierarchicalPlantDataset(
        df=val_subset,
        transform=get_transforms(mode="val", image_size=image_size),
        cat_to_fam=cat_to_fam,
        cat_to_gen=cat_to_gen,
        cat_to_label=cat_to_label,
        mode="val",
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Test Data
    test_ds = HierarchicalPlantDataset(
        df=test_subset,
        transform=get_transforms(mode="val", image_size=image_size),
        mode="test",  # Mappings not needed for test mode
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # 5. Initialize Model
    print("Initializing HierarchicalConvNeXt model...")
    # Using pretrained=False to avoid downloading weights during this timed run
    model = HierarchicalConvNeXt(
        num_families=num_families,
        num_genera=num_genera,
        num_species=num_species,
        pretrained=False,
    )

    # Verify model output shape with a dummy batch
    dummy_input = torch.randn(2, 3, image_size, image_size)
    with torch.no_grad():
        outputs = model(dummy_input)

    assert "family" in outputs
    assert "genus" in outputs
    assert "species" in outputs
    assert outputs["species"].shape == (2, num_species)
    print("Model forward pass verification successful.")

    # 6. Initialize Trainer and Train
    print("Initializing Trainer...")
    trainer = HierarchicalTrainer(
        model=model,
        device=device,
        num_families=num_families,
        num_genera=num_genera,
        num_species=num_species,
        learning_rate_backbone=1e-4,
        learning_rate_head=1e-3,
    )

    print("Starting Training Loop (2 Epochs)...")
    trainer.fit(train_loader, val_loader, num_epochs=2, patience=1)

    # 7. Verification of Results
    checkpoint_path = "./working/idea_2/best_model.pth"
    if os.path.exists(checkpoint_path):
        print(f"Success: Checkpoint found at {checkpoint_path}")
    else:
        raise FileNotFoundError("Training finished but no checkpoint was saved.")

    # 8. Inference Demonstration
    print("Running Inference on Test Subset...")
    # Load best state
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    predictions = []
    image_ids = []

    with torch.no_grad():
        for images, _, ids in test_loader:
            images = images.to(device)
            outputs = model(images)

            # Get species predictions
            preds = torch.argmax(outputs["species"], dim=1).cpu().numpy()

            predictions.extend(preds)
            image_ids.extend(ids)

    # Create submission DataFrame
    # Note: The model predicts the mapped index (0..N-1).
    # We need to map it back to the original category_id using cat_to_label reversed.
    label_to_cat = {v: k for k, v in cat_to_label.items()}
    predicted_category_ids = [label_to_cat[p] for p in predictions]

    submission_df = pd.DataFrame({"Id": image_ids, "Predicted": predicted_category_ids})

    print("Sample Submission Head:")
    print(submission_df.head())

    assert len(submission_df) == len(test_subset)
    print("==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    main()
