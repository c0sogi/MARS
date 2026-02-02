import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.utils import set_seed, process_taxonomy
from library.dataset import HerbariumDataset, get_transforms
from library.model import CascadedEfficientNet
from library.losses import FocalLoss
from library.engine import train_model, predict_test_set


def run_demo():
    # 1. Setup and Configuration
    print("Setting up configuration...")
    set_seed(42)

    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/demo_run"
    TRAIN_META_JSON = os.path.join(INPUT_DIR, "nybg2020/train/metadata.json")

    os.makedirs(WORKING_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Process Taxonomy
    # This maps category_id to species, genus, and family indices
    print("\nProcessing taxonomy...")
    taxonomy_df = process_taxonomy(
        metadata_path=TRAIN_META_JSON,
        output_dir=WORKING_DIR,
        load_cached_data=False,  # Force re-compute for demo purposes
    )

    # Verify taxonomy structure
    assert "species_label" in taxonomy_df.columns
    assert "genus_label" in taxonomy_df.columns
    assert "family_label" in taxonomy_df.columns

    num_species = taxonomy_df["species_label"].nunique()
    num_genera = taxonomy_df["genus_label"].nunique()
    num_families = taxonomy_df["family_label"].nunique()

    print(
        f"Taxonomy loaded: {num_species} species, {num_genera} genera, {num_families} families."
    )

    # 3. Prepare Data Subsets (for speed)
    print("\nPreparing data subsets...")
    full_train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    full_val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    full_test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Select a small subset of data (e.g., 50 images)
    train_subset = full_train_df.head(50)
    val_subset = full_val_df.head(20)
    test_subset = full_test_df.head(20)

    train_subset_path = os.path.join(WORKING_DIR, "train_subset.csv")
    val_subset_path = os.path.join(WORKING_DIR, "val_subset.csv")
    test_subset_path = os.path.join(WORKING_DIR, "test_subset.csv")

    train_subset.to_csv(train_subset_path, index=False)
    val_subset.to_csv(val_subset_path, index=False)
    test_subset.to_csv(test_subset_path, index=False)

    # 4. Dataset and Dataloaders
    print("\nInitializing Datasets and Dataloaders...")
    img_size = 128  # Small size for speed
    batch_size = 8

    # Training Dataset
    train_dataset = HerbariumDataset(
        csv_path=train_subset_path,
        taxonomy_map=taxonomy_df,
        transform=get_transforms(img_size, mode="train"),
        input_dir=INPUT_DIR,
    )

    # Validation Dataset
    val_dataset = HerbariumDataset(
        csv_path=val_subset_path,
        taxonomy_map=taxonomy_df,
        transform=get_transforms(img_size, mode="val"),
        input_dir=INPUT_DIR,
    )

    # Test Dataset
    test_dataset = HerbariumDataset(
        csv_path=test_subset_path,
        taxonomy_map=None,  # No labels for test
        transform=get_transforms(img_size, mode="test"),
        input_dir=INPUT_DIR,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Verify Data Loading
    images, targets = next(iter(train_loader))
    assert images.shape == (
        batch_size,
        3,
        img_size,
        img_size,
    ), f"Unexpected image shape: {images.shape}"
    assert len(targets) == 3, "Targets should contain (species, genus, family)"
    print("Data loading verified.")

    # 5. Model Initialization
    print("\nInitializing Model...")
    # Using efficientnet_b0 for speed in this demo, though code supports b3 default
    model = CascadedEfficientNet(
        num_families=num_families,
        num_genera=num_genera,
        num_species=num_species,
        backbone_name="efficientnet_b0",
        pretrained=True,
    )
    model = model.to(device)

    # Verify Forward Pass
    with torch.no_grad():
        s_logits, g_logits, f_logits = model(images.to(device))
        assert s_logits.shape == (batch_size, num_species)
        assert g_logits.shape == (batch_size, num_genera)
        assert f_logits.shape == (batch_size, num_families)
    print("Model forward pass verified.")

    # 6. Training Setup
    criterion_species = FocalLoss(gamma=2.0)
    criterion_aux = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=1
    )

    save_path = os.path.join(WORKING_DIR, "best_model.pth")

    # 7. Run Training
    print("\nStarting Training Demo...")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion_species=criterion_species,
        criterion_aux=criterion_aux,
        device=device,
        num_epochs=1,  # Only 1 epoch for demo
        save_path=save_path,
        patience=1,
        loss_weights=(1.0, 0.5, 0.5),
        use_amp=True,
    )

    assert os.path.exists(save_path), "Model checkpoint was not saved."
    print("Training demo completed.")

    # 8. Inference
    print("\nRunning Inference Demo...")
    submission_path = os.path.join(WORKING_DIR, "submission.csv")
    predict_test_set(
        model=trained_model,
        test_loader=test_loader,
        device=device,
        output_path=submission_path,
    )

    # Verify Submission
    sub_df = pd.read_csv(submission_path)
    assert len(sub_df) == len(test_subset), "Submission file row count mismatch."
    assert (
        "Id" in sub_df.columns and "Predicted" in sub_df.columns
    ), "Submission columns missing."
    print("Inference demo completed.")

    print("\nAll demonstration steps finished successfully.")


if __name__ == "__main__":
    run_demo()
