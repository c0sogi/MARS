import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

# Import provided library modules
from library.utils import set_seed, get_taxonomy_mappings
from library.model import MultiTaskResNet
from library.dataset import HerbariumDataset
from library.trainer import Trainer


def main():
    print("=== Starting Demonstration ===")

    # 1. Configuration
    SEED = 42
    BATCH_SIZE = 8
    EPOCHS = 1
    DEBUG_LIMIT = 50  # Limit dataset size for speed
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/demo"
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model.pth")

    os.makedirs(WORKING_DIR, exist_ok=True)

    # Set reproducibility
    set_seed(SEED)
    print(f"Device: {DEVICE}")

    # 2. Taxonomy Mappings
    print("\n[Step 1] Loading Taxonomy Mappings...")
    # This function parses the metadata JSON to map category_ids to family/order
    # It handles caching internally.
    (
        species_to_family,
        species_to_order,
        species_to_idx,
        idx_to_species,
        num_families,
        num_orders,
        num_species,
    ) = get_taxonomy_mappings()

    print(f"  Num Species: {num_species}")
    print(f"  Num Families: {num_families}")
    print(f"  Num Orders: {num_orders}")

    # Assertion to verify mappings are populated
    assert num_species > 0, "Species count should be positive"
    assert len(species_to_idx) == num_species, "Mapping size mismatch"

    # 3. Data Preparation
    print("\n[Step 2] Preparing Datasets (Debug Mode)...")

    # Load metadata CSVs
    df_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    df_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Subset for speed
    df_train = df_train.head(DEBUG_LIMIT)
    df_val = df_val.head(DEBUG_LIMIT)
    df_test = df_test.head(DEBUG_LIMIT)

    print(f"  Train samples: {len(df_train)}")
    print(f"  Val samples: {len(df_val)}")
    print(f"  Test samples: {len(df_test)}")

    # Define Transforms
    # Using simple resize for demonstration speed
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    taxonomy_maps = (species_to_idx, species_to_family, species_to_order)

    # Instantiate Datasets
    train_dataset = HerbariumDataset(
        df_train, INPUT_DIR, taxonomy_maps, transform=transform
    )
    val_dataset = HerbariumDataset(
        df_val, INPUT_DIR, taxonomy_maps, transform=transform
    )
    test_dataset = HerbariumDataset(
        df_test, INPUT_DIR, transform=transform, is_test=True
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Verification: Check a single batch
    images, species_targets, family_targets, order_targets = next(iter(train_loader))
    print(f"  Batch Image Shape: {images.shape}")
    assert images.shape == (BATCH_SIZE, 3, 224, 224), "Incorrect image tensor shape"
    assert species_targets.shape == (BATCH_SIZE,), "Incorrect target shape"

    # 4. Model Initialization & Verification
    print("\n[Step 3] Initializing MultiTaskResNet...")
    model = MultiTaskResNet(num_species, num_families, num_orders, pretrained=False).to(
        DEVICE
    )

    # Verification: Forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(DEVICE)
    sp_out, fam_out, ord_out = model(dummy_input)

    print(
        f"  Model Output Shapes: Species={sp_out.shape}, Family={fam_out.shape}, Order={ord_out.shape}"
    )
    assert sp_out.shape == (2, num_species), "Species output shape mismatch"
    assert fam_out.shape == (2, num_families), "Family output shape mismatch"
    assert ord_out.shape == (2, num_orders), "Order output shape mismatch"

    # 5. Training
    print("\n[Step 4] Training Loop...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        save_path=MODEL_SAVE_PATH,
    )

    # Run training
    trainer.fit(train_loader, val_loader, epochs=EPOCHS, patience=1)

    # Verify model file was created (if validation improved) or at least training finished
    # Note: If val score doesn't improve, file might not be saved, but fit() should complete.
    # For this demo, we check if the code reached this point successfully.
    print("  Training finished.")

    # 6. Inference
    print("\n[Step 5] Generating Submission...")
    # We use the current model weights if best model wasn't saved (due to short training)
    # The Trainer.predict method handles loading if file exists, or warns if not.
    # To ensure predict works even if no best model was saved, we can save manually or rely on fallback.
    if not os.path.exists(MODEL_SAVE_PATH):
        print("  Saving current model weights for inference...")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    trainer.predict(test_loader, idx_to_species, SUBMISSION_PATH)

    # Verify Submission
    if os.path.exists(SUBMISSION_PATH):
        df_sub = pd.read_csv(SUBMISSION_PATH)
        print(f"  Submission generated with {len(df_sub)} rows.")
        print(df_sub.head())

        assert (
            "Id" in df_sub.columns and "Predicted" in df_sub.columns
        ), "Submission columns missing"
        assert len(df_sub) == len(df_test), "Submission row count mismatch"
        print("  Submission verification passed.")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
