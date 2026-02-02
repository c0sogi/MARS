import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.utils import set_seed
from library.dataset import BirdDataset, get_transforms
from library.models import BirdClassifier
from library.engine import train_one_epoch, validate


def main():
    # 1. Setup
    print("Initializing demonstration...")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define paths
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")

    # 2. Data Loading & Preparation
    print("\n--- Data Loading ---")

    # Load metadata
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(VAL_CSV):
        raise FileNotFoundError(
            "Metadata files not found. Ensure ./metadata/train.csv exists."
        )

    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)

    # Subset data for speed (32 train samples, 16 val samples)
    df_train_subset = df_train.head(32).copy()
    df_val_subset = df_val.head(16).copy()

    print(f"Training subset size: {len(df_train_subset)}")
    print(f"Validation subset size: {len(df_val_subset)}")

    # Instantiate Datasets
    # We disable caching to demonstrate on-the-fly processing and avoid writing large files
    train_dataset = BirdDataset(
        df_train_subset,
        phase="train",
        transform=get_transforms("train"),
        load_cached_data=False,
    )

    val_dataset = BirdDataset(
        df_val_subset,
        phase="val",
        transform=get_transforms("val"),
        load_cached_data=False,
    )

    # Verify dataset length
    assert len(train_dataset) == 32
    assert len(val_dataset) == 16

    # Create DataLoaders
    batch_size = 8
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Check one batch
    images, targets = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")  # Expected: [8, 3, 224, 224]
    print(f"Batch target shape: {targets.shape}")  # Expected: [8, 19]

    assert images.shape == (batch_size, 3, 224, 224)
    assert targets.shape == (batch_size, 19)

    # 3. Model Initialization
    print("\n--- Model Initialization ---")
    model = BirdClassifier(backbone="resnet18", pretrained=True, num_classes=19)
    model = model.to(device)

    # Dummy forward pass verification
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"Model output shape: {dummy_output.shape}")
    assert dummy_output.shape == (2, 19), "Model output shape mismatch"

    # 4. Training Loop Demonstration
    print("\n--- Training Loop Demonstration ---")

    # Calculate pos_weight for BCEWithLogitsLoss
    # Simple calculation based on subset: total_samples / (positive_counts + epsilon)
    # For demo, we just use ones or a simple heuristic
    train_labels = df_train_subset[
        [c for c in df_train_subset.columns if c.startswith("species_")]
    ].values
    pos_counts = train_labels.sum(axis=0)
    # Avoid division by zero
    pos_weight_val = (len(df_train_subset) - pos_counts) / (pos_counts + 1e-5)
    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train for 2 epochs
    num_epochs = 2
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, pos_weight)
        print(f"  Training Loss: {train_loss:.4f}")

        # Verify loss is valid
        assert train_loss > 0, "Training loss should be positive"
        assert train_loss < 100, "Training loss is suspiciously high"

    # 5. Validation Demonstration
    print("\n--- Validation Demonstration ---")
    val_loss, val_auc = validate(model, val_loader, device, pos_weight)

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation ROC AUC: {val_auc:.4f}")

    # Assertions on metrics
    assert val_loss >= 0, "Validation loss must be non-negative"
    assert 0 <= val_auc <= 1, "AUC must be between 0 and 1"

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
