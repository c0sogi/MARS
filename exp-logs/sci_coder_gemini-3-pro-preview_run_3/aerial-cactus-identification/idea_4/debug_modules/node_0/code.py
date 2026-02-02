import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Import from the provided library files
from library.utils import seed_everything, save_state_dict
from library.models import ModifiedWideSEResNet, ModifiedDenseNet
from library.data import get_transforms, load_and_cache_data, CactusDataset
from library.trainer import train_one_epoch, validate, predict_tta


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    print("Initializing setup...")
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Define paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n--- Loading Data ---")
    # Load a small subset of data for demonstration speed (debug_subset_size)
    # We force reload_cached_data=False to demonstrate the loading logic from metadata
    train_imgs, train_ids, train_lbls = load_and_cache_data(
        metadata_path=os.path.join(METADATA_DIR, "train_metadata.csv"),
        input_dir=INPUT_DIR,
        cache_prefix="demo_train",
        load_cached_data=False,
        debug_subset_size=64,
    )

    val_imgs, val_ids, val_lbls = load_and_cache_data(
        metadata_path=os.path.join(METADATA_DIR, "val_metadata.csv"),
        input_dir=INPUT_DIR,
        cache_prefix="demo_val",
        load_cached_data=False,
        debug_subset_size=32,
    )

    # Verification of loaded data
    print(f"Train Images Shape: {train_imgs.shape}")
    print(f"Val Images Shape: {val_imgs.shape}")
    assert train_imgs.shape == (64, 32, 32, 3), "Train images shape mismatch"
    assert val_imgs.shape == (32, 32, 32, 3), "Val images shape mismatch"
    assert len(train_lbls) == 64
    assert len(val_lbls) == 32

    # -------------------------------------------------------------------------
    # 3. Dataset and DataLoader
    # -------------------------------------------------------------------------
    print("\n--- Creating Datasets and Loaders ---")
    train_dataset = CactusDataset(
        images=train_imgs,
        labels=train_lbls,
        image_ids=train_ids,
        transform=get_transforms("train"),
    )

    val_dataset = CactusDataset(
        images=val_imgs,
        labels=val_lbls,
        image_ids=val_ids,
        transform=get_transforms("valid"),
    )

    # Small batch size for the small subset
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # -------------------------------------------------------------------------
    # 4. Model Instantiation & Verification
    # -------------------------------------------------------------------------
    print("\n--- Instantiating Models ---")
    # Instantiate ModifiedWideSEResNet with reduced layers for speed in this demo
    model_resnet = ModifiedWideSEResNet(layers=[1, 1, 1, 1], num_classes=1).to(device)

    # Instantiate ModifiedDenseNet with reduced config for speed
    model_densenet = ModifiedDenseNet(
        growth_rate=12, block_config=(2, 2, 2, 2), num_classes=1
    ).to(device)

    # Verify Forward Pass
    dummy_input = torch.randn(2, 3, 32, 32).to(device)

    with torch.no_grad():
        out_resnet = model_resnet(dummy_input)
        out_densenet = model_densenet(dummy_input)

    print(f"ResNet Output Shape: {out_resnet.shape}")
    print(f"DenseNet Output Shape: {out_densenet.shape}")

    assert out_resnet.shape == (2, 1), "ResNet output shape incorrect"
    assert out_densenet.shape == (2, 1), "DenseNet output shape incorrect"

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Running Training Loop (ResNet) ---")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model_resnet.parameters(), lr=1e-3)

    # Run for 2 epochs to verify logic
    for epoch in range(2):
        print(f"Epoch {epoch + 1}/2")

        # Train
        train_loss = train_one_epoch(
            model=model_resnet,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            alpha=0.4,  # Mixup alpha
        )

        # Validate
        val_loss, val_auc = validate(
            model=model_resnet, loader=val_loader, criterion=criterion, device=device
        )

        # Assertions to ensure training is proceeding numerically
        assert not np.isnan(train_loss), "Training loss is NaN"
        assert not np.isnan(val_loss), "Validation loss is NaN"
        print(
            f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

    # -------------------------------------------------------------------------
    # 6. Model Saving
    # -------------------------------------------------------------------------
    print("\n--- Saving Model ---")
    save_path = os.path.join(WORKING_DIR, "demo_resnet.pth")
    save_state_dict(model_resnet, save_path)

    if os.path.exists(save_path):
        print(f"Model successfully saved to {save_path}")
    else:
        raise AssertionError("Model file was not created.")

    # -------------------------------------------------------------------------
    # 7. Inference (TTA) Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Running TTA Prediction ---")
    # Using val_loader as a proxy for test data for this demonstration
    preds = predict_tta(model_resnet, val_loader, device)

    # Verify predictions format
    assert isinstance(preds, dict), "Predictions should be a dictionary"
    assert len(preds) == 32, f"Expected 32 predictions, got {len(preds)}"

    # Check a single prediction
    sample_id = val_ids[0]
    sample_prob = preds[sample_id]
    print(f"Sample ID: {sample_id}, Predicted Probability: {sample_prob:.4f}")

    assert 0.0 <= sample_prob <= 1.0, "Probability out of bounds"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
