import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam

# Import provided library modules
from library.utils import seed_everything
from library.image_processing import generate_dual_views
from library.dataset import LungDataset
from library.architecture import AttentionFusedDualAxisNet
from library.loss import ModifiedLaplaceLoss
from library.engine import fit


def main():
    # ==========================================
    # 1. Setup and Configuration
    # ==========================================
    print("Initializing demonstration...")
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Directory for saving checkpoints
    checkpoint_dir = "./working/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ==========================================
    # 2. Verify Image Processing
    # ==========================================
    print("\n[1/5] Verifying Image Processing...")

    # Load training metadata to get a valid patient ID and path
    train_meta = pd.read_csv("./metadata/train.csv")
    sample_row = train_meta.iloc[0]
    patient_id = sample_row["Patient"]
    dicom_dir = os.path.join("./input", sample_row["dicom_dir"])

    print(f"Processing patient: {patient_id}")
    # Generate views (Axial + Coronal)
    # Note: This function caches results to ./working/idea_5
    dual_views = generate_dual_views(dicom_dir, patient_id, load_cached_data=False)

    print(f"Output shape: {dual_views.shape}")

    # Expecting shape (2, 224, 224, 3) -> (Views, Height, Width, Channels)
    assert dual_views.shape == (
        2,
        224,
        224,
        3,
    ), f"Image processing failed. Expected (2, 224, 224, 3), got {dual_views.shape}"

    print("Image processing verification passed.")

    # ==========================================
    # 3. Verify Dataset and DataLoader
    # ==========================================
    print("\n[2/5] Verifying Dataset and DataLoader...")

    # Initialize dataset
    # We use a subset to keep the demo fast
    full_dataset = LungDataset(mode="train")
    subset_indices = list(range(8))  # Use only 8 samples
    train_subset = Subset(full_dataset, subset_indices)

    batch_size = 4
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)

    # Fetch one batch
    batch = next(iter(train_loader))
    images = batch["image"]
    tabular = batch["tabular"]

    print(f"Batch Image Tensor Shape: {images.shape}")
    print(f"Batch Tabular Tensor Shape: {tabular.shape}")

    # Verify shapes
    # Images: (Batch, 2_Views, 3_Channels, 224, 224)
    assert images.shape == (
        batch_size,
        2,
        3,
        224,
        224,
    ), "Incorrect image batch dimensions."
    # Tabular: (Batch, 6_Features)
    assert tabular.shape == (batch_size, 6), "Incorrect tabular batch dimensions."

    print("Dataset verification passed.")

    # ==========================================
    # 4. Verify Model Architecture
    # ==========================================
    print("\n[3/5] Verifying Model Architecture...")

    # Initialize model
    # pretrained=False to avoid downloading weights during this demo
    model = AttentionFusedDualAxisNet(
        tabular_input_dim=6, feature_dim=1280, pretrained=False
    )
    model.to(device)

    # Move batch to device
    images = images.to(device)
    tabular = tabular.to(device)

    # Forward pass
    alpha, sigma_base, sigma_growth = model(images, tabular)

    print(f"Alpha (Slope) Shape: {alpha.shape}")
    print(f"Sigma Base Shape: {sigma_base.shape}")

    # Verify output shapes and constraints
    assert alpha.shape == (batch_size,), "Alpha output shape mismatch."
    assert sigma_base.shape == (batch_size,), "Sigma base output shape mismatch."
    assert sigma_growth.shape == (batch_size,), "Sigma growth output shape mismatch."

    # Sigma values must be positive (enforced by Softplus in architecture)
    assert (sigma_base >= 0).all(), "Model produced negative sigma_base values."
    assert (sigma_growth >= 0).all(), "Model produced negative sigma_growth values."

    print("Model architecture verification passed.")

    # ==========================================
    # 5. Verify Loss Function
    # ==========================================
    print("\n[4/5] Verifying Loss Function...")

    loss_fn = ModifiedLaplaceLoss()

    # Extract other batch data for loss calculation
    time = batch["time"].to(device)
    target_fvc = batch["fvc"].to(device)
    baseline_fvc = batch["baseline_fvc"].to(device)

    # Calculate loss
    loss = loss_fn(alpha, sigma_base, sigma_growth, time, baseline_fvc, target_fvc)

    print(f"Calculated Loss: {loss.item():.4f}")

    # Verify loss is a valid number
    assert not torch.isnan(loss), "Loss function returned NaN."
    assert not torch.isinf(loss), "Loss function returned Infinity."

    print("Loss function verification passed.")

    # ==========================================
    # 6. Verify Training Engine (Fit)
    # ==========================================
    print("\n[5/5] Verifying Training Loop...")

    # Prepare a small validation set
    val_dataset = LungDataset(mode="val")
    val_subset = Subset(val_dataset, list(range(4)))  # 4 samples for validation
    val_loader = DataLoader(val_subset, batch_size=4)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Save path
    save_path = os.path.join(checkpoint_dir, "best_model_demo.pth")

    # Run training for 2 epochs
    print("Starting short training run...")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        num_epochs=2,
        patience=1,
        save_path=save_path,
    )

    # Check if model was saved
    assert os.path.exists(save_path), "Training loop failed to save the best model."

    print("Training loop verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
