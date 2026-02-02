import os
import shutil
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Import components from the provided library files
from library.config import DataConfig, TrainConfig, ModelConfig, WORKING_DIR
from library.utils import seed_everything, ensure_dir, get_device
from library.transforms import get_transforms
from library.dataset import DogCatDataset
from library.models import CustomEnsembleModel
from library.engine import train_one_epoch, evaluate


def run_demonstration():
    print("=== Starting Library Demonstration ===\n")

    # 1. Setup Environment
    # -------------------------------------------------------------------------
    print("[1] Setting up environment...")
    seed_everything(42)
    device = get_device()
    print(f"    Device detected: {device}")

    # Define temporary directories for the demo
    demo_meta_dir = os.path.join(WORKING_DIR, "demo_metadata")
    ensure_dir(demo_meta_dir)

    # 2. Prepare Data Configuration (Mini Subset)
    # -------------------------------------------------------------------------
    print("\n[2] Preparing mini-datasets for rapid verification...")

    # Load the existing metadata
    # We use a very small subset (32 train, 16 val) to ensure the demo runs instantly
    train_csv_path = "./metadata/train.csv"
    val_csv_path = "./metadata/val.csv"

    if not os.path.exists(train_csv_path) or not os.path.exists(val_csv_path):
        raise FileNotFoundError("Metadata files not found in ./metadata/")

    full_train_df = pd.read_csv(train_csv_path)
    full_val_df = pd.read_csv(val_csv_path)

    mini_train_df = full_train_df.head(32).copy()
    mini_val_df = full_val_df.head(16).copy()

    # Save mini metadata to working directory
    mini_train_path = os.path.join(demo_meta_dir, "mini_train.csv")
    mini_val_path = os.path.join(demo_meta_dir, "mini_val.csv")

    mini_train_df.to_csv(mini_train_path, index=False)
    mini_val_df.to_csv(mini_val_path, index=False)

    # Instantiate DataConfig with the mini paths
    data_config = DataConfig(
        train_csv=mini_train_path, val_csv=mini_val_path, num_classes=1
    )
    print(
        f"    Created mini_train.csv ({len(mini_train_df)} rows) and mini_val.csv ({len(mini_val_df)} rows)."
    )

    # 3. Verify Transforms
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Data Transforms...")
    input_size = 224
    train_transform = get_transforms(image_size=input_size, mode="train")
    val_transform = get_transforms(image_size=input_size, mode="val")

    # Check if transforms are callable
    assert callable(train_transform), "Train transform is not callable"
    assert callable(val_transform), "Val transform is not callable"
    print("    Transforms instantiated successfully.")

    # 4. Verify Dataset Loading
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Dataset Logic...")
    train_dataset = DogCatDataset(
        mini_train_df, transform=train_transform, mode="train"
    )

    # Retrieve a single sample
    sample_img, sample_label = train_dataset[0]

    # Assertions to ensure data integrity
    assert isinstance(
        sample_img, torch.Tensor
    ), "Dataset must return a torch.Tensor for the image"
    assert sample_img.shape == (
        3,
        input_size,
        input_size,
    ), f"Image shape mismatch. Expected (3, {input_size}, {input_size}), got {sample_img.shape}"
    assert isinstance(
        sample_label, torch.Tensor
    ), "Dataset must return a torch.Tensor for the label"
    assert sample_label.dtype == torch.float32, "Label tensor must be float32"

    print(
        f"    Sample retrieval successful. Image Shape: {sample_img.shape}, Label: {sample_label.item()}"
    )

    # 5. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Model Architecture...")
    # Create a lightweight ModelConfig for the demo (ResNet18)
    # We use pretrained=False to avoid downloading weights during the demo
    demo_model_config = ModelConfig(
        model_name="resnet18",
        input_size=input_size,
        batch_size=8,
        learning_rate=1e-4,
        use_multi_sample_dropout=True,
        dropout_rates=[0.1, 0.2],
    )

    model = CustomEnsembleModel(
        config=demo_model_config, num_classes=1, pretrained=False
    )
    model.to(device)

    # Perform a dummy forward pass to verify dimensions
    dummy_input = torch.randn(2, 3, input_size, input_size).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"
    print("    Model initialized and forward pass verified.")

    # 6. Verify Training Engine
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Training Engine (1 Epoch)...")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=demo_model_config.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 workers for simple demo to avoid overhead
    )

    val_dataset = DogCatDataset(mini_val_df, transform=val_transform, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=demo_model_config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Setup Optimizer and Loss
    optimizer = AdamW(model.parameters(), lr=demo_model_config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Run Training Step
    print("    Executing train_one_epoch()...")
    train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)

    # Run Evaluation Step
    print("    Executing evaluate()...")
    val_loss, val_acc = evaluate(model, val_loader, device, criterion)

    # Check results
    print(
        f"    Results -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    assert not pd.isna(train_loss), "Train loss returned NaN"
    assert not pd.isna(val_loss), "Validation loss returned NaN"
    assert 0.0 <= val_acc <= 1.0, "Validation accuracy out of bounds"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
