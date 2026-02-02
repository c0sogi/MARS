import os
import torch
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_device, load_checkpoint
from library.dataset import DogCatDataset, get_transforms
from library.models import get_model
from library.engine import train_one_epoch, evaluate, predict, EarlyStopping


def run_demo():
    print("Starting Library Usage Demo...")

    # 1. Setup & Reproducibility
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Data Preparation (Using Subsets for Speed)
    # Load metadata
    train_df_full = pd.read_csv(Config.TRAIN_METADATA)
    val_df_full = pd.read_csv(Config.VAL_METADATA)
    test_df_full = pd.read_csv(Config.TEST_METADATA)

    # Create small subsets (e.g., 16 samples) to ensure the demo finishes in seconds
    train_subset = train_df_full.head(16).reset_index(drop=True)
    val_subset = val_df_full.head(16).reset_index(drop=True)
    test_subset = test_df_full.head(16).reset_index(drop=True)

    print(
        f"Subset sizes -> Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}"
    )

    # 3. Dataset & DataLoader Instantiation
    # Retrieve image size from config for consistency (ResNet50 uses 256)
    img_size = Config.MODELS["resnet50"]["img_size"]

    # Get transforms
    train_tfm = get_transforms(mode="train", img_size=img_size)
    val_tfm = get_transforms(mode="val", img_size=img_size)

    # Initialize Datasets
    train_ds = DogCatDataset(train_subset, transform=train_tfm, mode="train")
    val_ds = DogCatDataset(val_subset, transform=val_tfm, mode="val")
    test_ds = DogCatDataset(test_subset, transform=val_tfm, mode="test")

    # Verify Dataset Logic
    sample_img, sample_target = train_ds[0]
    assert sample_img.shape == (
        3,
        img_size,
        img_size,
    ), f"Unexpected image shape: {sample_img.shape}"
    assert isinstance(sample_target, torch.Tensor), "Target should be a tensor"
    print("Dataset verification passed.")

    # Initialize DataLoaders
    batch_size = 4
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 4. Model Initialization
    # Using ResNet50 as defined in Config.MODELS
    print("Initializing Model...")
    model = get_model(
        "resnet50", pretrained=False
    )  # pretrained=False for speed/offline safety in demo
    model.to(device)

    # Verify Model Output Shape
    dummy_batch = next(iter(train_loader))[0].to(device)
    with torch.no_grad():
        output = model(dummy_batch)
    # Expecting [Batch_Size, 1] for binary classification logits
    assert output.shape == (
        batch_size,
        1,
    ), f"Model output shape mismatch: {output.shape}"
    print("Model verification passed.")

    # 5. Training Loop Demo
    print("Testing Training Engine...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Train for 1 epoch on the subset
    train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch=1)

    assert isinstance(train_loss, float), "Train loss must be a float"
    assert train_loss > 0, "Train loss must be positive"
    print(f"Training demo finished. Loss: {train_loss:.4f}")

    # 6. Evaluation Demo
    print("Testing Evaluation Engine...")
    val_loss = evaluate(model, val_loader, device)
    assert isinstance(val_loss, float), "Validation loss must be a float"
    print(f"Evaluation demo finished. Loss: {val_loss:.4f}")

    # 7. Early Stopping & Checkpointing Demo
    print("Testing Early Stopping...")
    checkpoint_path = os.path.join(Config.WORKING_DIR, "demo_checkpoint.pth")
    es = EarlyStopping(patience=2, verbose=True, path=checkpoint_path)

    # Simulate Step 1: Improvement
    es(val_loss=0.5, model=model, optimizer=optimizer)
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created."
    assert es.counter == 0, "Counter should be 0 after improvement."

    # Simulate Step 2: No Improvement
    es(val_loss=0.6, model=model, optimizer=optimizer)
    assert es.counter == 1, "Counter should increment after no improvement."

    print("Early Stopping verification passed.")

    # 8. Loading Checkpoint
    print("Testing Checkpoint Loading...")
    loaded_state = load_checkpoint(checkpoint_path, model, optimizer)
    assert loaded_state is not None, "Failed to load checkpoint."
    assert "model_state_dict" in loaded_state, "Checkpoint content invalid."
    print("Checkpoint loading passed.")

    # 9. Inference Demo
    print("Testing Inference Engine...")
    # Predict on test subset
    predictions = predict(model, test_loader, device, tta=False)

    # Verify predictions
    assert len(predictions) == len(
        test_subset
    ), "Number of predictions matches test set size."

    first_id, first_prob = predictions[0]
    # Check ID type (numpy int or similar) and Probability range
    assert 0.0 <= first_prob <= 1.0, f"Probability {first_prob} out of range [0, 1]"
    print(f"Sample Prediction -> ID: {first_id}, Probability: {first_prob:.4f}")

    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    run_demo()
