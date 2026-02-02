import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library functions
from library.utils import seed_everything
from library.dataset import get_cached_dataset, RetinopathyDataset, get_transforms
from library.model import get_model
from library.engine import train_loop, generate_submission


def create_mini_metadata(source_path, dest_path, n_samples):
    """
    Creates a smaller metadata CSV file by sampling from the source.
    Used to speed up the demonstration.
    """
    df = pd.read_csv(source_path)
    # Sample only if we have more data than requested
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    df.to_csv(dest_path, index=False)
    print(f"Created mini metadata at {dest_path} with {len(df)} samples.")
    return len(df)


def main():
    # 1. Setup
    print("Initializing demonstration...")
    seed_everything(42)

    # Configuration
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "demo_cache")

    # Hyperparameters for Demo
    IMAGE_SIZE = 224  # Smaller size for speed
    BATCH_SIZE = 16
    EPOCHS = 2
    LEARNING_RATE = 1e-4
    MODEL_NAME = "resnet18"  # Lightweight model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(WORKING_DIR, exist_ok=True)

    # 2. Data Preparation
    print("\n--- Data Preparation ---")

    # Create mini metadata for train/val to speed up training demo
    mini_train_path = os.path.join(WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(WORKING_DIR, "mini_val.csv")

    n_train = create_mini_metadata(
        os.path.join(METADATA_DIR, "train.csv"), mini_train_path, 50
    )
    n_val = create_mini_metadata(
        os.path.join(METADATA_DIR, "val.csv"), mini_val_path, 20
    )

    # Use full test set to match sample_submission.csv length requirements
    test_metadata_path = os.path.join(METADATA_DIR, "test.csv")

    # Load and process datasets using library functions
    # Train
    print("Loading Training Data...")
    train_imgs, train_lbls = get_cached_dataset(
        metadata_path=mini_train_path,
        cache_dir=CACHE_DIR,
        cache_name="mini_train",
        input_dir=INPUT_DIR,
        image_size=IMAGE_SIZE,
        load_cached_data=True,
    )

    # Val
    print("Loading Validation Data...")
    val_imgs, val_lbls = get_cached_dataset(
        metadata_path=mini_val_path,
        cache_dir=CACHE_DIR,
        cache_name="mini_val",
        input_dir=INPUT_DIR,
        image_size=IMAGE_SIZE,
        load_cached_data=True,
    )

    # Test
    print("Loading Test Data...")
    test_imgs, _ = get_cached_dataset(
        metadata_path=test_metadata_path,
        cache_dir=CACHE_DIR,
        cache_name="full_test",
        input_dir=INPUT_DIR,
        image_size=IMAGE_SIZE,
        load_cached_data=True,
    )

    # Create Dataset Objects
    train_dataset = RetinopathyDataset(
        train_imgs, train_lbls, transform=get_transforms(IMAGE_SIZE, "train")
    )
    val_dataset = RetinopathyDataset(
        val_imgs, val_lbls, transform=get_transforms(IMAGE_SIZE, "val")
    )
    test_dataset = RetinopathyDataset(
        test_imgs, None, transform=get_transforms(IMAGE_SIZE, "test")
    )

    # Validation: Check dataset sizes
    assert (
        len(train_dataset) == n_train
    ), f"Train dataset size mismatch: {len(train_dataset)} vs {n_train}"
    assert (
        len(val_dataset) == n_val
    ), f"Val dataset size mismatch: {len(val_dataset)} vs {n_val}"
    print(
        f"Datasets ready. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 3. Model Initialization
    print("\n--- Model Initialization ---")
    model = get_model(model_name=MODEL_NAME, pretrained=True)
    model = model.to(DEVICE)

    # Verification: Check model output shape
    dummy_input = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"Model: {MODEL_NAME}")
    print(f"Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (
        2,
        1,
    ), f"Expected output shape (2, 1), got {dummy_output.shape}"

    # 4. Training Loop
    print("\n--- Starting Training ---")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    save_path = os.path.join(WORKING_DIR, "demo_model.pth")

    best_kappa = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=DEVICE,
        epochs=EPOCHS,
        accumulation_steps=1,
        patience=1,
        save_path=save_path,
        scheduler=scheduler,
    )

    print(f"Training complete. Best Kappa: {best_kappa}")
    assert os.path.exists(save_path), "Model checkpoint was not saved."

    # 5. Inference & Submission
    print("\n--- Generating Submission ---")

    # Load best model weights
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))

    submission_path = os.path.join(WORKING_DIR, "submission.csv")
    generate_submission(
        model=model, test_loader=test_loader, device=DEVICE, output_path=submission_path
    )

    # Verification: Check submission file
    assert os.path.exists(submission_path), "Submission file not found."
    df_sub = pd.read_csv(submission_path)

    # Check columns
    expected_cols = ["id_code", "diagnosis"]
    assert list(df_sub.columns) == expected_cols, f"Invalid columns: {df_sub.columns}"

    # Check length matches sample_submission
    sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
    assert len(df_sub) == len(
        sample_sub
    ), f"Submission length mismatch: {len(df_sub)} vs {len(sample_sub)}"

    # Check value range
    assert (
        df_sub["diagnosis"].min() >= 0 and df_sub["diagnosis"].max() <= 4
    ), "Predictions out of range [0, 4]"

    print("\nDemonstration completed successfully.")
    print(f"Submission saved to: {submission_path}")


if __name__ == "__main__":
    main()
