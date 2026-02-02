import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import DataLoader

# Import provided library modules
from library.utils import seed_everything, get_device
from library.dataset import (
    DogBreedDataset,
    get_transforms,
    load_processed_metadata,
)
from library.model import ResNet50Baseline
from library.engine import train_model, predict, save_submission

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants for the demonstration
BATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_CLASSES = 120
NUM_EPOCHS = 1  # Reduced for speed
PATIENCE = 1
SUBSET_SIZE_TRAIN = 64  # Small subset for quick demo
SUBSET_SIZE_VAL = 32
SUBSET_SIZE_TEST = 32
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "cache_demo")
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model_demo.pth")
SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission_demo.csv")


def main():
    print("Initializing Demonstration...")

    # 1. Setup
    seed_everything(42)
    device = get_device()
    print(f"Device selected: {device}")

    # 2. Data Loading & Preparation
    print("\n--- Data Preparation ---")

    # Load metadata
    # We use a custom cache dir to avoid interfering with other potential runs
    train_df, val_df, test_df, class_names = load_processed_metadata(
        load_cached_data=False, cache_dir=CACHE_DIR
    )

    # Validate metadata loading
    assert (
        len(class_names) == NUM_CLASSES
    ), f"Expected {NUM_CLASSES} classes, found {len(class_names)}"
    assert "label_idx" in train_df.columns, "Train DataFrame missing label_idx"

    # Subsample data for speed optimization
    train_df_sub = train_df.sample(
        n=min(len(train_df), SUBSET_SIZE_TRAIN), random_state=42
    ).reset_index(drop=True)
    val_df_sub = val_df.sample(
        n=min(len(val_df), SUBSET_SIZE_VAL), random_state=42
    ).reset_index(drop=True)
    test_df_sub = test_df.sample(
        n=min(len(test_df), SUBSET_SIZE_TEST), random_state=42
    ).reset_index(drop=True)

    print(
        f"Subsampled Data: Train={len(train_df_sub)}, Val={len(val_df_sub)}, Test={len(test_df_sub)}"
    )

    # Create Datasets
    train_dataset = DogBreedDataset(
        train_df_sub,
        transform=get_transforms(phase="train", image_size=IMAGE_SIZE),
        mode="train",
    )
    val_dataset = DogBreedDataset(
        val_df_sub,
        transform=get_transforms(phase="val", image_size=IMAGE_SIZE),
        mode="val",
    )
    test_dataset = DogBreedDataset(
        test_df_sub,
        transform=get_transforms(phase="test", image_size=IMAGE_SIZE),
        mode="test",
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Validate DataLoader output
    sample_imgs, sample_labels = next(iter(train_loader))
    print(f"Batch Shape: {sample_imgs.shape}")
    assert sample_imgs.shape == (
        BATCH_SIZE,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ), "Incorrect batch image shape"
    assert sample_labels.shape == (BATCH_SIZE,), "Incorrect batch label shape"

    # 3. Model Initialization
    print("\n--- Model Initialization ---")
    model = ResNet50Baseline(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # Validate Model Forward Pass
    dummy_input = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)
    assert dummy_output.shape == (
        2,
        NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (2, {NUM_CLASSES})"
    print("Model initialized and verified.")

    # 4. Training
    print("\n--- Training Loop ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        device=device,
        save_path=MODEL_SAVE_PATH,
    )

    # Validate Model Saving
    assert os.path.exists(MODEL_SAVE_PATH), "Best model file was not saved."
    print("Training complete and model saved.")

    # 5. Prediction
    print("\n--- Prediction ---")
    ids, probs = predict(trained_model, test_loader, device)

    # Validate Predictions
    assert len(ids) == len(
        test_df_sub
    ), "Number of predicted IDs does not match test set size"
    assert probs.shape == (
        len(test_df_sub),
        NUM_CLASSES,
    ), "Probability matrix shape mismatch"
    # Check if probabilities sum to roughly 1 (tolerance for float precision)
    sums = np.sum(probs, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"
    print("Prediction complete.")

    # 6. Submission
    print("\n--- Saving Submission ---")
    save_submission(ids, probs, class_names, output_path=SUBMISSION_PATH)

    # Validate Submission File
    assert os.path.exists(SUBMISSION_PATH), "Submission file not found"

    # Read back to check format
    sub_df = pd.read_csv(SUBMISSION_PATH)
    assert sub_df.shape == (
        len(test_df_sub),
        NUM_CLASSES + 1,
    ), "Submission CSV has incorrect shape"
    assert "id" in sub_df.columns, "Submission CSV missing 'id' column"
    assert sub_df.columns[1] == class_names[0], "First breed column mismatch"

    print(f"Submission saved successfully to {SUBMISSION_PATH}")
    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
