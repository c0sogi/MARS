import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

# Import from the provided library
import library.utils as utils
import library.dataset as dataset_lib
import library.model as model_lib
import library.trainer as trainer_lib

# Define working directories for this demo
DEMO_WORK_DIR = "./working/demo_usage/"
DEMO_CACHE_DIR = os.path.join(DEMO_WORK_DIR, "cache")
DEMO_CHECKPOINT_DIR = os.path.join(DEMO_WORK_DIR, "checkpoints")
DEMO_SUBMISSION_DIR = os.path.join(DEMO_WORK_DIR, "submission")


def main():
    print("Initializing Demo...")

    # 1. Setup and Reproducibility
    # ----------------------------------------------------------------
    utils.set_seed(42)
    device = utils.get_device()
    print(f"Device: {device}")

    # Ensure clean directories
    for d in [DEMO_CACHE_DIR, DEMO_CHECKPOINT_DIR, DEMO_SUBMISSION_DIR]:
        os.makedirs(d, exist_ok=True)

    # Monkey-patch the cache directories in the library modules to avoid
    # interfering with the main training artifacts or relying on specific paths.
    dataset_lib.CACHE_DIR = DEMO_CACHE_DIR
    model_lib.CACHE_DIR = DEMO_CACHE_DIR
    model_lib.CHECKPOINT_DIR = DEMO_CHECKPOINT_DIR

    # 2. Data Loading and Preprocessing
    # ----------------------------------------------------------------
    print("\n[Step 1] Loading and Slicing Data...")
    # Load data (will process from scratch if cache doesn't exist in demo dir)
    # We force load_cached_data=False initially to ensure we demonstrate processing logic
    # or rely on the library to handle it if files are missing.
    data = dataset_lib.load_data(load_cached_data=True)

    # Create a small subset for speed (50 samples for train, 20 for val, 20 for test)
    subset_size_train = 50
    subset_size_val = 20
    subset_size_test = 20

    X_train_sub = data["X_train"][:subset_size_train]
    angle_train_sub = data["angle_train"][:subset_size_train]
    y_train_sub = data["y_train"][:subset_size_train]

    X_val_sub = data["X_val"][:subset_size_val]
    angle_val_sub = data["angle_val"][:subset_size_val]
    y_val_sub = data["y_val"][:subset_size_val]

    X_test_sub = data["X_test"][:subset_size_test]
    angle_test_sub = data["angle_test"][:subset_size_test]
    ids_test_sub = data["ids_test"][:subset_size_test]

    print(f"Train subset shape: {X_train_sub.shape}")
    print(f"Val subset shape: {X_val_sub.shape}")

    # verify shapes
    assert X_train_sub.shape == (subset_size_train, 3, 75, 75)
    assert angle_train_sub.shape == (subset_size_train,)
    assert y_train_sub.shape == (subset_size_train,)

    # 3. Dataset and DataLoader Instantiation
    # ----------------------------------------------------------------
    print("\n[Step 2] Creating Datasets and Loaders...")

    # Define transforms (simple flip)
    train_transform = T.Compose(
        [T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5)]
    )

    # Instantiate Datasets using the library class
    train_dataset = dataset_lib.IcebergDataset(
        X_train_sub, angle_train_sub, y_train_sub, transform=train_transform
    )
    val_dataset = dataset_lib.IcebergDataset(
        X_val_sub, angle_val_sub, y_val_sub, transform=None
    )
    test_dataset = dataset_lib.IcebergDataset(
        X_test_sub, angle_test_sub, y=None, transform=None
    )

    # Create Loaders
    batch_size = 10
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Verify a single batch
    images, angles, labels = next(iter(train_loader))
    assert images.shape == (batch_size, 3, 75, 75)
    assert angles.shape == (batch_size, 1)
    assert labels.shape == (batch_size, 1)
    print("DataLoader verification successful.")

    # 4. Model Instantiation and Forward Pass Check
    # ----------------------------------------------------------------
    print("\n[Step 3] Verifying Model Architecture...")
    model = model_lib.SPPCNN().to(device)

    # Dummy forward pass
    dummy_img = torch.randn(2, 3, 75, 75).to(device)
    dummy_ang = torch.tensor([[35.0], [40.0]]).to(device)

    with torch.no_grad():
        output = model(dummy_img, dummy_ang)

    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"
    print("Model forward pass successful.")

    # 5. Training Loop Demonstration
    # ----------------------------------------------------------------
    print("\n[Step 4] Running Training Loop (Fold 0)...")

    # We use trainer_lib.train_fold
    # Limiting epochs to 2 for speed
    fold_idx = 0
    best_loss = trainer_lib.train_fold(
        fold_idx=fold_idx,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        patience=2,
        lr=1e-3,
        weight_decay=1e-4,
        checkpoint_dir=DEMO_CHECKPOINT_DIR,
    )

    # Verify checkpoint creation
    checkpoint_path = os.path.join(DEMO_CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    print(f"Training complete. Best Val Loss: {best_loss:.4f}")
    print(f"Checkpoint saved at: {checkpoint_path}")

    # 6. Inference and Submission
    # ----------------------------------------------------------------
    print("\n[Step 5] Running Inference...")

    # Load best model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Predict using library function
    preds = trainer_lib.predict(model, test_loader, device)

    # Verify predictions
    assert len(preds) == subset_size_test
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions must be probabilities between 0 and 1"

    print(f"Generated {len(preds)} predictions.")
    print(f"Sample predictions: {preds[:5]}")

    # Create submission file
    submission_df = pd.DataFrame({"id": ids_test_sub, "is_iceberg": preds})

    # We also need to fill the rest of the submission to match sample_submission length
    # if we were doing a real submission, but for this demo, we just save the subset results
    # or load the sample_submission and merge.
    # To strictly follow the format of the provided sample_submission.csv (321 rows),
    # we will load the sample submission and update the rows we predicted.

    sample_sub_path = "./input/sample_submission.csv"
    full_submission = pd.read_csv(sample_sub_path)

    # Map predictions to the full dataframe
    # Create a dictionary for quick lookup
    pred_dict = dict(zip(ids_test_sub, preds))

    # Update values
    full_submission["is_iceberg"] = (
        full_submission["id"].map(pred_dict).fillna(full_submission["is_iceberg"])
    )

    output_path = os.path.join(DEMO_SUBMISSION_DIR, "submission.csv")
    full_submission.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print("Demo completed successfully.")


if __name__ == "__main__":
    main()
