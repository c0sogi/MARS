import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import set_seed
from library.dataset import load_data, BirdDataset, InferenceDataset
from library.model import BirdResNet
from library.engine import train_model, generate_submission


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Configuration and Setup
    # Initialize directories and set device
    Config.initialize()

    # Override Config for rapid demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 20  # Use only 20 samples for speed
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution
    Config.PRETRAINED = False  # Skip downloading weights for this demo

    # Set reproducibility
    set_seed(Config.SEED)
    print("Configuration initialized and overrides applied.")

    # 2. Data Loading
    print("\n--- Loading Data ---")
    # Load Train Data
    # load_data handles caching and debug slicing internally
    train_imgs, train_lbls, train_ids = load_data(Config.TRAIN_METADATA_PATH, "train")

    # Verify Train Data
    assert (
        len(train_imgs) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} training samples, got {len(train_imgs)}"
    assert train_imgs.ndim == 3, "Training images should be (N, H, W)"
    assert train_lbls.shape == (
        Config.DEBUG_SAMPLES,
        Config.NUM_CLASSES,
    ), "Label shape mismatch"

    # Load Validation Data
    val_imgs, val_lbls, val_ids = load_data(Config.VAL_METADATA_PATH, "val")

    # Load Test Data
    test_imgs, test_lbls, test_ids = load_data(Config.TEST_METADATA_PATH, "test")

    print(
        f"Data Loaded: Train({len(train_imgs)}), Val({len(val_imgs)}), Test({len(test_imgs)})"
    )

    # 3. Dataset and DataLoader Creation
    print("\n--- Creating Datasets and Loaders ---")
    train_dataset = BirdDataset(train_imgs, train_lbls)
    val_dataset = InferenceDataset(val_imgs, val_ids, val_lbls)
    test_dataset = InferenceDataset(test_imgs, test_ids)

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Verify Batch Shapes
    sample_inputs, sample_labels = next(iter(train_loader))
    # Expected Input: (Batch, 3, Height, Crop_Width) -> (4, 3, 224, 512)
    expected_input_shape = (Config.BATCH_SIZE, 3, Config.IMG_HEIGHT, Config.CROP_WIDTH)
    assert (
        sample_inputs.shape == expected_input_shape
    ), f"Batch input shape mismatch. Expected {expected_input_shape}, got {sample_inputs.shape}"
    assert sample_labels.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Batch label shape mismatch"
    print("DataLoader batch shapes verified.")

    # 4. Model Initialization
    print("\n--- Initializing Model ---")
    device = Config.DEVICE
    model = BirdResNet(pretrained=False)  # Use random weights for demo speed
    model = model.to(device)

    # Verify Forward Pass
    with torch.no_grad():
        # Move sample batch to device
        logits = model(sample_inputs.to(device))

    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"
    print("Model initialized and forward pass verified.")

    # 5. Training Loop
    print("\n--- Running Training Loop (1 Epoch) ---")
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Train the model using the engine
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        num_epochs=Config.EPOCHS,
        patience=1,
    )
    print("Training loop completed successfully.")

    # 6. Submission Generation
    print("\n--- Generating Submission ---")
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    generate_submission(trained_model, test_loader, device, submission_path)

    # Verify Submission File
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission loaded. Rows: {len(df_sub)}")

    # Validate submission format
    # Total rows should be Num_Test_Samples * Num_Classes
    expected_rows = len(test_imgs) * Config.NUM_CLASSES
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"
    assert (
        "Id" in df_sub.columns and "Probability" in df_sub.columns
    ), "Submission columns missing"

    # Check probability range
    probs = df_sub["Probability"].values
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of range [0, 1]"

    print("Submission format verified.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
