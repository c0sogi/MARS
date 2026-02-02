import os
import sys
import torch
import pandas as pd
import warnings

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import HotelClassifier
from library.engine import train_model
from library.inference import predict_and_submit


def main():
    # 1. Setup
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set random seeds for reproducibility
    print("--- Setting up environment ---")
    seed_everything(Config.SEED)

    # Get computation device
    device = get_device()
    print(f"Device selected: {device}")

    # 2. Data Loading
    # We use debug=True to load a small subset of data for quick demonstration
    print("\n--- Initializing Data Loaders (Debug Mode) ---")
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Validate Data Loaders
    try:
        images, labels = next(iter(train_loader))
        print(f"Train batch shape: Images {images.shape}, Labels {labels.shape}")

        # Assert batch dimensions match Config
        assert images.shape == (
            Config.BATCH_SIZE,
            3,
            Config.IMAGE_SIZE,
            Config.IMAGE_SIZE,
        ), f"Expected image shape {(Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)}, got {images.shape}"
        assert labels.shape == (
            Config.BATCH_SIZE,
        ), f"Expected label shape {(Config.BATCH_SIZE,)}, got {labels.shape}"

        print("Data loader verification passed.")
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    # 3. Model Initialization
    print("\n--- Initializing Model ---")
    # Initialize model with the specific number of classes found in the dataset
    model = HotelClassifier(n_classes=len(classes))
    model.to(device)

    # Validate Model Architecture
    # Create a dummy input to check forward pass dimensions
    dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model output shape: {output.shape}")
    assert output.shape == (
        2,
        len(classes),
    ), f"Expected output shape (2, {len(classes)}), got {output.shape}"
    print("Model architecture verification passed.")

    # 4. Training Loop
    print("\n--- Starting Training Demo ---")
    # Define Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Train for 1 epoch to demonstrate the pipeline quickly
    # train_model handles training, validation, and saving the best model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=1,
        patience=1,
    )

    # Verify model file creation
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model file not found at {Config.MODEL_SAVE_PATH}"
    print("Training completed and model saved.")

    # 5. Inference and Submission
    print("\n--- Generating Submission ---")
    # predict_and_submit wraps the inference engine logic
    predict_and_submit(trained_model, test_loader, classes, device)

    # 6. Validate Submission File
    sub_path = Config.SUBMISSION_PATH
    assert os.path.exists(sub_path), f"Submission file not found at {sub_path}"

    df_sub = pd.read_csv(sub_path)
    print(f"Submission file loaded. Rows: {len(df_sub)}")

    # Check columns
    assert (
        "image" in df_sub.columns and "hotel_id" in df_sub.columns
    ), "Submission file missing required columns."

    # Check format of the first prediction
    if len(df_sub) > 0:
        first_pred = df_sub.iloc[0]["hotel_id"]
        assert isinstance(first_pred, str), "Prediction must be a string."
        pred_list = first_pred.split()
        assert (
            len(pred_list) == 5
        ), f"Expected 5 hotel IDs per image, found {len(pred_list)}"
        print(f"Sample prediction format verified: {first_pred}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
