import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Import provided library components
from library.config import Config, seed_everything
from library.dataset import get_dataloaders
from library.model import HierarchicalEfficientNet
from library.train import train_one_epoch, validate
from library.predict import predict
from library.utils import get_device


def main():
    print("==== Plant Species Classification Demo ====")

    # 1. Setup and Configuration
    # We override Config parameters to ensure the demo runs quickly and uses minimal resources.
    print("[1] Configuring environment...")
    seed_everything(42)

    # Override Config for speed and resource constraints
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 50  # Use a tiny subset of data
    Config.BATCH_SIZE = 8  # Small batch size
    Config.IMG_SIZE = 128  # Reduced image size for speed
    Config.NUM_EPOCHS = 1  # Single epoch
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo
    Config.PRETRAINED = False  # Disable downloading weights

    # Define paths for demo outputs
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    device = get_device()
    print(f"    Device: {device}")

    # 2. Data Loading
    print("\n[2] Loading Data...")
    # get_dataloaders handles metadata reading, taxonomy mapping, and loader creation
    train_loader, val_loader, test_loader, num_families = get_dataloaders(debug=True)

    print(f"    Num Families: {num_families}")
    print(f"    Train Batches: {len(train_loader)}")

    # Verify Data Integrity
    try:
        images, species_targets, family_targets = next(iter(train_loader))
        print(f"    Sample Batch Shape: {images.shape}")

        assert images.shape == (
            Config.BATCH_SIZE,
            3,
            Config.IMG_SIZE,
            Config.IMG_SIZE,
        ), "Incorrect image shape"
        assert species_targets.shape == (
            Config.BATCH_SIZE,
        ), "Incorrect species target shape"
        assert family_targets.shape == (
            Config.BATCH_SIZE,
        ), "Incorrect family target shape"
        print("    Data shapes verified.")
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    # 3. Model Initialization
    print("\n[3] Initializing Model...")
    model = HierarchicalEfficientNet(
        num_families=num_families,
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED,
    )
    model = model.to(device)

    # Verify Forward Pass
    dummy_input = images.to(device)
    with torch.no_grad():
        species_logits, family_logits = model(dummy_input)

    print(f"    Species Logits: {species_logits.shape}")
    print(f"    Family Logits: {family_logits.shape}")

    assert species_logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Species output dimension mismatch"
    assert family_logits.shape == (
        Config.BATCH_SIZE,
        num_families,
    ), "Family output dimension mismatch"
    print("    Model forward pass verified.")

    # 4. Training Step
    print("\n[4] Running Training Step...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Run a single epoch (on the tiny debug dataset)
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, scheduler, device, epoch=0
    )

    print(f"    Training Loss: {train_loss:.4f}")
    assert isinstance(train_loss, float), "Train loss must be a float"
    assert train_loss > 0, "Train loss must be positive"

    # 5. Validation Step
    print("\n[5] Running Validation Step...")
    val_loss, val_f1 = validate(model, val_loader, criterion, device)

    print(f"    Val Loss: {val_loss:.4f}")
    print(f"    Val F1 Score: {val_f1:.4f}")

    assert isinstance(val_loss, float), "Val loss must be a float"
    assert 0 <= val_f1 <= 1.0, "F1 score must be between 0 and 1"

    # 6. Saving Model
    print("\n[6] Saving Model Checkpoint...")
    torch.save(model.state_dict(), Config.MODEL_PATH)
    assert os.path.exists(Config.MODEL_PATH), "Model file not found after saving"
    print(f"    Model saved to {Config.MODEL_PATH}")

    # 7. Prediction / Inference
    print("\n[7] Running Inference...")
    # predict() loads the model from Config.MODEL_PATH and saves to Config.SUBMISSION_PATH
    # We pass debug=True to ensure it uses the small test set
    predict(
        model_path=Config.MODEL_PATH, output_path=Config.SUBMISSION_PATH, debug=True
    )

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"
    print(f"    Submission saved to {Config.SUBMISSION_PATH}")

    # Verify Submission Format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission Head:\n{sub_df.head(3)}")

    assert list(sub_df.columns) == ["Id", "Predicted"], "Submission columns mismatch"
    assert len(sub_df) > 0, "Submission file is empty"
    assert (
        sub_df["Id"].dtype == "int64" or sub_df["Id"].dtype == "int32"
    ), "Id column should be integer"
    assert (
        sub_df["Predicted"].dtype == "int64" or sub_df["Predicted"].dtype == "int32"
    ), "Predicted column should be integer"

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    main()
