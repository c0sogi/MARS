import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import set_seed, calculate_class_weights
from library.dataset import AppleDataset, get_transforms
from library.model import get_model
from library.loss import WeightedSoftTargetCrossEntropy
from library.engine import fit


def main():
    print("==== Starting Demonstration Script ====")

    # 1. Setup and Configuration Overrides
    # We override specific Config attributes to ensure the demo runs quickly
    set_seed(Config.SEED)

    DEMO_EPOCHS = 1
    DEMO_BATCH_SIZE = 8
    DEMO_SUBSET_SIZE = 64  # Small subset for speed

    # Define paths for demo outputs
    DEMO_MODEL_DIR = os.path.join(Config.WORKING_DIR, "demo_models")
    os.makedirs(DEMO_MODEL_DIR, exist_ok=True)
    DEMO_MODEL_PATH = os.path.join(DEMO_MODEL_DIR, "resnet34_demo.pth")

    print(f"Device: {Config.DEVICE}")
    print(f"Demo Configuration: Epochs={DEMO_EPOCHS}, Batch Size={DEMO_BATCH_SIZE}")

    # 2. Data Preparation
    print("\n[Step 1] Loading and Preparing Data...")

    # Load metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    df_train_full = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val_full = pd.read_csv(Config.VAL_METADATA_PATH)

    # Subsample for demonstration speed
    df_train_demo = df_train_full.sample(
        n=min(len(df_train_full), DEMO_SUBSET_SIZE), random_state=Config.SEED
    ).reset_index(drop=True)
    df_val_demo = df_val_full.sample(
        n=min(len(df_val_full), DEMO_SUBSET_SIZE // 2), random_state=Config.SEED
    ).reset_index(drop=True)

    print(f"Training subset shape: {df_train_demo.shape}")
    print(f"Validation subset shape: {df_val_demo.shape}")

    # Calculate class weights on the demo subset
    # Note: In a real scenario, use the full dataset for weights
    class_weights = calculate_class_weights(df_train_demo, load_cached_data=False)
    print(f"Class weights: {class_weights}")

    # Instantiate Datasets
    train_dataset = AppleDataset(
        df_train_demo, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = AppleDataset(
        df_val_demo, transforms=get_transforms("val"), mode="val"
    )

    # Verify Dataset Output
    sample_img, sample_target = train_dataset[0]
    assert sample_img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected image shape (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {sample_img.shape}"
    assert sample_target.shape == (
        Config.N_CLASSES,
    ), f"Expected target shape ({Config.N_CLASSES},), got {sample_target.shape}"
    print("Dataset verification passed.")

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=DEMO_BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for simple script execution
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=DEMO_BATCH_SIZE, shuffle=False, num_workers=0
    )

    # 3. Model Initialization
    print("\n[Step 2] Initializing Model and Loss...")

    # Initialize Model
    # We use pretrained=False to avoid downloading weights during this time-constrained demo,
    # but in production Config.PRETRAINED (True) should be used.
    model = get_model(pretrained=False, n_classes=Config.N_CLASSES)
    model.to(Config.DEVICE)

    # Verify Model Output Shape
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(Config.DEVICE)
    with torch.no_grad():
        dummy_output = model(dummy_input)
    assert dummy_output.shape == (
        2,
        Config.N_CLASSES,
    ), f"Expected model output shape (2, {Config.N_CLASSES}), got {dummy_output.shape}"
    print("Model shape verification passed.")

    # Initialize Loss and Optimizer
    criterion = WeightedSoftTargetCrossEntropy(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # 4. Training Loop Execution
    print("\n[Step 3] Running Training Loop (Engine)...")

    # We use the provided fit function
    best_auc = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,  # Skipping scheduler for short demo
        device=Config.DEVICE,
        epochs=DEMO_EPOCHS,
        save_path=DEMO_MODEL_PATH,
    )

    print(f"Training complete. Best Validation AUC: {best_auc}")

    # Verify model was saved
    if os.path.exists(DEMO_MODEL_PATH):
        print(f"Model successfully saved to {DEMO_MODEL_PATH}")
    else:
        raise FileNotFoundError("Model file was not saved by the engine.")

    # 5. Inference Demonstration
    print("\n[Step 4] Running Inference on Test Data...")

    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    # Take a few samples
    df_test_demo = df_test.head(10)

    test_dataset = AppleDataset(
        df_test_demo, transforms=get_transforms("test"), mode="test"
    )
    test_loader = DataLoader(
        test_dataset, batch_size=DEMO_BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Load best model weights
    model.load_state_dict(torch.load(DEMO_MODEL_PATH, map_location=Config.DEVICE))
    model.eval()

    predictions = []
    image_ids = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(Config.DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            predictions.append(probs.cpu().numpy())
            image_ids.extend(ids)

    predictions = np.concatenate(predictions, axis=0)

    # Verify predictions
    assert len(predictions) == len(df_test_demo), "Mismatch in number of predictions"
    assert predictions.shape[1] == Config.N_CLASSES, "Mismatch in number of classes"

    # Create submission dataframe
    submission_df = pd.DataFrame(predictions, columns=Config.CLASSES)
    submission_df.insert(0, "image_id", image_ids)

    print("Sample Predictions:")
    print(submission_df.head())

    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    main()
