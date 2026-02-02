import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, EarlyStopping
from library.dataset import WhaleDataset, get_dataloaders
from library.models import WhaleModel
from library.engine import train_one_epoch, validate
from library.ensemble import MetaLearner, save_submission


def create_mini_metadata(n_samples=50):
    """
    Creates smaller versions of the metadata CSVs to speed up the demo.
    """
    print("\n--- Creating Mini Metadata for Demo ---")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define paths for mini metadata
    mini_train_path = os.path.join(Config.WORKING_DIR, "train_mini.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "val_mini.csv")
    mini_test_path = os.path.join(Config.WORKING_DIR, "test_mini.csv")

    # Load original metadata
    train_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    # Sample and save
    train_df.head(n_samples).to_csv(mini_train_path, index=False)
    val_df.head(n_samples).to_csv(mini_val_path, index=False)
    test_df.head(n_samples).to_csv(mini_test_path, index=False)

    print(f"Created mini metadata with {n_samples} samples each.")
    return mini_train_path, mini_val_path, mini_test_path


def demo_dataset_and_loader():
    print("\n--- Testing Dataset and DataLoader ---")

    # 1. Test Dataset Instantiation (this will trigger caching logic)
    # We use the 'train' mode which applies augmentation
    ds = WhaleDataset(mode="train", load_cached_data=False)

    # Verify length
    print(f"Dataset length: {len(ds)}")
    assert len(ds) > 0, "Dataset should not be empty"

    # Verify item shape
    img, target = ds[0]
    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Target: {target}")

    # Expected shape: (1, N_MELS, Time) -> (1, 128, 125) given SR=2000, HOP=64, 2s duration
    assert img.shape[0] == 1, "Image should have 1 channel"
    assert img.shape[1] == Config.N_MELS, f"Image height should be {Config.N_MELS}"
    assert isinstance(target, torch.Tensor), "Target should be a tensor"

    # 2. Test DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8, num_workers=0)

    # Fetch one batch
    inputs, targets = next(iter(train_loader))
    print(f"Batch Input Shape: {inputs.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    assert inputs.shape[0] == 8, "Batch size should be 8"
    assert inputs.shape[1] == 1, "Channel dim should be 1"

    return train_loader, val_loader, test_loader


def demo_model_training(train_loader, val_loader):
    print("\n--- Testing Model and Training Loop ---")

    device = Config.DEVICE
    print(f"Using device: {device}")

    # Instantiate Model
    # Using resnet34 as it is lighter than efficientnet for demo
    model_name = "resnet34"
    model = WhaleModel(model_name, pretrained=True).to(device)

    # Verify Model Output Shape
    dummy_input = torch.randn(2, 1, Config.N_MELS, 125).to(device)
    dummy_output = model(dummy_input)
    print(f"Model Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (2, 1), "Output shape should be (Batch, 1)"

    # Setup Training Components
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # Run 1 Epoch of Training
    print("Running training epoch...")
    train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
    print(f"Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss should not be NaN"

    # Run Validation
    print("Running validation...")
    val_loss, val_auc = validate(model, val_loader, device, criterion)
    print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

    # Save model for ensemble demo
    model_path = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return model


def demo_ensemble_and_submission(model, val_loader, test_loader):
    print("\n--- Testing Ensemble and Submission ---")

    device = Config.DEVICE

    # 1. Generate Predictions for 'resnet34' (the model we just trained)
    # Validation (OOF) preds
    val_targets = []
    val_preds_m1 = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().ravel()

            val_preds_m1.extend(probs)
            val_targets.extend(targets.numpy().ravel())

    # Test preds
    test_clips = []
    test_preds_m1 = []

    with torch.no_grad():
        for inputs, clips in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().ravel()

            test_preds_m1.extend(probs)
            test_clips.extend(clips)

    # 2. Simulate a second model (e.g., EfficientNet) with random noise for demonstration
    # In a real scenario, we would load the second model and predict
    print("Simulating second model predictions for ensemble...")
    np.random.seed(Config.SEED)
    val_preds_m2 = np.random.uniform(0, 1, size=len(val_preds_m1))
    test_preds_m2 = np.random.uniform(0, 1, size=len(test_preds_m1))

    # 3. Train Meta-Learner
    meta_learner = MetaLearner()

    oof_df = pd.DataFrame({"resnet34": val_preds_m1, "efficientnet": val_preds_m2})

    # Fit meta-learner
    oof_auc = meta_learner.fit(oof_df, val_targets)
    assert oof_auc >= 0.0 and oof_auc <= 1.0, "AUC must be between 0 and 1"

    # 4. Predict on Test Set
    test_df_in = pd.DataFrame(
        {"resnet34": test_preds_m1, "efficientnet": test_preds_m2}
    )

    final_probs = meta_learner.predict(test_df_in)

    # 5. Save Submission
    save_submission(test_clips, final_probs, filename="demo_submission.csv")

    # Verify file creation
    expected_path = os.path.join("./submission", "demo_submission.csv")
    assert os.path.exists(
        expected_path
    ), "Submission file was not created in ./submission"

    # Verify content format
    sub_df = pd.read_csv(expected_path)
    assert list(sub_df.columns) == [
        "clip",
        "probability",
    ], "Submission columns incorrect"
    assert len(sub_df) == len(test_clips), "Submission row count mismatch"
    print("Submission verified.")


if __name__ == "__main__":
    # --- 1. Setup & Configuration Override ---
    seed_everything(42)

    # Modify Config for Demo execution
    Config.WORKING_DIR = "./working/demo_execution"
    Config.OUTPUT_DIR = Config.WORKING_DIR
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.EPOCHS = 1  # Only 1 epoch for speed
    Config.BATCH_SIZE = 8

    # Initialize directories
    Config.setup()

    # Create mini metadata to avoid processing 20GB of audio
    mini_train, mini_val, mini_test = create_mini_metadata(n_samples=20)

    # Point Config to mini metadata
    Config.TRAIN_CSV = mini_train
    Config.VAL_CSV = mini_val
    Config.TEST_CSV = mini_test

    # --- 2. Run Demos ---
    try:
        # Dataset & Loader
        train_loader, val_loader, test_loader = demo_dataset_and_loader()

        # Model Training
        trained_model = demo_model_training(train_loader, val_loader)

        # Ensemble & Submission
        demo_ensemble_and_submission(trained_model, val_loader, test_loader)

        print("\n=== Demo Completed Successfully ===")

    except AssertionError as e:
        print(f"\n!!! Demo Failed Assertion: {e} !!!")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! Demo Failed with Error: {e} !!!")
        import traceback

        traceback.print_exc()
        sys.exit(1)
