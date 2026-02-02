import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_multilabel_auc, average_checkpoints
from library.dataset import prepare_data, BirdDataset, get_transforms, load_test_data
from library.model import BirdClassifier
from library.trainer import run_fold, Trainer


def run_demo():
    # 1. Setup and Configuration Override for Speed
    print(">>> Setting up configuration for fast demonstration...")
    seed_everything(Config.SEED)

    # Override Config for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True  # Limits data to a few batches
    Config.TOP_K_CHECKPOINTS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directories exist (Config.setup() does this, but good to ensure)
    Config.setup()

    # 2. Data Preparation
    print("\n>>> Testing Data Preparation...")
    # Force reload to verify logic, though usually we'd use cached
    # We pass load_cached_data=False to test the fold creation logic
    df = prepare_data(load_cached_data=False)

    # Validation
    assert isinstance(df, pd.DataFrame), "prepare_data should return a DataFrame"
    assert "fold" in df.columns, "Dataframe must contain 'fold' column"
    assert len(df) > 0, "Dataframe should not be empty"
    print(f"Data prepared. Shape: {df.shape}")

    # 3. Dataset and Transforms
    print("\n>>> Testing BirdDataset and Transforms...")
    # Create a dataset instance
    train_ds = BirdDataset(
        df.head(10), transforms=get_transforms("train"), mode="train"
    )

    # Fetch one sample
    img, label = train_ds[0]

    # Validation
    assert isinstance(img, torch.Tensor), "Image should be a tensor"
    assert img.shape == (
        3,
        224,
        224,
    ), f"Expected image shape (3, 224, 224), got {img.shape}"
    assert isinstance(label, torch.Tensor), "Label should be a tensor"
    assert label.shape == (
        Config.NUM_CLASSES,
    ), f"Expected label shape ({Config.NUM_CLASSES},), got {label.shape}"
    print("Dataset verification passed.")

    # 4. Model Initialization
    print("\n>>> Testing BirdClassifier Model...")
    backbone = "resnet18"
    # We use pretrained=False here just to avoid downloading weights during this quick check
    # Note: run_fold uses pretrained=True hardcoded, so we test that integration later.
    model = BirdClassifier(backbone, Config.NUM_CLASSES, pretrained=False)

    # Test Forward Pass
    dummy_input = torch.randn(2, 3, 224, 224)  # Batch size 2
    output = model(dummy_input)

    # Validation
    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Expected output shape (2, {Config.NUM_CLASSES}), got {output.shape}"
    print("Model forward pass verification passed.")

    # 5. Training Loop (Integration Test)
    print("\n>>> Testing Training Loop (run_fold) in DEBUG mode...")
    # This runs training for 1 epoch on a tiny subset of data
    # It saves checkpoints to Config.CHECKPOINT_DIR
    fold_idx = 0
    best_model_path = run_fold(fold_idx, df, backbone)

    # Validation
    assert os.path.exists(best_model_path), f"Expected saved model at {best_model_path}"
    print(f"Training integration test passed. Model saved at {best_model_path}")

    # 6. Utility Functions Verification
    print("\n>>> Testing Utility Functions...")

    # Test AUC Calculation
    # Scenario: 3 classes, 2 samples
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_pred = np.array([[0.9, 0.1, 0.8], [0.1, 0.8, 0.2]])
    auc = calculate_multilabel_auc(y_true, y_pred)
    assert 0.0 <= auc <= 1.0, "AUC should be between 0 and 1"
    print(f"AUC Calculation verified: {auc:.4f}")

    # Test Checkpoint Averaging
    # Create dummy checkpoints
    ckpt_path_1 = os.path.join(Config.CHECKPOINT_DIR, "dummy_1.pth")
    ckpt_path_2 = os.path.join(Config.CHECKPOINT_DIR, "dummy_2.pth")

    # Save state dicts
    torch.save(model.state_dict(), ckpt_path_1)
    torch.save(model.state_dict(), ckpt_path_2)

    # Average them
    avg_state_dict = average_checkpoints([ckpt_path_1, ckpt_path_2])

    # Validation
    assert isinstance(avg_state_dict, dict), "Averaged checkpoint should be a dict"
    # Check a key
    first_key = list(avg_state_dict.keys())[0]
    assert torch.is_tensor(avg_state_dict[first_key]), "Weights should be tensors"
    print("Checkpoint averaging verified.")

    # 7. Inference Mock
    print("\n>>> Testing Inference Setup...")
    test_df = load_test_data()
    test_ds = BirdDataset(
        test_df.head(5), transforms=get_transforms("test"), mode="test"
    )
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=2, shuffle=False)

    model.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            preds = model(images)
            assert preds.shape[1] == Config.NUM_CLASSES
            break
    print("Inference setup verified.")

    print("\n>>> All demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
