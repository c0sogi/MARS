import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil
from torch.utils.data import DataLoader, Subset

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import get_datasets
from library.models import CactusRepVGG, CactusResNet
from library.engine import train_one_epoch, validate
from library.stacking import run_stacking


def main():
    print("=== Starting Cactus Identification Demo ===\n")

    # 1. Setup Configuration for Demo
    # Override Config to use a demo directory and fast settings
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Initialize directories
    Config.setup_directories()

    # Set seed
    seed_everything(Config.SEED)
    logger = get_logger("Demo")
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Dataset Demonstration
    print("\n--- Demonstrating Dataset Loading ---")
    # We force load_cached_data=False to verify the raw loading logic works
    train_ds, val_ds, test_ds = get_datasets(load_cached_data=False)

    print(f"Train Dataset Size: {len(train_ds)}")
    print(f"Val Dataset Size:   {len(val_ds)}")
    print(f"Test Dataset Size:  {len(test_ds)}")

    # Verify a single sample
    img, label, aux = train_ds[0]
    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Label: {label}")
    print(f"Sample Aux Target: {aux}")

    assert img.shape == (3, 32, 32), "Image shape mismatch"
    assert isinstance(label, torch.Tensor), "Label should be a tensor"
    assert isinstance(aux, torch.Tensor), "Aux target should be a tensor"

    # Create subsets for fast training demo
    subset_indices = list(range(50))  # Use only 50 images
    train_subset = Subset(train_ds, subset_indices)
    val_subset = Subset(val_ds, subset_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=8,
        shuffle=True,
        num_workers=0,  # 0 for simple debugging/demo
        pin_memory=True if device == "cuda" else False,
    )
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, num_workers=0)

    # 3. Model Demonstration
    print("\n--- Demonstrating Models ---")

    # Test RepVGG
    print("Initializing CactusRepVGG...")
    model_repvgg = CactusRepVGG(num_classes=1).to(device)
    dummy_input = torch.randn(4, 3, 32, 32).to(device)

    # Forward pass
    cls_out, aux_out = model_repvgg(dummy_input)
    print(f"RepVGG Output Shapes - Class: {cls_out.shape}, Aux: {aux_out.shape}")

    assert cls_out.shape == (4, 1), "RepVGG class output shape incorrect"
    assert aux_out.shape == (4, 1), "RepVGG aux output shape incorrect"

    # Test Switch to Deploy
    print("Testing RepVGG switch_to_deploy...")
    model_repvgg.eval()
    model_repvgg.switch_to_deploy()
    cls_deploy, _ = model_repvgg(dummy_input)
    assert cls_deploy.shape == (4, 1), "RepVGG deploy mode output shape incorrect"
    print("RepVGG switch_to_deploy successful.")

    # Test ResNet
    print("Initializing CactusResNet...")
    model_resnet = CactusResNet(num_classes=1).to(device)
    cls_out, aux_out = model_resnet(dummy_input)
    print(f"ResNet Output Shapes - Class: {cls_out.shape}, Aux: {aux_out.shape}")

    assert cls_out.shape == (4, 1), "ResNet class output shape incorrect"

    # 4. Engine Demonstration (Training Loop)
    print("\n--- Demonstrating Training Engine ---")

    # Use ResNet for training demo
    model = model_resnet
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Running Train One Epoch...")
    loss = train_one_epoch(model, train_loader, optimizer, device, epoch=1)
    print(f"Epoch 1 Loss: {loss:.4f}")
    assert loss > 0, "Training loss should be positive"

    print("Running Validation...")
    metrics, preds, aux_preds = validate(model, val_loader, device)
    print(f"Validation Metrics: {metrics}")

    assert "AUC" in metrics, "AUC missing from metrics"
    assert len(preds) == 50, "Prediction count mismatch"
    assert len(aux_preds) == 50, "Aux prediction count mismatch"

    # 5. Stacking Demonstration
    print("\n--- Demonstrating Stacking Ensemble ---")

    # Generate synthetic data for stacking
    # Assume 2 models, 100 samples for train, 50 samples for test
    n_train = 100
    n_test = 50

    # Synthetic Training Data (OOF)
    train_class_preds = {
        "RepVGG": np.random.rand(n_train),
        "ResNet": np.random.rand(n_train),
    }
    train_aux_preds = {
        "RepVGG": np.random.rand(n_train),
        "ResNet": np.random.rand(n_train),
    }
    train_aux_targets = np.random.rand(n_train)
    train_labels = np.random.randint(0, 2, n_train)

    # Synthetic Test Data
    test_class_preds = {
        "RepVGG": np.random.rand(n_test),
        "ResNet": np.random.rand(n_test),
    }
    test_aux_preds = {
        "RepVGG": np.random.rand(n_test),
        "ResNet": np.random.rand(n_test),
    }
    test_aux_targets = np.random.rand(n_test)
    test_ids = [f"test_{i}.jpg" for i in range(n_test)]

    # Run Stacking
    final_preds = run_stacking(
        train_class_preds,
        train_aux_preds,
        train_aux_targets,
        train_labels,
        test_class_preds,
        test_aux_preds,
        test_aux_targets,
        test_ids,
        submission_path=Config.SUBMISSION_PATH,
    )

    print(f"Final Predictions Shape: {final_preds.shape}")
    assert len(final_preds) == n_test, "Final predictions length mismatch"

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission file content sample:")
    print(df_sub.head())

    assert list(df_sub.columns) == ["id", "has_cactus"], "Submission columns incorrect"
    assert len(df_sub) == n_test, "Submission row count incorrect"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
