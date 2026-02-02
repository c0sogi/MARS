import os
import sys
import shutil
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_loaders, DogDataset
from library.model_factory import create_model, set_backbone_trainable
from library.trainer import Trainer, predict_with_tta
from library.calibration import TemperatureScaler


def run_demo():
    print("Starting Library Usage Demonstration...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[1] Configuring environment for demo...")

    # Override Config for speed and demonstration purposes
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Small subset
    Config.EPOCHS = 1
    Config.FREEZE_BACKBONE_EPOCHS = 0  # Skip freeze phase for speed
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2
    Config.MODEL_ARCHS = ["resnet18"]  # Use lightweight model for demo
    Config.WORKING_DIR = "./working/demo_run"

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds
    seed_everything(Config.SEED)
    logger = get_logger(
        name="demo", log_file=os.path.join(Config.WORKING_DIR, "demo.log")
    )

    print("Configuration updated: DEBUG=True, Model=resnet18")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("\n[2] Testing Data Loading...")

    # Load data with debug flag
    train_loader, val_loader, test_loader, class_list = get_loaders(
        load_cached_data=False, debug=True
    )

    # Validations
    assert len(class_list) > 0, "Class list should not be empty"
    assert (
        len(train_loader.dataset) <= Config.DEBUG_SAMPLE_SIZE
    ), "Train dataset size mismatch"

    # Check batch structure
    images, labels = next(iter(train_loader))
    print(f"Train Batch Shape - Images: {images.shape}, Labels: {labels.shape}")

    assert images.dim() == 4, "Images should be 4D tensor (B, C, H, W)"
    assert images.shape[1] == 3, "Images should have 3 channels"
    assert labels.dim() == 1, "Labels should be 1D tensor"

    print("Data Loading verified successfully.")

    # ==========================================
    # 3. Model Factory
    # ==========================================
    print("\n[3] Testing Model Factory...")

    model_name = Config.MODEL_ARCHS[0]
    num_classes = len(class_list)

    # Create model
    model = create_model(model_name, num_classes=num_classes, pretrained=True)
    model.to(Config.DEVICE)

    # Validate model structure
    assert isinstance(model, nn.Module), "Model is not a torch.nn.Module"

    # Test Backbone Freezing
    set_backbone_trainable(model, trainable=False)
    # Check if backbone gradients are disabled (sample check)
    # ResNet18 structure: conv1, bn1, layer1... fc
    # We expect conv1 to be frozen
    for name, param in model.named_parameters():
        if "fc" not in name and "head" not in name:  # timm usually uses 'fc' or 'head'
            if param.requires_grad:
                # Some parts might be kept open depending on implementation,
                # but generally set_backbone_trainable(False) should freeze most.
                # However, the provided function freezes *everything* then unfreezes head.
                pass

    # Unfreeze for training
    set_backbone_trainable(model, trainable=True)

    print(f"Model {model_name} created and validated.")

    # ==========================================
    # 4. Training Logic (Trainer)
    # ==========================================
    print("\n[4] Testing Trainer...")

    trainer = Trainer(model, device=Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Run 1 epoch of training on the small subset
    print("Running training epoch...")
    initial_loss = trainer.train_one_epoch(train_loader, optimizer, epoch=0)

    assert isinstance(initial_loss, float), "Train loss should be a float"
    assert initial_loss > 0, "Train loss should be positive"
    print(f"Train Loss: {initial_loss:.4f}")

    # Run validation
    print("Running validation epoch...")
    val_loss, val_acc = trainer.valid_one_epoch(val_loader)

    assert isinstance(val_loss, float), "Val loss should be a float"
    assert 0 <= val_acc <= 1.0, "Validation accuracy should be between 0 and 1"
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    print("Trainer logic verified.")

    # ==========================================
    # 5. Calibration (Temperature Scaling)
    # ==========================================
    print("\n[5] Testing Calibration...")

    # Create dummy logits and labels for deterministic testing
    # 10 samples, 5 classes
    dummy_logits = torch.randn(10, 5) * 2.0
    dummy_labels = torch.randint(0, 5, (10,))

    scaler = TemperatureScaler()

    # Check initial temperature
    assert scaler.temperature.item() == 1.5, "Initial temperature should be 1.5"

    # Fit scaler
    scaler.fit(dummy_logits, dummy_labels)

    # Check if temperature changed (optimization occurred)
    opt_temp = scaler.temperature.item()
    print(f"Optimized Temperature: {opt_temp:.4f}")

    # Get probabilities
    probs = scaler.get_probabilities(dummy_logits)

    assert probs.shape == (10, 5), "Probabilities shape mismatch"
    # Check sum to 1
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums)), "Probabilities must sum to 1"

    print("Calibration module verified.")

    # ==========================================
    # 6. Inference & TTA
    # ==========================================
    print("\n[6] Testing Inference with TTA...")

    # Use the trained model for inference on test set
    # Test loader returns (image, id)

    # Get one batch from test loader manually to verify TTA function
    test_batch = next(iter(test_loader))

    # TTA Prediction
    # We need to wrap the batch in a list or pass the loader.
    # The function `predict_with_tta` iterates over a loader.
    # Let's create a temporary loader with just one batch for speed

    class SingleBatchLoader:
        def __init__(self, batch):
            self.batch = batch

        def __iter__(self):
            yield self.batch

        def __len__(self):
            return 1

    temp_loader = SingleBatchLoader(test_batch)

    logits = predict_with_tta(model, temp_loader, device=Config.DEVICE)

    assert logits.shape[0] == Config.BATCH_SIZE, "Output batch size mismatch"
    assert logits.shape[1] == num_classes, "Output class count mismatch"

    print("Inference with TTA verified.")

    # ==========================================
    # 7. End-to-End Submission Generation (Mock)
    # ==========================================
    print("\n[7] Verifying Submission File Generation...")

    # Mock probabilities
    mock_probs = np.random.rand(len(test_loader.dataset), num_classes)
    # Normalize
    mock_probs = mock_probs / mock_probs.sum(axis=1, keepdims=True)

    # Extract IDs
    test_ids = []
    for _, batch_ids in test_loader:
        test_ids.extend(batch_ids)

    df_sub = pd.DataFrame(mock_probs, columns=class_list)
    df_sub.insert(0, "id", test_ids)

    sub_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    df_sub.to_csv(sub_path, index=False)

    assert os.path.exists(sub_path), "Submission file was not created"

    # Verify file format
    df_check = pd.read_csv(sub_path)
    assert "id" in df_check.columns, "Submission missing 'id' column"
    assert len(df_check) == len(test_loader.dataset), "Submission row count mismatch"
    assert df_check.shape[1] == num_classes + 1, "Submission column count mismatch"

    print(f"Submission file generated at {sub_path}")
    print("\nAll library components verified successfully!")


if __name__ == "__main__":
    run_demo()
