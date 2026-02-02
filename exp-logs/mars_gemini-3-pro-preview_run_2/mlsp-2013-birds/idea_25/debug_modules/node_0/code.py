import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import set_seed, calculate_multilabel_auc
from library.dataset import (
    BirdDataset,
    get_train_transforms,
    get_valid_transforms,
    Mixup,
)
from library.models import BirdClassifier
from library.sam import SAM
from library.engine import (
    train_one_epoch,
    evaluate,
    predict_cyclic_tta,
    get_weighted_loss,
)


def run_demo():
    print("Starting Library Usage Demonstration...")

    # ------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed/Demo
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment...")
    # Enable debug mode to use a small subset of data (50 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 20
    Config.BATCH_SIZE = 4
    # Use a single lightweight backbone for demonstration
    Config.BACKBONES = ["resnet18"]
    # Disable pretrained weights to avoid downloading/network issues during demo
    # We only want to verify architecture and logic.
    USE_PRETRAINED = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------------
    # 2. Verify Utils
    # ------------------------------------------------------------------------
    print("\n[2] Verifying Utils...")
    set_seed(42)

    # Test AUC calculation
    # Case: Perfect prediction
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_pred = np.array([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1], [0.8, 0.8, 0.2]])
    auc = calculate_multilabel_auc(y_true, y_pred)
    print(f"Calculated AUC (Synthetic): {auc:.4f}")
    assert 0.0 <= auc <= 1.0, "AUC should be between 0 and 1"

    # Case: Empty class (column with only 0s) - should be skipped
    y_true_edge = np.array([[1, 0], [0, 0], [1, 0]])  # 2nd col is all 0
    y_pred_edge = np.array([[0.9, 0.1], [0.1, 0.1], [0.8, 0.1]])
    auc_edge = calculate_multilabel_auc(y_true_edge, y_pred_edge)
    # Only first column is valid. Perfect pred for first col -> AUC 1.0
    assert np.isclose(
        auc_edge, 1.0
    ), "AUC calculation failed to handle constant class columns"
    print("Utils verification passed.")

    # ------------------------------------------------------------------------
    # 3. Verify Dataset and Augmentations
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Dataset...")
    # Force reload from CSV to avoid using old cache if exists
    train_ds = BirdDataset(
        csv_path=Config.TRAIN_CSV,
        mode="train",
        load_cached_data=False,
        transforms=get_train_transforms(),
    )

    assert len(train_ds) > 0, "Dataset should not be empty"
    print(f"Dataset size (Debug Mode): {len(train_ds)}")

    # Fetch one sample
    img, lbl, rid = train_ds[0]
    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Label Shape: {lbl.shape}")

    # Validate Shapes
    # Expected: (3, 224, 448)
    assert img.shape == (
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    ), f"Incorrect image dimensions: {img.shape}"
    assert lbl.shape == (
        Config.NUM_CLASSES,
    ), f"Incorrect label dimensions: {lbl.shape}"
    assert isinstance(img, torch.Tensor), "Image should be a tensor"

    # Verify Mixup Logic
    print("Verifying Mixup...")
    dataloader = DataLoader(train_ds, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    mixup_fn = Mixup(alpha=1.0)
    mixed_imgs, lbl_a, lbl_b, lam = mixup_fn(batch)

    assert mixed_imgs.shape == batch[0].shape, "Mixed images shape mismatch"
    assert lbl_a.shape == batch[1].shape, "Label A shape mismatch"
    assert 0.0 <= lam <= 1.0, "Lambda should be between 0 and 1"
    print("Dataset verification passed.")

    # ------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # ------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")
    model = BirdClassifier(backbone_name="resnet18", pretrained=USE_PRETRAINED)
    model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(2, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, Config.NUM_CLASSES), "Model output shape incorrect"
    print("Model verification passed.")

    # ------------------------------------------------------------------------
    # 5. Verify SAM Optimizer Logic
    # ------------------------------------------------------------------------
    print("\n[5] Verifying SAM Optimizer...")
    # Use a simple linear model for optimizer check to be fast
    simple_model = torch.nn.Linear(10, 1).to(device)
    optimizer = SAM(simple_model.parameters(), torch.optim.SGD, lr=0.1, rho=0.05)

    # Dummy data
    x = torch.randn(4, 10).to(device)
    y = torch.randn(4, 1).to(device)

    # Step 1
    pred = simple_model(x)
    loss = torch.nn.MSELoss()(pred, y)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    # Step 2
    pred_adv = simple_model(x)
    loss_adv = torch.nn.MSELoss()(pred_adv, y)
    loss_adv.backward()
    optimizer.second_step(zero_grad=True)

    print("SAM Optimizer step executed successfully.")

    # ------------------------------------------------------------------------
    # 6. Verify Engine (Train/Eval/Predict)
    # ------------------------------------------------------------------------
    print("\n[6] Verifying Engine Functions...")

    # Re-init model for training check
    model = BirdClassifier(backbone_name="resnet18", pretrained=USE_PRETRAINED)
    model.to(device)

    # Optimizer
    optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=1e-3)

    # Criterion
    # Note: get_weighted_loss reads the CSV file to calculate weights
    criterion = get_weighted_loss(device)

    # A. Train One Epoch
    print("Running train_one_epoch...")
    epoch_loss = train_one_epoch(
        model, dataloader, optimizer, criterion, device, epoch=1
    )
    print(f"Train Loss: {epoch_loss:.4f}")
    assert not np.isnan(epoch_loss), "Training loss is NaN"

    # B. Evaluate
    print("Running evaluate...")
    # Use same ds for val just for demo
    val_loss, val_auc = evaluate(model, dataloader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"

    # C. Predict Cyclic TTA
    print("Running predict_cyclic_tta...")
    ids, preds = predict_cyclic_tta(model, dataloader, device)

    print(f"Predictions Shape: {preds.shape}")
    print(f"IDs Shape: {ids.shape}")

    assert preds.shape[0] == len(
        train_ds
    ), "Number of predictions does not match dataset size"
    assert preds.shape[1] == Config.NUM_CLASSES, "Number of prediction classes mismatch"
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Predictions must be probabilities [0, 1]"

    print("Engine verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
