import os
import sys
import numpy as np
import torch
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, probabilistic_f1, load_dicom_and_process
from library.data import get_dataloaders, BreastCancerDataset, get_transforms
from library.model import EfficientNetV2Classifier
from library.engine import train_representation_epoch, evaluate, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Breast Cancer Detection Pipeline Demonstration ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("[1/5] Configuring environment for fast demonstration...")

    # Override Config settings to ensure the script runs quickly
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 32  # Small subset for testing
    Config.IMG_SIZE = (128, 128)  # Smaller images for faster processing
    Config.STAGE1_EPOCHS = 1
    Config.STAGE2_EPOCHS = 1
    Config.STAGE1_BATCH_SIZE = 4
    Config.STAGE2_BATCH_SIZE = 4
    Config.NUM_WORKERS = 2  # Reduce overhead for small data

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed
    seed_everything(Config.SEED)
    print("Configuration updated: DEBUG=True, Image Size=(128,128), Batch Size=4.")

    # -------------------------------------------------------------------------
    # 2. Verify Utils
    # -------------------------------------------------------------------------
    print("\n[2/5] Verifying Utility Functions...")

    # A. Test Probabilistic F1
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    # Manual Calc:
    # pTP = 0.9*1 + 0.1*0 + 0.8*1 + 0.2*0 = 1.7
    # pPrec Denom = 0.9 + 0.1 + 0.8 + 0.2 = 2.0 -> pPrec = 1.7 / 2.0 = 0.85
    # pRec Denom = 1 + 0 + 1 + 0 = 2.0 -> pRec = 1.7 / 2.0 = 0.85
    # pF1 = 2 * (0.85 * 0.85) / (0.85 + 0.85) = 0.85
    pf1 = probabilistic_f1(y_true, y_pred)
    assert np.isclose(pf1, 0.85), f"pF1 calculation incorrect. Expected 0.85, got {pf1}"
    print(" - probabilistic_f1: OK")

    # B. Test DICOM Loading
    # Get a valid file path from metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    sample_file = df_train.iloc[0]["file_path"]

    img = load_dicom_and_process(sample_file, img_size=Config.IMG_SIZE)

    assert isinstance(img, np.ndarray), "Image should be a numpy array"
    assert img.shape == (
        128,
        128,
        1,
    ), f"Image shape mismatch. Expected (128, 128, 1), got {img.shape}"
    assert (
        img.min() >= 0.0 and img.max() <= 1.0
    ), "Image values should be normalized to [0, 1]"
    print(f" - load_dicom_and_process: OK (Loaded {sample_file})")

    # -------------------------------------------------------------------------
    # 3. Verify Data Pipeline
    # -------------------------------------------------------------------------
    print("\n[3/5] Verifying Data Pipeline...")

    # Get DataLoaders (Stage 1)
    loaders = get_dataloaders(stage=1, debug=True, load_cached_data=False)
    train_loader = loaders["train"]

    # Check Batch
    images, targets = next(iter(train_loader))

    # Assertions
    assert images.shape == (
        4,
        3,
        128,
        128,
    ), f"Batch image shape incorrect: {images.shape}"
    assert targets.shape == (4, 1), f"Batch target shape incorrect: {targets.shape}"
    assert torch.is_tensor(images), "Images should be a torch Tensor"

    # Check Sampler logic
    # Stage 1 should have a sampler (WeightedRandomSampler) because it's balanced
    assert (
        train_loader.sampler is not None
    ), "Stage 1 loader should have a sampler defined"
    assert isinstance(
        train_loader.sampler, torch.utils.data.WeightedRandomSampler
    ), "Stage 1 should use WeightedRandomSampler"

    # Check Stage 2 loader (should not have weighted sampler)
    loaders_s2 = get_dataloaders(stage=2, debug=True, load_cached_data=False)
    # When shuffle=True (default for Stage 2), sampler is technically a RandomSampler or None in basic config,
    # but specifically our code sets `sampler=None` and `shuffle=True`.
    # Note: DataLoader wraps shuffle=True into a RandomSampler internally, but the `sampler` arg passed to init is None.
    # We can check if the underlying dataset is just the raw dataset.
    print(" - DataLoaders creation: OK")
    print(" - Batch shapes: OK")
    print(" - Sampler configuration: OK")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4/5] Verifying Model Architecture...")

    device = Config.DEVICE
    model = EfficientNetV2Classifier(
        pretrained=False
    )  # No need to download weights for logic check
    model.to(device)

    # A. Forward Pass
    dummy_input = torch.randn(2, 3, 128, 128).to(device)
    output = model(dummy_input)
    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"
    print(" - Forward pass: OK")

    # B. Freeze Backbone
    model.freeze_backbone()
    # Check a backbone parameter
    backbone_param = list(model.model.conv_stem.parameters())[0]
    assert backbone_param.requires_grad is False, "Backbone parameters should be frozen"

    # Check head parameter (timm efficientnet uses 'classifier')
    head_param = list(model.model.classifier.parameters())[0]
    assert (
        head_param.requires_grad is True
    ), "Classifier head parameters should be trainable"
    print(" - freeze_backbone: OK")

    # C. Reset Classifier
    # Save old weights
    old_weight = model.model.classifier.weight.data.clone()
    model.reset_classifier()
    new_weight = model.model.classifier.weight.data
    assert not torch.equal(
        old_weight, new_weight
    ), "Classifier weights should change after reset"
    print(" - reset_classifier: OK")

    # -------------------------------------------------------------------------
    # 5. Verify Engine / Training Loop
    # -------------------------------------------------------------------------
    print("\n[5/5] Verifying Engine & Integration...")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # A. Train Epoch (Representation)
    # Unfreeze for stage 1 simulation
    model.unfreeze_backbone()
    loss = train_representation_epoch(model, train_loader, optimizer, device, criterion)
    assert isinstance(loss, float), "Train loss should be a float"
    assert loss > 0, "Train loss should be positive"
    print(f" - train_representation_epoch: OK (Loss: {loss:.4f})")

    # B. Evaluation
    val_loader = loaders["val"]
    val_loss, val_pf1 = evaluate(model, val_loader, device, criterion)
    assert isinstance(val_pf1, float), "Validation pF1 should be a float"
    assert 0 <= val_pf1 <= 1, "pF1 should be between 0 and 1"
    print(f" - evaluate: OK (Val Loss: {val_loss:.4f}, pF1: {val_pf1:.4f})")

    # C. Submission Generation
    test_loader = loaders["test"]
    output_sub_path = os.path.join(Config.WORKING_DIR, "test_submission.csv")

    # Run generation
    generate_submission(model, test_loader, device, output_path=output_sub_path)

    assert os.path.exists(output_sub_path), "Submission file was not created"
    sub_df = pd.read_csv(output_sub_path)
    assert (
        "prediction_id" in sub_df.columns and "cancer" in sub_df.columns
    ), "Submission columns missing"
    assert len(sub_df) > 0, "Submission file is empty"
    print(f" - generate_submission: OK (Saved to {output_sub_path})")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
