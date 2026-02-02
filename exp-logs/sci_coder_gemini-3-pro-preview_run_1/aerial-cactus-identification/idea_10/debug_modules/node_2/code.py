import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.dataset import (
    get_data,
    get_dataloaders,
    get_test_dataloader,
    CactusDataset,
    get_transforms,
    Mixup,
)
from library.models import create_model
from library.engine import train_one_epoch, validate, predict_tta
from library.stacking import train_stacker, predict_stacker


def run_demo():
    print("=" * 40)
    print("Running Cactus Classification Demo")
    print("=" * 40)

    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # Override Config for speed in this demo
    # We only run 1 epoch to demonstrate the pipeline quickly.
    Config.EPOCHS = 1

    # 2. Data Loading & Verification
    print("\n[Step 1] Verifying Data Loading and Augmentation...")

    # Load raw data arrays (Train)
    # get_data handles caching automatically
    train_imgs, train_labels, train_fs = get_data(mode="train", load_cached_data=True)
    print(f"Train Images Shape: {train_imgs.shape}")
    print(f"Train Labels Shape: {train_labels.shape}")
    print(f"Train File Sizes Shape: {train_fs.shape}")

    # Create a dummy dataset/loader to check pipeline integrity
    demo_ds = CactusDataset(
        train_imgs[:128],
        train_fs[:128],
        train_labels[:128],
        transforms=get_transforms("train"),
    )
    demo_loader = DataLoader(demo_ds, batch_size=32, shuffle=True)

    # Fetch a single batch
    imgs, f_sizes, lbls = next(iter(demo_loader))
    print(
        f"Batch Shapes -> Imgs: {imgs.shape}, FileSizes: {f_sizes.shape}, Labels: {lbls.shape}"
    )

    # Assertions to ensure data shapes are correct
    assert imgs.shape == (32, 3, 32, 32), "Image batch shape mismatch"
    assert f_sizes.shape == (32,), "File size batch shape mismatch"
    assert lbls.shape == (32,), "Label batch shape mismatch"

    # Verify Mixup Augmentation
    print("Verifying Mixup...")
    mixup_fn = Mixup(alpha=0.2)
    mixed_imgs, mixed_fs, t_a, t_b, lam = mixup_fn(
        (imgs.to(device), f_sizes.to(device), lbls.to(device))
    )

    assert mixed_imgs.shape == imgs.shape, "Mixed image shape mismatch"
    assert t_a.shape == lbls.shape, "Mixed target shape mismatch"
    print("Mixup verification successful.")

    # 3. Model Instantiation & Forward Pass
    print("\n[Step 2] Verifying Model Architectures...")
    model_names = ["resnet", "repvgg", "next"]

    for name in model_names:
        print(f"Instantiating {name}...")
        model = create_model(name, num_classes=1).to(device)

        # Run dummy forward pass to check GPU compatibility and output shape
        with torch.no_grad():
            output = model(imgs.to(device))

        print(f"  {name} output shape: {output.shape}")
        assert output.shape == (32, 1), f"{name} output shape mismatch"

        # Clean up to save memory
        del model
        torch.cuda.empty_cache()

    # 4. Training Engine Demo
    print("\n[Step 3] Running Training Engine (Fold 0, 1 Epoch)...")

    # Get dataloaders for Fold 0 (Stratified Split)
    train_loader, val_loader = get_dataloaders(fold_idx=0, load_cached_data=True)

    # Create Model (using ResNet for this training demonstration)
    model = create_model("resnet", num_classes=1).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Train for 1 Epoch
    print("Training...")
    avg_loss = train_one_epoch(
        model, train_loader, optimizer, device, criterion, mixup_fn=mixup_fn
    )
    print(f"  Epoch 1 Loss: {avg_loss:.4f}")

    # Validate on validation set
    print("Validating...")
    val_loss, val_auc, val_preds, val_targets = validate(
        model, val_loader, criterion, device
    )
    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Validation AUC:  {val_auc:.4f}")

    # Assertions for metric validity
    assert 0 <= val_auc <= 1, "AUC out of range"
    assert len(val_preds) == len(val_targets), "Prediction/Target length mismatch"

    # 5. Inference Engine Demo (TTA)
    print("\n[Step 4] Running Inference Engine (TTA)...")
    test_loader, test_ids = get_test_dataloader(load_cached_data=True)

    # Predict using Test Time Augmentation (4 views)
    test_preds_resnet = predict_tta(model, test_loader, device)
    print(f"  Test Predictions Shape: {test_preds_resnet.shape}")
    assert len(test_preds_resnet) == len(test_ids), "Test predictions length mismatch"

    # 6. Stacking Demo
    print("\n[Step 5] Demonstrating Stacking Meta-Learner...")

    # To demonstrate the stacking API without waiting for 3 full models to train,
    # we will use the full training set and generate synthetic "OOF" predictions
    # for the other two models (RepVGG and NeXt).

    # Get full training data for stacker training
    full_train_imgs, full_train_labels, full_train_fs = get_data("train")

    # Generate synthetic OOF predictions
    # We add noise to the ground truth to simulate model outputs
    np.random.seed(Config.SEED)
    noise_level = 0.3

    def simulate_preds(y_true):
        # Create noisy predictions based on truth
        preds = y_true * (1 - noise_level) + np.random.rand(len(y_true)) * noise_level
        return np.clip(preds, 0, 1)

    # Construct the dictionary expected by train_stacker
    oof_preds = {
        "resnet": simulate_preds(full_train_labels),  # Simulating ResNet OOF
        "repvgg": simulate_preds(full_train_labels),  # Simulating RepVGG OOF
        "next": simulate_preds(full_train_labels),  # Simulating NeXt OOF
    }

    # Train the Logistic Regression Meta-Learner
    # This combines model predictions + file size metadata
    meta_learner, stacker_auc = train_stacker(
        oof_preds, full_train_fs, full_train_labels
    )
    print(f"  Stacker OOF AUC (Synthetic): {stacker_auc:.4f}")

    # Prepare Test Predictions for Stacking
    # We use the actual ResNet predictions we computed earlier, plus dummies for others
    n_test = len(test_ids)
    test_preds_dict = {
        "resnet": test_preds_resnet,  # Real predictions from our trained model
        "repvgg": np.random.rand(n_test),  # Dummy predictions
        "next": np.random.rand(n_test),  # Dummy predictions
    }

    # Get Test File Sizes for metadata injection
    _, _, test_fs = get_data("test")

    # Generate Final Probabilities using the Stacker
    final_probs = predict_stacker(meta_learner, test_preds_dict, test_fs)
    print(f"  Final Stacked Predictions Shape: {final_probs.shape}")

    # 7. Submission Generation
    print("\n[Step 6] Generating Submission File...")
    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": final_probs})

    # Verify format
    print(submission_df.head(3))

    # Save to disk
    save_path = Config.SUBMISSION_PATH
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")

    # Final verification
    assert os.path.exists(save_path), "Submission file not found"
    print("\nDemo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
