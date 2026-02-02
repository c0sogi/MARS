import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import components from the provided library
from library.config import DEVICE, SUBMISSION_DIR, CHECKPOINT_DIR
from library.utils import seed_everything
from library.dataset import get_datasets
from library.model import UltraWideRepResNeXt
from library.engine import train_model, predict_tta


def main():
    print("=== Starting Cactus Classification Demonstration ===")

    # 1. Set Seed for Reproducibility
    seed_everything(42)
    print("Random seed set to 42.")

    # 2. Load Data
    # We use debug=True and a small sample size to ensure the demo runs quickly.
    print("\nLoading datasets (Debug Mode)...")
    train_dataset, val_dataset, test_dataset = get_datasets(
        load_cached_data=False, debug=True, debug_size=200
    )

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    # 3. Initialize Model
    print("\nInitializing UltraWideRepResNeXt model...")
    model = UltraWideRepResNeXt(num_classes=1, deploy=False)
    model.to(DEVICE)

    # 4. Setup Training Components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    # Short schedule for demonstration
    num_epochs = 2
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 5. Run Training Loop
    print(f"\nStarting training for {num_epochs} epochs...")
    best_auc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=DEVICE,
        save_name="demo_checkpoint.pth",
    )
    print(f"Training complete. Best Validation AUC: {best_auc:.4f}")

    # 6. Verify Re-parameterization Logic
    # This step ensures that fusing the branches (Conv+BN + 1x1 + Identity)
    # results in the same mathematical operation as the multi-branch structure.
    print("\nVerifying Re-parameterization (Train Mode -> Deploy Mode)...")
    model.eval()

    # Generate random input for verification
    dummy_input = torch.randn(2, 3, 32, 32).to(DEVICE)

    with torch.no_grad():
        # Get output from the multi-branch model
        output_train_mode = model(dummy_input)

        # Switch model to deploy mode (fuses layers in-place)
        model.switch_to_deploy()

        # Get output from the fused single-branch model
        output_deploy_mode = model(dummy_input)

        # Calculate difference
        diff = (output_train_mode - output_deploy_mode).abs().max().item()
        print(f"Max absolute difference between modes: {diff:.8f}")

        # Assert that the difference is negligible (allowing for small float precision errors)
        # Typically < 1e-5 is expected.
        if diff > 1e-4:
            raise AssertionError(
                f"Re-parameterization failed! Difference {diff} is too large."
            )
        print("Verification Successful: Model outputs match after fusion.")

    # 7. Inference
    print("\nRunning Inference with Test Time Augmentation (TTA)...")
    # The model is now in deploy mode, which is faster for inference
    test_ids, test_preds = predict_tta(model, test_loader, DEVICE)

    print(f"Inference complete. Generated {len(test_preds)} predictions.")

    # 8. Save Submission
    print("\nSaving submission file...")
    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": test_preds})

    save_path = os.path.join(SUBMISSION_DIR, "submission_demo.csv")
    submission_df.to_csv(save_path, index=False)

    # Final Validation
    assert os.path.exists(save_path), "Submission file was not created."
    assert len(submission_df) == len(test_dataset), "Submission row count mismatch."
    assert submission_df["has_cactus"].min() >= 0.0, "Probabilities < 0 detected."
    assert submission_df["has_cactus"].max() <= 1.0, "Probabilities > 1 detected."

    print(f"Submission saved to: {save_path}")
    print("=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
