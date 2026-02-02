import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Import from the provided library files
from library.config import (
    DEVICE,
    WORKING_DIR,
    SUBMISSION_DIR,
    MODEL_CHECKPOINT_TEMPLATE,
    SEEDS,
)
from library.utils import seed_everything, load_checkpoint
from library.dataset import get_dataloaders
from library.model import MultiScaleResNet
from library.engine import train_model, predict_with_tta


def main():
    print("Starting demonstration of Cactus Identification pipeline...")

    # 1. Reproducibility
    seed_everything(42)
    print("Seed set to 42.")

    # 2. Data Loading Demonstration
    # We use a small subset (max_samples=100) and small batch size for speed
    print("\n--- Demonstrating Data Loading ---")
    batch_size = 16
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=2,
        load_cached_data=False,  # Force processing to demo that logic
        max_samples=100,
    )

    # Verify Train Loader
    images, labels, ids = next(iter(train_loader))

    # Check shapes
    assert images.shape == (
        batch_size,
        3,
        32,
        32,
    ), f"Expected image batch shape (16, 3, 32, 32), got {images.shape}"
    assert labels.shape == (
        batch_size,
    ), f"Expected label batch shape (16,), got {labels.shape}"
    assert len(ids) == batch_size, "IDs length mismatch"

    # Check types
    assert isinstance(images, torch.Tensor), "Images should be a Tensor"
    assert isinstance(labels, torch.Tensor), "Labels should be a Tensor"
    assert labels.dtype == torch.float32, "Labels should be float32"

    print(f"Data verification passed. Batch shape: {images.shape}")

    # 3. Model Instantiation & Forward Pass
    print("\n--- Demonstrating Model Architecture ---")
    model = MultiScaleResNet(num_classes=1).to(DEVICE)

    # Create a dummy input on the correct device
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(DEVICE)

    # Forward pass
    output = model(dummy_input)

    # Verify output
    assert output.shape == (
        batch_size,
        1,
    ), f"Expected output shape (16, 1), got {output.shape}"
    print(f"Model forward pass successful. Output shape: {output.shape}")

    # 4. Training Loop Demonstration
    print("\n--- Demonstrating Training Loop ---")

    # Setup training components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    # Define a specific checkpoint path for this demo
    demo_checkpoint_path = os.path.join(WORKING_DIR, "demo_model.pth")

    # Run training for 2 epochs
    best_auc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        num_epochs=2,
        patience=2,
        save_path=demo_checkpoint_path,
    )

    print(f"Training complete. Best Validation AUC: {best_auc}")

    # Verify checkpoint existence
    assert os.path.exists(demo_checkpoint_path), "Checkpoint file was not created."
    print(f"Checkpoint verified at: {demo_checkpoint_path}")

    # 5. Inference Demonstration
    print("\n--- Demonstrating Inference with TTA ---")

    # Load the model from checkpoint
    loaded_model = MultiScaleResNet(num_classes=1).to(DEVICE)
    epoch, loaded_score = load_checkpoint(demo_checkpoint_path, loaded_model)

    print(f"Model loaded from epoch {epoch} with score {loaded_score}")

    # Run inference
    ids_pred, probs_pred = predict_with_tta(loaded_model, test_loader, DEVICE)

    # Verify predictions
    # Note: test_loader also has max_samples=100 applied
    expected_samples = min(100, len(test_loader.dataset))

    assert (
        len(ids_pred) == expected_samples
    ), f"Expected {expected_samples} IDs, got {len(ids_pred)}"
    assert (
        len(probs_pred) == expected_samples
    ), f"Expected {expected_samples} predictions, got {len(probs_pred)}"
    assert np.all(
        (probs_pred >= 0) & (probs_pred <= 1)
    ), "Probabilities must be between 0 and 1"

    print(f"Inference successful. Generated {len(probs_pred)} predictions.")
    print(f"Sample Prediction: ID={ids_pred[0]}, Prob={probs_pred[0]:.4f}")

    # 6. Submission File Generation (Mock)
    print("\n--- Demonstrating Submission Generation ---")
    submission_df = pd.DataFrame({"id": ids_pred, "has_cactus": probs_pred})

    demo_submission_path = os.path.join(SUBMISSION_DIR, "demo_submission.csv")
    submission_df.to_csv(demo_submission_path, index=False)

    assert os.path.exists(demo_submission_path), "Submission file not found."
    print(f"Submission file saved to: {demo_submission_path}")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
