import os
import shutil
import torch
import numpy as np
from library.utils import set_seed
from library.dataset import get_loaders
from library.model import DPACNN
from library.trainer import run_fold


def main():
    # ------------------------------------------------------------------------
    # 1. Setup and Configuration
    # ------------------------------------------------------------------------
    print("Initializing demonstration...")
    set_seed(42)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define temporary directories for this demo
    # We use a specific directory for checkpoints to avoid conflicts
    demo_checkpoint_dir = "./working/demo_checkpoints"
    if os.path.exists(demo_checkpoint_dir):
        shutil.rmtree(demo_checkpoint_dir)
    os.makedirs(demo_checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------------
    # 2. Data Loading
    # ------------------------------------------------------------------------
    print("\n--- Data Loading ---")
    # We use num_workers=0 to avoid multiprocessing overhead for this quick demo
    # load_cached_data=True allows the dataset to cache processed numpy arrays
    # in ./working/idea_35/ (as defined in dataset.py)
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=16, load_cached_data=True, num_workers=0
    )

    # Verify Train Loader
    print("Verifying Train Loader...")
    tr_images, tr_angles, tr_labels = next(iter(train_loader))

    # Expected shape: (Batch, 3, 75, 75)
    print(f"Train Batch Images Shape: {tr_images.shape}")
    print(f"Train Batch Angles Shape: {tr_angles.shape}")
    print(f"Train Batch Labels Shape: {tr_labels.shape}")

    assert tr_images.shape == (16, 3, 75, 75), "Train image batch shape incorrect"
    assert tr_angles.shape == (16,), "Train angle batch shape incorrect"
    assert tr_labels.shape == (16,), "Train label batch shape incorrect"
    assert tr_images.dtype == torch.float32, "Image tensor dtype should be float32"

    # Verify Test Loader (Yields IDs instead of labels)
    print("Verifying Test Loader...")
    te_images, te_angles, te_ids = next(iter(test_loader))
    print(f"Test Batch IDs: {te_ids[:5]}...")

    assert len(te_ids) == 16, "Test ID batch size incorrect"
    assert isinstance(te_ids[0], str), "Test IDs should be strings"

    # ------------------------------------------------------------------------
    # 3. Model Instantiation
    # ------------------------------------------------------------------------
    print("\n--- Model Instantiation ---")
    model = DPACNN().to(device)

    # Verify Forward Pass
    print("Verifying Forward Pass...")
    dummy_images = tr_images.to(device)
    dummy_angles = tr_angles.to(device)

    with torch.no_grad():
        logits = model(dummy_images, dummy_angles)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (16, 1), "Model output shape mismatch. Expected (16, 1)"

    # ------------------------------------------------------------------------
    # 4. Training Demonstration
    # ------------------------------------------------------------------------
    print("\n--- Training Demonstration ---")
    # Train for a minimal number of epochs to demonstrate functionality
    epochs = 2
    learning_rate = 1e-3

    trained_model, best_val_loss = run_fold(
        train_loader=train_loader,
        val_loader=val_loader,
        fold_idx=0,
        epochs=epochs,
        patience=2,  # Short patience for demo
        learning_rate=learning_rate,
        device=device,
        checkpoint_dir=demo_checkpoint_dir,
    )

    print(f"Training finished. Best Validation Loss: {best_val_loss:.4f}")

    # Verify checkpoint creation
    expected_ckpt_path = os.path.join(demo_checkpoint_dir, "model_fold_0.pth")
    assert os.path.exists(
        expected_ckpt_path
    ), f"Checkpoint not found at {expected_ckpt_path}"

    # ------------------------------------------------------------------------
    # 5. Inference Demonstration
    # ------------------------------------------------------------------------
    print("\n--- Inference Demonstration ---")
    trained_model.eval()

    predictions = []
    ids_processed = []

    # Process just the first few batches of the test set
    print("Predicting on test set (subset)...")
    with torch.no_grad():
        for i, (images, angles, ids) in enumerate(test_loader):
            if i >= 3:
                break  # Limit to 3 batches for speed

            images = images.to(device)
            angles = angles.to(device)

            # Forward pass
            outputs = trained_model(images, angles)

            # Convert logits to probabilities
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            predictions.extend(probs)
            ids_processed.extend(ids)

    print(f"Generated {len(predictions)} predictions.")
    print(f"Sample predictions: {list(zip(ids_processed[:3], predictions[:3]))}")

    # Verify probability range
    assert all(
        0.0 <= p <= 1.0 for p in predictions
    ), "Predictions must be probabilities between 0 and 1"

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
