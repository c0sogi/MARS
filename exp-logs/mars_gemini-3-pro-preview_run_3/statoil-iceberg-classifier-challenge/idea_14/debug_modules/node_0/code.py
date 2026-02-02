import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Import library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.model as model_lib
import library.train_eval as train_eval


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # 1. Setup and Configuration Override
    # We create a specific directory for this demo to avoid cluttering the main working dir
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Monkey-patch the config working directory
    config.WORKING_DIR = demo_dir
    print(f"Working Directory set to: {config.WORKING_DIR}")

    # Set seed for reproducibility
    utils.set_seed(config.SEED)
    print("Random seed set.")

    # 2. Data Loader Demonstration
    print("\n--- Testing Data Loader (Debug Mode) ---")
    # Use debug=True to load only 100 samples for speed
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        load_cached_data=True, debug=True
    )

    print(f"Train Loader batches: {len(train_loader)}")
    print(f"Val Loader batches: {len(val_loader)}")
    print(f"Test Loader batches: {len(test_loader)}")

    # Fetch one batch to verify shapes
    images, angles, labels, ids = next(iter(train_loader))

    print(
        f"Batch Shapes -> Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    # Assertions
    # Expected image shape: (Batch, 3, 75, 75)
    assert images.dim() == 4, "Images should be 4D tensors (B, C, H, W)"
    assert images.size(1) == 3, "Images should have 3 channels (Band1, Band2, Avg)"
    assert images.size(2) == 75 and images.size(3) == 75, "Images should be 75x75"
    # Expected angle shape: (Batch,)
    assert angles.dim() == 1, "Angles should be 1D tensors"
    # Expected label shape: (Batch,)
    assert labels.dim() == 1, "Labels should be 1D tensors"

    print("Data Loader verification passed.")

    # 3. Model Demonstration
    print("\n--- Testing Model Architecture ---")
    device = config.DEVICE
    print(f"Using device: {device}")

    net = model_lib.HybridWideSEResNet().to(device)

    # Move batch to device
    images = images.to(device)
    angles = angles.to(device)

    # Forward pass
    outputs = net(images, angles)

    print(f"Model Output Shape: {outputs.shape}")

    # Assertions
    # Output should be (Batch, 1) logits
    assert outputs.dim() == 2, "Output should be 2D (B, 1)"
    assert outputs.size(0) == images.size(0), "Output batch size should match input"
    assert outputs.size(1) == 1, "Output should have 1 class logit"
    assert outputs.requires_grad, "Output should require gradients for training"

    print("Model architecture verification passed.")

    # 4. Training and Evaluation Functions
    print("\n--- Testing Training and Evaluation Functions ---")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # Train for one epoch
    print("Running train_one_epoch...")
    train_loss = train_eval.train_one_epoch(
        net, train_loader, criterion, optimizer, device
    )
    print(f"Train Loss: {train_loss:.4f}")

    assert np.isfinite(train_loss), "Training loss should be finite"

    # Evaluate
    print("Running evaluate...")
    val_loss = train_eval.evaluate(net, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")

    assert np.isfinite(val_loss), "Validation loss should be finite"

    # Predict
    print("Running predict...")
    pred_ids, pred_probs = train_eval.predict(net, test_loader, device)

    print(f"Predictions generated: {len(pred_probs)}")
    print(f"Sample predictions: {pred_probs[:5]}")

    # Assertions
    assert len(pred_ids) == len(pred_probs), "IDs and Predictions count mismatch"
    assert len(pred_probs) == len(
        test_loader.dataset
    ), "Should predict for all test samples"
    assert (pred_probs >= 0).all() and (
        pred_probs <= 1
    ).all(), "Probabilities must be in [0, 1]"

    print("Training/Eval/Predict functions verification passed.")

    # 5. Checkpoint Utility
    print("\n--- Testing Checkpoint Utilities ---")

    # Save current state
    checkpoint_state = {
        "epoch": 1,
        "state_dict": net.state_dict(),
        "best_score": val_loss,
        "optimizer": optimizer.state_dict(),
    }

    fold_idx = 0
    utils.save_checkpoint(
        checkpoint_state, is_best=True, checkpoint_dir=demo_dir, fold_idx=fold_idx
    )

    checkpoint_path = os.path.join(demo_dir, f"checkpoint_fold_{fold_idx}.pth")
    best_model_path = os.path.join(demo_dir, f"model_best_fold_{fold_idx}.pth")

    assert os.path.exists(checkpoint_path), "Checkpoint file not created"
    assert os.path.exists(best_model_path), "Best model file not created"

    # Load checkpoint
    # Create a new model instance to verify loading
    new_net = model_lib.HybridWideSEResNet().to(device)
    new_optimizer = optim.Adam(new_net.parameters(), lr=1e-3)

    start_epoch, best_score = utils.load_checkpoint(
        checkpoint_path, new_net, new_optimizer, device
    )

    print(f"Loaded Checkpoint -> Epoch: {start_epoch}, Best Score: {best_score:.4f}")

    # Verify weights match
    original_weights = net.head[0].weight.data
    loaded_weights = new_net.head[0].weight.data

    if torch.equal(original_weights, loaded_weights):
        print("Weights match successfully.")
    else:
        raise AssertionError("Weights do not match after loading checkpoint!")

    assert start_epoch == 1, "Loaded epoch incorrect"
    assert best_score == val_loss, "Loaded best score incorrect"

    print("Checkpoint utilities verification passed.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
