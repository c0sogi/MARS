import os
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Import functions and classes from the provided library files
from library.utils import seed_everything
from library.dataset import get_datasets
from library.models import get_bird_model
from library.optimization import get_optimizer_with_llrd, Lookahead
from library.training import train_model
from library.inference import (
    predict_with_tta,
    save_submission,
    load_and_average_checkpoints,
)


def main():
    # 1. Setup and Configuration
    print("Initializing demonstration...")
    seed_everything(42)

    # Define working directories
    work_dir = "./working/demo_execution"
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    submission_dir = os.path.join(work_dir, "submission")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Data Loading (Dataset & DataLoader)
    print("\n--- Step 1: Data Loading ---")
    # Load a small subset of data (32 samples) for speed
    batch_size = 8
    train_dataset, val_dataset, test_dataset = get_datasets(
        load_cached_data=False,  # Force processing from metadata to demonstrate logic
        max_samples=32,
    )

    # Verify dataset lengths
    assert len(train_dataset) <= 32
    assert len(val_dataset) <= 32
    assert len(test_dataset) <= 32

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Verify batch structure
    images, labels, rec_ids = next(iter(train_loader))
    print(
        f"Batch shapes - Images: {images.shape}, Labels: {labels.shape}, IDs: {rec_ids.shape}"
    )

    # Assertions for data integrity
    # Images: (Batch, Channels, Height, Width) -> (8, 3, 224, 224)
    assert images.shape == (batch_size, 3, 224, 224), "Incorrect image tensor shape"
    # Labels: (Batch, NumClasses) -> (8, 19)
    assert labels.shape == (batch_size, 19), "Incorrect label tensor shape"
    assert images.dtype == torch.float32, "Images should be float32"
    assert labels.dtype == torch.float32, "Labels should be float32"

    # 3. Model Instantiation
    print("\n--- Step 2: Model Initialization ---")
    model_name = "resnet18"
    model = get_bird_model(model_name, num_classes=19, pretrained=True)
    model.to(device)

    # Verify model output shape
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (
        2,
        19,
    ), f"Model output shape mismatch. Expected (2, 19), got {output.shape}"
    print(f"Model {model_name} initialized successfully.")

    # 4. Optimization Configuration
    print("\n--- Step 3: Optimizer Setup ---")
    # Use the custom LLRD optimizer setup
    optimizer = get_optimizer_with_llrd(
        model, model_name, lr=1e-3, weight_decay=1e-4, layer_decay=0.9
    )

    # Verify it is a Lookahead optimizer
    assert isinstance(
        optimizer, Lookahead
    ), "Optimizer should be an instance of Lookahead"
    print("Lookahead optimizer with LLRD configured.")

    # 5. Training Loop
    print("\n--- Step 4: Training ---")
    save_path = os.path.join(checkpoint_dir, "best_model.pth")

    # Train for 1 epoch to demonstrate the loop
    best_auc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,  # Skipping scheduler for this short demo
        device=device,
        num_epochs=1,
        patience=1,
        save_path=save_path,
    )

    print(f"Training finished. Best AUC: {best_auc}")
    assert os.path.exists(save_path), "Checkpoint file was not created."

    # 6. Inference (TTA & Submission)
    print("\n--- Step 5: Inference ---")

    # Demonstrate predict_with_tta directly
    print("Running TTA inference on test set...")
    probs, ids = predict_with_tta(model, test_loader, device)

    assert probs.shape == (
        len(test_dataset),
        19,
    ), "Prediction probability shape mismatch"
    assert len(ids) == len(test_dataset), "Prediction IDs count mismatch"

    # Demonstrate load_and_average_checkpoints (Ensembling)
    # We will use the single checkpoint we just saved
    print("Running ensemble inference from checkpoint...")
    avg_probs, final_ids = load_and_average_checkpoints(
        model_name=model_name,
        checkpoint_paths=[save_path],
        loader=test_loader,
        device=device,
        num_classes=19,
    )

    assert np.allclose(
        probs, avg_probs, atol=1e-5
    ), "Direct inference and checkpoint inference should match for single model"

    # 7. Saving Submission
    print("\n--- Step 6: Saving Submission ---")
    submission_path = os.path.join(submission_dir, "submission.csv")
    save_submission(final_ids, avg_probs, submission_path)

    # Verify submission file
    assert os.path.exists(submission_path), "Submission file not found"
    df_sub = pd.read_csv(submission_path)

    # Expected rows: num_test_samples * 19 classes
    expected_rows = len(test_dataset) * 19
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"
    assert (
        "Id" in df_sub.columns and "Probability" in df_sub.columns
    ), "Submission columns missing"

    print("Demonstration completed successfully!")


if __name__ == "__main__":
    main()
