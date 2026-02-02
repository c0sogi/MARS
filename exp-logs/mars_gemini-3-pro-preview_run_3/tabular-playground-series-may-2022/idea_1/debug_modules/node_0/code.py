import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import (
    DEVICE,
    WORKING_DIR,
    SUBMISSION_DIR,
    CONTINUOUS_FEATURES,
    CACHE_DIR,
    SEED,
)
from library.utils import seed_everything
from library.data_processor import make_dataloaders
from library.model import EntityEmbeddingMLP
from library.trainer import Trainer, run_training_pipeline


def clean_cache():
    """Cleans up the cache directory to ensure fresh processing for demonstration."""
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)


def demo_data_loading():
    print("\n=== Demo: Data Loading and Processing ===")

    # Force fresh processing by cleaning cache and setting load_cached_data=False
    clean_cache()

    batch_size = 1024
    print(f"Creating DataLoaders with batch_size={batch_size}...")
    train_loader, val_loader, test_loader = make_dataloaders(
        batch_size=batch_size, load_cached_data=False
    )

    # Verify Train Loader
    batch = next(iter(train_loader))
    continuous = batch["continuous"]
    categorical = batch["categorical"]
    targets = batch["target"]

    print(f"Train Batch - Continuous Shape: {continuous.shape}")
    print(f"Train Batch - Categorical Shape: {categorical.shape}")
    print(f"Train Batch - Target Shape: {targets.shape}")

    # Assertions
    # 28 continuous features (f_00..f_26 + f_28)
    assert (
        continuous.shape[1] == 28
    ), f"Expected 28 continuous features, got {continuous.shape[1]}"
    # 2 discrete (f_29, f_30) + 10 chars from f_27 = 12 categorical features
    assert (
        categorical.shape[1] == 12
    ), f"Expected 12 categorical features, got {categorical.shape[1]}"
    assert targets.shape[0] == batch_size, "Target batch size mismatch"

    print("Data loading and feature engineering verified successfully.")
    return train_loader, val_loader, test_loader


def demo_model_initialization(train_loader):
    print("\n=== Demo: Model Initialization and Forward Pass ===")

    # Get vocab sizes from the dataset
    vocab_sizes = train_loader.dataset.vocab_sizes
    num_continuous = 28  # Known from config/verification

    print(f"Vocabulary Sizes: {vocab_sizes}")

    model = EntityEmbeddingMLP(
        vocab_sizes=vocab_sizes,
        num_continuous=num_continuous,
        embedding_dim=8,  # Reduced for demo speed
        hidden_layers=[64, 32],  # Reduced for demo speed
    )
    model.to(DEVICE)

    # Run a dummy forward pass
    batch = next(iter(train_loader))
    cont_data = batch["continuous"].to(DEVICE)
    cat_data = batch["categorical"].to(DEVICE)

    output = model(cont_data, cat_data)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (cont_data.size(0), 1), "Output shape mismatch"
    assert (
        output.min() >= 0 and output.max() <= 1
    ), "Output probabilities out of range [0, 1]"

    print("Model initialization and forward pass verified successfully.")
    return model


def demo_training_loop(model, train_loader, val_loader):
    print("\n=== Demo: Training Loop (1 Epoch) ===")

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    # Use a temporary checkpoint path
    ckpt_path = os.path.join(WORKING_DIR, "demo_model.pth")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        patience=1,
        checkpoint_path=ckpt_path,
    )

    # Run for just 1 epoch to verify the loop works
    print("Starting training for 1 epoch...")
    trainer.fit(train_loader, val_loader, epochs=1)

    # Verify that a model was saved (since we start with -inf best_auc, any valid epoch saves it)
    assert os.path.exists(ckpt_path), "Checkpoint file was not created."
    assert (
        trainer.best_auc > 0.5
    ), f"Expected AUC > 0.5 (random guess), got {trainer.best_auc}"

    print(f"Training loop verified. Best AUC: {trainer.best_auc:.4f}")
    return trainer


def demo_inference(trainer, test_loader):
    print("\n=== Demo: Inference ===")

    preds = trainer.predict(test_loader)

    print(f"Predictions Shape: {preds.shape}")
    print(f"Sample Predictions: {preds[:5]}")

    # Assertions
    assert len(preds) == len(
        test_loader.dataset
    ), "Prediction count mismatch with test set size"
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Predictions out of valid probability range"

    print("Inference verified successfully.")


def demo_full_pipeline():
    print("\n=== Demo: Full Training Pipeline ===")

    # Run the high-level pipeline function provided in library
    # We use cached data this time to save time, as we generated it in the first step
    best_auc = run_training_pipeline(
        batch_size=2048,  # Larger batch for speed
        epochs=1,
        learning_rate=0.005,
        load_cached_data=True,
    )

    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Verify Submission
    assert os.path.exists(submission_path), "Submission file not found."

    df_sub = pd.read_csv(submission_path)
    print(f"Submission File Head:\n{df_sub.head()}")

    assert (
        "id" in df_sub.columns and "target" in df_sub.columns
    ), "Submission columns mismatch"
    assert len(df_sub) == 100000, f"Expected 100,000 predictions, got {len(df_sub)}"

    print(f"Full pipeline execution successful. Final AUC: {best_auc:.4f}")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(SEED)

    try:
        # 1. Test Data Loading
        train_loader, val_loader, test_loader = demo_data_loading()

        # 2. Test Model
        model = demo_model_initialization(train_loader)

        # 3. Test Training
        trainer = demo_training_loop(model, train_loader, val_loader)

        # 4. Test Inference
        demo_inference(trainer, test_loader)

        # 5. Test Full Pipeline Integration
        demo_full_pipeline()

        print("\nAll demonstrations and verifications passed successfully!")

    except AssertionError as e:
        print(f"\nVERIFICATION FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        exit(1)
