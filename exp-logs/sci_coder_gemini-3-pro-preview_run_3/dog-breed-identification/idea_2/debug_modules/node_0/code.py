import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library import utils
from library import dataset
from library import model
from library import train
from library import predict


def main():
    print("Starting Dog Breed Classification Library Demonstration...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[1] Configuring environment for rapid demonstration...")

    # Modify Config for speed and debugging
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use a very small subset
    Config.BATCH_SIZE = 8
    Config.WARMUP_EPOCHS = 1
    Config.FINE_TUNE_EPOCHS = 1
    Config.NUM_WORKERS = 2  # Reduce workers for small debug run

    # Ensure reproducibility
    utils.set_seed(Config.SEED)
    print("Configuration updated: DEBUG=True, Epochs=1, BatchSize=8")

    # ==========================================
    # 2. Verify Utilities
    # ==========================================
    print("\n[2] Verifying Utility Functions...")

    # Test Log Loss Calculation
    # Scenario: 2 samples, 3 classes.
    # Sample 0: True class 0. Pred: [0.8, 0.1, 0.1] -> Good
    # Sample 1: True class 2. Pred: [0.1, 0.2, 0.7] -> Good
    y_true_indices = np.array([0, 2])
    y_pred_probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])
    labels = [0, 1, 2]

    loss = utils.calculate_log_loss(y_true_indices, y_pred_probs, labels=labels)
    print(f"Calculated Log Loss (Dummy Data): {loss:.4f}")

    # Assert loss is reasonable (low because predictions are good)
    assert loss < 0.4, "Log loss calculation seems incorrect for good predictions."

    # ==========================================
    # 3. Verify Dataset & DataLoaders
    # ==========================================
    print("\n[3] Verifying Dataset and DataLoaders...")

    # Force processing of data (this will generate cache files in ./working/idea_2)
    dataloaders, class_names = dataset.get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,  # Force reload to test processing logic
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    print(f"Number of classes: {len(class_names)}")
    assert (
        len(class_names) == Config.NUM_CLASSES
    ), f"Expected {Config.NUM_CLASSES} classes, found {len(class_names)}"

    # Fetch one batch from training
    images, targets = next(iter(train_loader))

    print(f"Train Batch - Images Shape: {images.shape}")
    print(f"Train Batch - Targets Shape: {targets.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect image tensor shape"
    assert targets.shape == (Config.BATCH_SIZE,), "Incorrect target tensor shape"
    assert targets.max() < Config.NUM_CLASSES, "Target index out of bounds"

    # ==========================================
    # 4. Verify Model Architecture
    # ==========================================
    print("\n[4] Verifying Model Architecture...")

    net = model.get_model(num_classes=Config.NUM_CLASSES, pretrained=False)
    net = net.to(Config.DEVICE)

    # Test Forward Pass
    with torch.no_grad():
        # Use the batch fetched earlier
        images = images.to(Config.DEVICE)
        outputs = net(images)

    print(f"Model Output Shape: {outputs.shape}")
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"

    # Test Freeze Backbone
    print("Testing Backbone Freezing...")
    model.freeze_backbone(net)

    # Check if features are frozen and classifier is not
    for name, param in net.features.named_parameters():
        assert (
            param.requires_grad is False
        ), f"Backbone parameter {name} should be frozen"

    for name, param in net.classifier.named_parameters():
        assert (
            param.requires_grad is True
        ), f"Classifier parameter {name} should be trainable"

    # Test Unfreeze All
    print("Testing Unfreeze All...")
    model.unfreeze_all(net)
    for param in net.parameters():
        assert (
            param.requires_grad is True
        ), "All parameters should be trainable after unfreeze"

    # ==========================================
    # 5. Run Training Pipeline
    # ==========================================
    print("\n[5] Running Training Pipeline (Warmup + Finetune)...")

    # We use the train.run_training function which handles the loops
    # Note: This function uses the global Config, which we modified in step 1.
    best_model_path = train.run_training()

    print(f"Training finished. Best model saved at: {best_model_path}")
    assert os.path.exists(best_model_path), "Best model file was not created."

    # ==========================================
    # 6. Generate Submission
    # ==========================================
    print("\n[6] Generating Submission...")

    # Generate submission using the trained model
    predict.generate_submission(
        model_path=best_model_path, batch_size=Config.BATCH_SIZE
    )

    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file not found."

    # Verify Submission Content
    df_sub = pd.read_csv(submission_path)
    print(f"Submission DataFrame Shape: {df_sub.shape}")
    print("First 3 rows of submission:")
    print(df_sub.head(3))

    # Assertions on submission
    # Rows should be equal to DEBUG_SAMPLE_SIZE (50)
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} rows in submission, got {len(df_sub)}"

    # Columns should be id + 120 breeds
    expected_cols = 1 + Config.NUM_CLASSES
    assert (
        len(df_sub.columns) == expected_cols
    ), f"Expected {expected_cols} columns, got {len(df_sub.columns)}"

    # Check probability constraints (sum close to 1 is not guaranteed by softmax on logits individually unless normalized,
    # but predict_tta applies softmax. Let's check if rows sum to approx 1)
    row_sums = df_sub.iloc[:, 1:].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1.0"

    print("\nAll demonstrations and verifications passed successfully!")


if __name__ == "__main__":
    main()
