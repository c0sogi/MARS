import os
import torch
import pandas as pd
import numpy as np
import sys

# Import the provided library modules
import library.config as config
import library.utils as utils
import library.dataset as dataset
import library.model as model
import library.trainer as trainer


def run_demonstration():
    print("=== Starting Library Demonstration ===")

    # 1. Override Configuration for Speed
    # We use a small sample size and minimal epochs to demonstrate functionality quickly.
    print("\n[Step 1] Configuring for fast demonstration...")
    config.DEBUG_SAMPLE_SIZE = 100
    config.BATCH_SIZE = 16
    config.NUM_EPOCHS = 1
    config.NUM_WORKERS = 2  # Reduce workers to minimize overhead for small data

    # Ensure reproducibility
    config.set_seed(42)
    print(f"Debug Sample Size: {config.DEBUG_SAMPLE_SIZE}")
    print(f"Batch Size: {config.BATCH_SIZE}")

    # 2. Demonstrate Utils
    print("\n[Step 2] Verifying Utils...")

    # Test Class Mappings
    label_to_idx, idx_to_label = utils.get_class_mappings()
    assert len(label_to_idx) == config.NUM_CLASSES, "Label mapping length mismatch"
    assert len(idx_to_label) == config.NUM_CLASSES, "Index mapping length mismatch"
    # Verify round-trip consistency for a sample class
    sample_cat_id = list(label_to_idx.keys())[0]
    assert (
        idx_to_label[label_to_idx[sample_cat_id]] == sample_cat_id
    ), "Mapping round-trip failed"
    print("Class mappings verified.")

    # Test Class Weights
    # Note: This calculates weights based on the full CSV unless cached,
    # but the function handles caching.
    class_weights = utils.calculate_class_weights()
    assert isinstance(class_weights, torch.Tensor), "Class weights should be a Tensor"
    assert class_weights.shape[0] == config.NUM_CLASSES, "Class weights shape mismatch"
    print("Class weights calculation verified.")

    # 3. Demonstrate Dataset and DataLoaders
    print("\n[Step 3] Verifying Dataset and DataLoaders...")
    train_loader, val_loader, test_loader = dataset.get_dataloaders()

    # Verify Train Loader
    assert (
        len(train_loader.dataset) == config.DEBUG_SAMPLE_SIZE
    ), "Train dataset size mismatch"

    # Fetch one batch to verify shapes
    images, labels = next(iter(train_loader))
    print(f"Train Batch - Images: {images.shape}, Labels: {labels.shape}")

    assert images.shape == (
        config.BATCH_SIZE,
        3,
        config.IMG_SIZE,
        config.IMG_SIZE,
    ), "Image batch shape incorrect"
    assert labels.shape == (config.BATCH_SIZE,), "Label batch shape incorrect"
    assert images.dtype == torch.float32, "Images should be float32"
    assert labels.dtype == torch.long, "Labels should be long (int64)"
    print("DataLoaders verified.")

    # 4. Demonstrate Model
    print("\n[Step 4] Verifying Model...")
    net = model.get_model()

    # Verify Backbone Freezing
    # The first layer of ResNet is usually a Conv2d
    first_layer_param = next(net.backbone.conv1.parameters())
    assert first_layer_param.requires_grad is False, "Backbone should be frozen"

    # Verify Head Trainability
    # The fc layer should be trainable
    assert (
        net.backbone.fc.weight.requires_grad is True
    ), "Classification head should be trainable"

    # Verify Forward Pass
    net.eval()
    with torch.no_grad():
        outputs = net(images)  # Using the batch fetched earlier

    assert outputs.shape == (
        config.BATCH_SIZE,
        config.NUM_CLASSES,
    ), "Model output shape incorrect"
    print("Model architecture and forward pass verified.")

    # 5. Demonstrate Trainer (Training & Validation)
    print("\n[Step 5] Verifying Trainer (Fit Loop)...")
    # Initialize Trainer
    plant_trainer = trainer.Trainer(model=net)

    # Run Training
    # This will run for 1 epoch on the small subset
    plant_trainer.fit(train_loader, val_loader, num_epochs=config.NUM_EPOCHS)

    # Check if checkpoint was saved (only if validation improved, which is likely with random init vs data)
    # However, with 1 epoch and tiny data, it might or might not improve depending on init loss vs val loss.
    # We just ensure the code ran without error.
    print("Training loop completed successfully.")

    # 6. Demonstrate Inference
    print("\n[Step 6] Verifying Inference and Submission...")
    plant_trainer.predict(test_loader)

    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    # Verify Submission Content
    sub_df = pd.read_csv(submission_path)
    print(f"Submission shape: {sub_df.shape}")

    # Check columns
    assert (
        "Id" in sub_df.columns and "Predicted" in sub_df.columns
    ), "Submission columns missing"

    # Check length matches test dataset (DEBUG_SAMPLE_SIZE)
    assert (
        len(sub_df) == config.DEBUG_SAMPLE_SIZE
    ), f"Submission length {len(sub_df)} != {config.DEBUG_SAMPLE_SIZE}"

    # Check values are valid category IDs
    # We pick a prediction and ensure it exists in our known categories
    sample_pred = sub_df["Predicted"].iloc[0]
    assert (
        sample_pred in label_to_idx
    ), f"Predicted class {sample_pred} not found in category mappings"

    print("Inference and submission verification successful.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
