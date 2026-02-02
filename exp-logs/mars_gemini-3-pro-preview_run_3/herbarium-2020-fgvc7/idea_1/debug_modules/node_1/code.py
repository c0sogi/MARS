import os
import pandas as pd
import torch
import numpy as np
import library.config as config
import library.dataset as dataset
import library.model as model_lib
import library.trainer as trainer
import library.inference as inference


def setup_environment_and_data():
    """
    Sets up a fast execution environment by creating small subsets of the metadata
    and redirecting the configuration to use them.
    """
    print(">>> Setting up temporary environment for fast demonstration...")

    # 1. Set Seed
    config.set_seed(42)

    # 2. Create Subset Metadata
    # We create very small subsets (e.g., 50 samples) to ensure the code runs in seconds/minutes.
    subset_size = 50

    # Load originals
    train_df = pd.read_csv("./metadata/train.csv")
    val_df = pd.read_csv("./metadata/val.csv")
    test_df = pd.read_csv("./metadata/test.csv")

    # Sample
    train_subset = train_df.head(subset_size).copy()
    val_subset = val_df.head(subset_size).copy()
    test_subset = test_df.head(subset_size).copy()

    # Save to working directory
    temp_train_path = os.path.join(config.WORKING_DIR, "train_subset.csv")
    temp_val_path = os.path.join(config.WORKING_DIR, "val_subset.csv")
    temp_test_path = os.path.join(config.WORKING_DIR, "test_subset.csv")

    train_subset.to_csv(temp_train_path, index=False)
    val_subset.to_csv(temp_val_path, index=False)
    test_subset.to_csv(temp_test_path, index=False)

    print(f"Created subsets: {subset_size} samples each.")

    # 3. Override Config Paths and Hyperparameters
    # We modify the module variables directly.
    config.TRAIN_CSV = temp_train_path
    config.VAL_CSV = temp_val_path
    config.TEST_CSV = temp_test_path

    config.NUM_EPOCHS = 1
    config.BATCH_SIZE = 8
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny datasets

    # Force device to CPU if GPU is not critical for this tiny test,
    # but use GPU if available for realism.
    config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("Configuration updated for speed.")


def demonstrate_dataset():
    """
    Demonstrates loading data using the library.dataset module.
    """
    print("\n>>> Demonstrating Dataset Loading...")

    # Force reload to pick up the new subset CSVs
    # We pass load_cached_data=False to ensure we don't use old weights from a full run
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        load_cached_data=False
    )

    print(f"Train Loader batches: {len(train_loader)}")
    print(f"Val Loader batches:   {len(val_loader)}")
    print(f"Test Loader batches:  {len(test_loader)}")

    # Verify Train Batch
    images, labels = next(iter(train_loader))
    print(f"Train Batch - Image Shape: {images.shape}, Label Shape: {labels.shape}")

    # Assertions
    assert images.shape[0] == config.BATCH_SIZE, "Batch size mismatch"
    assert images.shape[1] == 3, "Image channel mismatch (should be RGB)"
    assert (
        images.shape[2] == config.IMG_SIZE and images.shape[3] == config.IMG_SIZE
    ), "Image resolution mismatch"
    assert isinstance(labels, torch.Tensor), "Labels should be a tensor"

    # Verify Test Batch (returns image and ID)
    test_images, test_ids = next(iter(test_loader))
    print(f"Test Batch - Image Shape: {test_images.shape}, ID Shape: {test_ids.shape}")
    assert test_images.shape[0] == config.BATCH_SIZE

    return train_loader, val_loader, test_loader


def demonstrate_model_architecture():
    """
    Demonstrates model instantiation and a forward pass.
    """
    print("\n>>> Demonstrating Model Architecture...")

    # Instantiate model
    model = model_lib.ResNet18Classifier(
        num_classes=config.NUM_CLASSES, pretrained=False
    )
    model.to(config.DEVICE)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE).to(config.DEVICE)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Assertions
    assert output.shape == (
        2,
        config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected (2, {config.NUM_CLASSES})"
    print("Model forward pass successful.")


def demonstrate_training_pipeline():
    """
    Demonstrates the full training loop using library.trainer.
    """
    print("\n>>> Demonstrating Training Pipeline...")

    # Run training
    # This will train for 1 epoch on the subset data, validate, and save the model.
    # It also generates a submission at the end.
    trainer.run_training(
        load_cached_data=False,  # Recompute weights for subset
        epochs=1,
        batch_size=config.BATCH_SIZE,
        patience=1,
    )

    # Verify Artifacts
    assert os.path.exists(config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not generated."

    print("Training pipeline completed successfully.")


def demonstrate_inference_module():
    """
    Demonstrates the inference module separately.
    """
    print("\n>>> Demonstrating Inference Module...")

    # Delete previous submission to ensure inference creates a new one
    if os.path.exists(config.SUBMISSION_PATH):
        os.remove(config.SUBMISSION_PATH)

    # Run inference
    inference.run_inference(
        checkpoint_path=config.MODEL_SAVE_PATH,
        batch_size=config.BATCH_SIZE,
        device=config.DEVICE,
        num_workers=0,
    )

    # Verify
    assert os.path.exists(
        config.SUBMISSION_PATH
    ), "Inference failed to generate submission."

    # Check content format
    df = pd.read_csv(config.SUBMISSION_PATH)
    print("Submission Head:")
    print(df.head())

    assert (
        "Id" in df.columns and "Predicted" in df.columns
    ), "Submission columns missing."
    assert len(df) > 0, "Submission file is empty."

    print("Inference module demonstration successful.")


if __name__ == "__main__":
    # 1. Setup
    setup_environment_and_data()

    # 2. Dataset Check
    demonstrate_dataset()

    # 3. Model Check
    demonstrate_model_architecture()

    # 4. Training Loop Check
    demonstrate_training_pipeline()

    # 5. Inference Check
    demonstrate_inference_module()

    print("\n>>> All demonstrations passed successfully!")
