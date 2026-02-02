import os
import sys
import torch
import pandas as pd
import numpy as np

# Import library modules
from library.config import Config
from library import utils, data, model, trainer, inference


def main():
    print("==== Apple Disease Detection Demo ====")

    # 1. Configuration Setup
    # Modify Config for a fast demonstration run
    print("\n[1] Configuring environment...")
    Config.DEBUG = True  # Use subset of data
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size

    # Set random seeds for reproducibility
    utils.set_seed(Config.SEED)
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Device: {utils.get_device()}")

    # 2. Data Loading Verification
    print("\n[2] Verifying Data Loading...")
    # Explicitly pass debug=True to ensure subsets are used
    train_loader, val_loader, test_loader = data.get_dataloaders(debug=True)

    # Fetch a single batch to verify shapes
    images, labels = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    # Assertions
    assert images.dim() == 4, "Images tensor must be 4-dimensional (B, C, H, W)"
    assert images.shape[1] == 3, "Images must have 3 color channels"
    assert images.shape[2] == Config.IMG_SIZE, f"Image height must be {Config.IMG_SIZE}"
    assert labels.dim() == 1, "Labels tensor must be 1-dimensional"
    assert (
        labels.shape[0] == images.shape[0]
    ), "Batch size mismatch between images and labels"
    print("Data loading logic verified.")

    # 3. Model Architecture Verification
    print("\n[3] Verifying Model Architecture...")
    # Instantiate model with pretrained=False to avoid downloading weights during this quick check
    net = model.AppleDiseaseModel(pretrained=False, num_classes=Config.NUM_CLASSES)
    net.to(utils.get_device())
    net.eval()

    # Perform a forward pass
    with torch.no_grad():
        sample_input = images.to(utils.get_device())
        logits = net(sample_input)

    print(f"Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        images.shape[0],
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected {(images.shape[0], Config.NUM_CLASSES)}, got {logits.shape}"
    print("Model forward pass verified.")

    # 4. Training Loop Verification
    print("\n[4] Running Training Loop (Fast Mode)...")
    # Initialize Trainer with debug=True
    t = trainer.Trainer(debug=True)

    # Run the training loop (1 epoch as configured above)
    t.fit()

    # Verify that the best model checkpoint was saved
    assert os.path.exists(
        t.best_model_path
    ), f"Model checkpoint not found at {t.best_model_path}"
    print("Training loop completed successfully.")

    # 5. Inference and Submission Verification
    print("\n[5] Running Inference and Generating Submission...")
    # Run the inference pipeline
    inference.predict_and_submit(debug=True)

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Validate submission content format
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {submission_df.shape}")

    expected_columns = ["image_id"] + Config.CLASSES
    assert (
        list(submission_df.columns) == expected_columns
    ), f"Column mismatch. Expected {expected_columns}, got {list(submission_df.columns)}"

    assert len(submission_df) > 0, "Submission file is empty"

    # Check if probabilities sum roughly to 1 (softmax check)
    # Taking the first row's probabilities
    first_row_probs = submission_df.iloc[0][Config.CLASSES].values.astype(float)
    prob_sum = np.sum(first_row_probs)
    assert np.isclose(
        prob_sum, 1.0, atol=1e-5
    ), f"Probabilities do not sum to 1: {prob_sum}"

    print("Inference and submission generation verified.")
    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    main()
