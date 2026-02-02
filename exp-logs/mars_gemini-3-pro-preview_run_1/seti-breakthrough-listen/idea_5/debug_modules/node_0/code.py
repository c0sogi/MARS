import os
import sys
import torch
import pandas as pd
import numpy as np

# Import provided library modules
from library import config, utils, data, model, train


def main():
    print("Starting demonstration of SETI Technosignature Detection library...")

    # 1. Setup and Configuration Override for Speed
    # We override config constants to run a fast debug pass
    print("Configuring environment for fast execution...")
    utils.seed_everything(42)

    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 12  # Small sample size for speed
    config.BATCH_SIZE = 4  # Small batch size
    config.EPOCHS = 1  # Single epoch
    config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution
    config.WORKING_DIR = "./working/demo_test"  # Separate dir for this demo
    config.SUBMISSION_DIR = "./working/demo_submission"
    config.SUBMISSION_PATH = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # 2. Verify Data Loading
    print("\n--- Verifying Data Pipeline ---")

    # Test Dataset instantiation
    # Note: We assume metadata files exist as per prompt description
    dataset = data.SETIDataset(
        metadata_path=config.TRAIN_METADATA,
        mode="train",
        debug=True,
        debug_sample_size=config.DEBUG_SAMPLE_SIZE,
    )

    print(f"Dataset length (Debug): {len(dataset)}")
    assert (
        len(dataset) == config.DEBUG_SAMPLE_SIZE
    ), "Dataset length mismatch with debug size"

    # Test single item retrieval
    sample_input, sample_target = dataset[0]
    print(f"Sample Input Shape: {sample_input.shape}")
    print(f"Sample Target: {sample_target}")

    # Assertions for shape: (6 frames, 1 channel, 273 height, 256 width)
    expected_shape = (6, 1, 273, 256)
    assert (
        sample_input.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {sample_input.shape}"
    assert isinstance(sample_target.item(), float), "Target should be a float scalar"

    # Test DataLoader
    train_loader = data.get_train_dataloader(
        batch_size=config.BATCH_SIZE, debug=True, num_workers=0
    )
    batch_inputs, batch_targets = next(iter(train_loader))

    print(f"Batch Input Shape: {batch_inputs.shape}")
    print(f"Batch Target Shape: {batch_targets.shape}")

    # Assertions for batch
    assert batch_inputs.shape == (
        config.BATCH_SIZE,
        *expected_shape,
    ), "Batch input shape mismatch"
    assert batch_targets.shape == (config.BATCH_SIZE,), "Batch target shape mismatch"

    # 3. Verify Model Architecture
    print("\n--- Verifying Model Architecture ---")

    # Initialize model
    net = model.TimeDistributedResNet50GN()
    net.eval()  # Set to eval for deterministic check (though we just check shape)

    # Forward pass with the batch fetched earlier
    with torch.no_grad():
        outputs = net(batch_inputs)

    print(f"Model Output Shape: {outputs.shape}")

    # Assertions for model output: (Batch_Size, 1)
    assert outputs.shape == (
        config.BATCH_SIZE,
        1,
    ), f"Expected output shape {(config.BATCH_SIZE, 1)}, got {outputs.shape}"

    # 4. Verify Training Loop
    print("\n--- Verifying Training Loop ---")

    # We call the high-level train_model function from library.train
    # This will run for 1 epoch on the small debug dataset
    best_model_path = train.train_model(debug=True)

    print(f"Training finished. Model saved at: {best_model_path}")

    # Assert model file was created
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Expected model file at {best_model_path} was not found."
        )

    # 5. Verify Inference and Submission
    print("\n--- Verifying Inference and Submission ---")

    # Generate submission using the trained model
    train.generate_submission(best_model_path, debug=True)

    print(f"Submission generated at: {config.SUBMISSION_PATH}")

    # Assert submission file exists
    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Expected submission file at {config.SUBMISSION_PATH} was not found."
        )

    # Verify submission content format
    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print("Submission Head:")
    print(df_sub.head())

    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert "target" in df_sub.columns, "Submission missing 'target' column"
    assert len(df_sub) > 0, "Submission file is empty"

    # Check that predictions are probabilities (or logits depending on model output processing)
    # The generate_submission function applies sigmoid, so they should be 0-1
    preds = df_sub["target"].values
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions are not valid probabilities (0-1)"

    print("\nAll demonstrations and verifications passed successfully.")


if __name__ == "__main__":
    main()
