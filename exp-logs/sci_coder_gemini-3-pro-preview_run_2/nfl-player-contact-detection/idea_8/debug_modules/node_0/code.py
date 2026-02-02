import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import library modules
from library import config, utils, data_processing, dataset, model, train, inference


def run_demo():
    print("=== Starting Demonstration of NFL Contact Detection Library ===")

    # 1. Configuration Overrides for Demo Speed
    # We modify the config module directly to affect all downstream imports
    print("\n[1] Configuring environment for fast demonstration...")
    config.WORKING_DIR = "./working/demo_execution"
    config.SUBMISSION_DIR = "./working/demo_submission"

    # Clean up previous demo runs if they exist to ensure a fresh start
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    if os.path.exists(config.SUBMISSION_DIR):
        shutil.rmtree(config.SUBMISSION_DIR)

    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Reduce training intensity for the demo
    config.EPOCHS = 2
    config.BATCH_SIZE = 64  # Small batch for debug data
    config.HIDDEN_DIM = 32  # Smaller model for speed
    config.NUM_HEADS = 2

    # Ensure reproducibility
    utils.set_seed(42)
    print("Configuration updated: EPOCHS=2, BATCH_SIZE=64, HIDDEN_DIM=32")

    # 2. Data Processing Demonstration
    print("\n[2] Demonstrating Data Processing (FeatureEngineer)...")
    engineer = data_processing.FeatureEngineer()

    # Process training data in debug mode (subset of first 5 games)
    print("Processing training data (debug=True)...")
    X_train, y_train, ids_train = engineer.process_dataset(
        split="train", load_cached_data=False, debug=True
    )

    # Assertions for Data Processing
    num_features = len(config.INPUT_FEATURES)
    window_size = config.WINDOW_SIZE
    expected_width = window_size * num_features

    print(f"Train Data Shape: {X_train.shape}")
    print(f"Train Labels Shape: {y_train.shape}")

    assert X_train.ndim == 2, "X_train should be 2D (N, Features)"
    assert (
        X_train.shape[1] == expected_width
    ), f"Expected width {expected_width}, got {X_train.shape[1]}"
    assert len(X_train) == len(y_train), "Mismatch between features and labels count"
    assert not np.isnan(X_train).any(), "X_train contains NaNs"

    # Process validation data
    print("Processing validation data (debug=True)...")
    X_val, y_val, ids_val = engineer.process_dataset(
        split="validation", load_cached_data=False, debug=True
    )
    assert X_val.shape[1] == expected_width
    print("Data Processing verification passed.")

    # 3. Dataset Demonstration
    print("\n[3] Demonstrating Dataset (ContactSequenceDataset)...")
    train_dataset = dataset.ContactSequenceDataset(X_train, y_train)

    # Fetch a single sample
    sample_inputs, sample_label = train_dataset[0]
    sequence, center_features = sample_inputs

    print(f"Sequence Shape: {sequence.shape}")
    print(f"Center Features Shape: {center_features.shape}")
    print(f"Label: {sample_label}")

    # Assertions for Dataset
    # Sequence should be (Window, Features)
    assert sequence.shape == (
        window_size,
        num_features,
    ), f"Incorrect sequence shape: {sequence.shape}"
    # Center features should be (Features,)
    assert center_features.shape == (
        num_features,
    ), f"Incorrect center features shape: {center_features.shape}"
    assert torch.is_tensor(sample_label), "Label should be a tensor"

    # Check that center features match the middle of the sequence (skip connection logic)
    center_idx = window_size // 2
    assert torch.allclose(
        sequence[center_idx], center_features
    ), "Center features do not match sequence center"
    print("Dataset verification passed.")

    # 4. Model Demonstration
    print("\n[4] Demonstrating Model (KCAN)...")
    device = config.get_device()
    net = model.KCAN().to(device)

    # Create a dummy batch of size 2
    batch_sequence = torch.stack([sequence, sequence]).to(device)  # (2, Window, Feats)
    batch_center = torch.stack([center_features, center_features]).to(
        device
    )  # (2, Feats)

    # Forward pass
    net.eval()
    with torch.no_grad():
        logits = net((batch_sequence, batch_center))

    print(f"Logits Shape: {logits.shape}")

    # Assertions for Model
    assert logits.shape == (
        2,
        1,
    ), f"Output shape should be (Batch, 1), got {logits.shape}"
    print("Model verification passed.")

    # 5. Training Loop Demonstration
    print("\n[5] Demonstrating Training Loop...")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dataset = dataset.ContactSequenceDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    trainer = train.Trainer(net, train_loader, val_loader, device)

    print("Starting short training run (2 epochs)...")
    best_mcc = trainer.fit(epochs=config.EPOCHS, patience=1)

    print(f"Training finished. Best MCC: {best_mcc}")
    assert os.path.exists(trainer.best_model_path), "Best model file was not saved"

    # Manually save a threshold for the inference step (normally handled by run_training)
    # We calculate it on the validation set using the best model
    print("Optimizing and saving threshold for inference...")
    net.load_state_dict(torch.load(trainer.best_model_path, map_location=device))
    net.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            seq, cent = inputs
            logits = net((seq.to(device), cent.to(device)))
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(targets.numpy())

    all_probs = np.concatenate(all_probs).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    best_thresh, _ = trainer.optimize_threshold(all_targets, all_probs)
    np.save(
        os.path.join(config.WORKING_DIR, "best_threshold.npy"), np.array([best_thresh])
    )
    print(f"Saved threshold: {best_thresh}")

    # 6. Inference Demonstration
    print("\n[6] Demonstrating Inference Pipeline...")
    inf_pipeline = inference.InferencePipeline()

    # Run inference on test data (debug mode)
    # This will load the model and threshold we just saved
    print("Running inference pipeline...")
    inf_pipeline.run(debug=True, load_cached=False)

    submission_file = inf_pipeline.submission_path
    assert os.path.exists(submission_file), "Submission file not created"

    # Verify submission content
    df_sub = pd.read_csv(submission_file)
    print(f"Submission Head:\n{df_sub.head()}")

    assert "contact_id" in df_sub.columns
    assert "contact" in df_sub.columns
    assert df_sub["contact"].isin([0, 1]).all(), "Predictions must be binary"

    # Check length matches the debug test set size
    # We can get the expected size from the FeatureEngineer
    test_engineer = data_processing.FeatureEngineer()
    _, _, test_ids = test_engineer.process_dataset(
        split="test", load_cached_data=True, debug=True
    )
    assert len(df_sub) == len(
        test_ids
    ), f"Submission length mismatch. Expected {len(test_ids)}, got {len(df_sub)}"

    print("Inference verification passed.")
    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    run_demo()
