import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from library import config, utils, data_loader, model_factory, engine, inference


def main():
    print("Initializing demonstration...")

    # 1. Configuration Overrides for Speed
    # We modify the global config object to run a fast demonstration
    config.DEBUG = True
    config.SUBSET_SIZE = 64  # Small subset for quick execution
    config.BATCH_SIZE = 16
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Reduce epochs for the 'resnet' model to 1 for demonstration
    config.MODEL_SPECS["resnet"]["epochs"] = 1

    # Ensure reproducibility
    utils.set_seed(config.SEED)
    device = config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("\n--- Data Loading ---")
    # We use the 'resnet' key to get transforms matching the resnet spec (img_size=256)
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        model_key="resnet",
        load_cached_data=False,  # Force reload to demonstrate reading from CSV
    )

    # Validation: Check if loaders are populated
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    assert len(train_loader) > 0, "Train loader is empty!"

    # Fetch one batch to verify shapes
    images, labels = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")  # Should be [16, 3, 256, 256]
    print(f"Batch Label Shape: {labels.shape}")  # Should be [16]

    assert images.shape == (config.BATCH_SIZE, 3, 256, 256)

    # 3. Model Creation
    print("\n--- Model Creation ---")
    model = model_factory.create_model("resnet", pretrained=True)
    model = model.to(device)

    # Validation: Check output shape
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, 1), "Model output shape mismatch! Expected (Batch, 1)"

    # 4. Training
    print("\n--- Training Loop ---")
    # Setup optimizer as per specs (simplified for demo)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.MODEL_SPECS["resnet"]["learning_rate"],
        weight_decay=config.MODEL_SPECS["resnet"]["weight_decay"],
    )

    # Scheduler (optional, using StepLR for demo)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Run training for 1 epoch
    trained_model = engine.fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=config.MODEL_SPECS["resnet"]["epochs"],
        scheduler=scheduler,
        checkpoint_name="resnet_demo_checkpoint.pth",
    )

    # Validation: Check if checkpoint exists
    checkpoint_path = os.path.join(config.WORKING_DIR, "resnet_demo_checkpoint.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created!"
    print("Training complete. Checkpoint verified.")

    # 5. Inference
    print("\n--- Inference ---")
    # Generate predictions using TTA
    predictions = inference.predict_with_tta(trained_model, test_loader, device)

    # Validation: Check predictions
    print(f"Number of predictions: {len(predictions)}")
    assert len(predictions) > 0, "No predictions generated!"

    # Check values are probabilities
    first_pred = list(predictions.values())[0]
    assert 0.0 <= first_pred <= 1.0, f"Prediction out of range [0, 1]: {first_pred}"

    # 6. Submission Generation
    print("\n--- Submission ---")
    submission_filename = "demo_submission.csv"
    inference.generate_submission(
        predictions, output_dir=config.WORKING_DIR, filename=submission_filename
    )

    submission_path = os.path.join(config.WORKING_DIR, submission_filename)
    assert os.path.exists(submission_path), "Submission file not found!"

    # Read back submission to verify format
    df_sub = pd.read_csv(submission_path)
    print(f"Submission head:\n{df_sub.head()}")
    assert list(df_sub.columns) == ["id", "label"], "Submission columns mismatch!"
    assert len(df_sub) == len(predictions), "Submission row count mismatch!"

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
