import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_logger, save_checkpoint
from library.transforms import get_transforms, get_mixup_fn
from library.dataset import CassavaDataset, load_dataset_dataframe
from library.model import get_model
from library.engine import train_one_epoch, validate


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides for Demo
    # -------------------------------------------------------------------------
    print("Initializing demonstration...")
    seed_everything(Config.seed)

    # Override Config for speed and demonstration purposes
    # We use a smaller model and smaller image size to ensure quick execution on CPU/GPU
    Config.model_name = "resnet18"
    Config.phase1_batch_size = 8
    Config.phase1_image_size = 224
    Config.num_workers = 2  # Reduce workers for small debug data

    device = torch.device(Config.device)
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Data Loading Demo ---")

    # Load a small subset of training data (Debug mode)
    df_train = load_dataset_dataframe(
        Config.train_metadata_path, debug=True, debug_size=32
    )
    df_val = load_dataset_dataframe(Config.val_metadata_path, debug=True, debug_size=16)

    print(
        f"Loaded {len(df_train)} training samples and {len(df_val)} validation samples."
    )

    # Get Transforms
    train_transforms = get_transforms(
        stage="train", image_size=Config.phase1_image_size
    )
    val_transforms = get_transforms(stage="valid", image_size=Config.phase1_image_size)

    # Create Datasets
    train_dataset = CassavaDataset(
        df_train, transform=train_transforms, output_label=True
    )
    val_dataset = CassavaDataset(df_val, transform=val_transforms, output_label=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.phase1_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.phase1_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Verify Data Loading Logic
    sample_batch = next(iter(train_loader))
    images, labels = sample_batch

    # Assertions
    assert len(images) == Config.phase1_batch_size, "Batch size mismatch"
    assert images.shape[1] == 3, "Image channel count mismatch (should be 3 for RGB)"
    assert images.shape[2] == Config.phase1_image_size, "Image height mismatch"
    assert images.shape[3] == Config.phase1_image_size, "Image width mismatch"
    assert isinstance(labels, torch.Tensor), "Labels should be a tensor"
    print("Data loading verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Model Initialization Demo ---")

    # Create model with EMA
    model, model_ema = get_model(
        model_name=Config.model_name,
        num_classes=Config.num_classes,
        pretrained=True,
        use_ema=True,
    )

    # Assertions
    assert isinstance(model, nn.Module), "Model is not a PyTorch Module"
    if Config.use_ema:
        assert (
            model_ema is not None
        ), "Model EMA was not initialized despite use_ema=True"
        print("Model EMA initialized successfully.")

    # Check forward pass shape
    dummy_input = torch.randn(
        2, 3, Config.phase1_image_size, Config.phase1_image_size
    ).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (
        2,
        Config.num_classes,
    ), f"Output shape mismatch. Expected (2, {Config.num_classes}), got {output.shape}"
    print("Model forward pass verification passed.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Training Loop Demo ---")

    # Setup components
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    # Get Mixup function
    phase_config = Config.get_phase_config(1)
    mixup_fn = get_mixup_fn(phase_config)

    # Run one epoch
    # Note: train_one_epoch handles moving data to device
    avg_loss = train_one_epoch(
        model=model,
        optimizer=optimizer,
        data_loader=train_loader,
        device=device,
        epoch=0,
        loss_fn=loss_fn,
        max_norm=10.0,
        model_ema=model_ema,
        mixup_fn=mixup_fn,
        accum_iter=1,
    )

    print(f"Training Epoch 0 complete. Average Loss: {avg_loss:.4f}")
    assert isinstance(avg_loss, float), "train_one_epoch should return a float loss"
    assert avg_loss > 0, "Loss should be positive"

    # -------------------------------------------------------------------------
    # 5. Validation Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Validation Loop Demo ---")

    # We use the EMA model for validation if available, as per standard practice
    eval_model = model_ema.module if model_ema else model

    metrics = validate(
        model=eval_model, data_loader=val_loader, loss_fn=loss_fn, device=device
    )

    print(f"Validation Metrics: {metrics}")
    assert "loss" in metrics, "Validation metrics missing 'loss'"
    assert "accuracy" in metrics, "Validation metrics missing 'accuracy'"
    assert 0 <= metrics["accuracy"] <= 100, "Accuracy should be between 0 and 100"

    # -------------------------------------------------------------------------
    # 6. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Inference Demo ---")

    # Load test metadata
    # In the provided environment, test.csv is derived from sample_submission.csv
    df_test = load_dataset_dataframe(
        Config.test_metadata_path, debug=True, debug_size=10
    )

    # Test Transforms (same as validation)
    test_transforms = get_transforms(stage="test", image_size=Config.phase1_image_size)

    # Test Dataset (output_label=False usually, but here we keep structure consistent)
    # The dataset class handles returning (image, label), but for inference we ignore label
    test_dataset = CassavaDataset(
        df_test, transform=test_transforms, output_label=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.phase1_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    eval_model.eval()
    predictions = []
    image_ids = df_test["image_id"].tolist()

    print("Running inference...")
    with torch.no_grad():
        for i, images in enumerate(test_loader):
            images = images.to(device)
            outputs = eval_model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)

    # Verify predictions
    assert len(predictions) == len(
        df_test
    ), "Number of predictions does not match number of test samples"

    # Create submission dataframe
    submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})

    print("Sample Submission:")
    print(submission_df.head())

    # Verify submission format
    assert "image_id" in submission_df.columns
    assert "label" in submission_df.columns
    assert submission_df["label"].dtype in [
        int,
        "int64",
        "int32",
    ], "Label column must be integer"

    print("\nAll demonstrations and verifications passed successfully.")


if __name__ == "__main__":
    main()
