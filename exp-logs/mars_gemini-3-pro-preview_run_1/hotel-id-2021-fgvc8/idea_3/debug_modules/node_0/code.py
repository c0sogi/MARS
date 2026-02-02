import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import process_data, HotelDataset, get_transforms
from library.model import HotelRecognitionModel
from library.engine import train_one_epoch, validate, generate_submission

if __name__ == "__main__":
    # 1. Setup and Configuration Overrides for Speed
    print("Initializing demonstration...")

    # Override Config for a fast demonstration run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Extremely small subset for speed
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2
    Config.EPOCHS = 1

    # Ensure reproducibility
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Data Processing
    print("Processing data...")
    # load_cached_data=False ensures we test the processing logic and don't pick up old files
    train_df, val_df, test_df, class_map = process_data(
        load_cached_data=False, debug=True
    )

    # Verify DataFrames
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Number of classes: {len(class_map)}")

    assert len(train_df) == Config.DEBUG_SAMPLE_SIZE, "Train DF size mismatch"
    assert len(val_df) == Config.DEBUG_SAMPLE_SIZE, "Val DF size mismatch"
    assert len(test_df) == Config.DEBUG_SAMPLE_SIZE, "Test DF size mismatch"
    assert "label" in train_df.columns, "Label column missing in train_df"

    # 3. Dataset and Dataloader
    print("Creating datasets and dataloaders...")
    train_dataset = HotelDataset(train_df, transform=get_transforms("train"))
    val_dataset = HotelDataset(val_df, transform=get_transforms("valid"))
    test_dataset = HotelDataset(test_df, transform=get_transforms("test"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Model Initialization
    print("Initializing model...")
    # Note: We use len(class_map) to ensure the head matches the actual data,
    # though usually this matches Config.NUM_CLASSES
    model = HotelRecognitionModel(n_classes=len(class_map), pretrained=True)
    model.to(device)

    # Verify Model Output Shape
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    # ArcFace forward pass requires labels during training
    dummy_labels = torch.tensor([0, 1]).to(device)

    # Check training forward pass (returns logits scaled by s)
    output_train = model(dummy_input, dummy_labels)
    assert output_train.shape == (
        2,
        len(class_map),
    ), f"Model output shape mismatch: {output_train.shape}"

    # Check inference forward pass (returns cosine similarities)
    output_infer = model(dummy_input, label=None)
    assert output_infer.shape == (
        2,
        len(class_map),
    ), f"Inference output shape mismatch: {output_infer.shape}"

    # 5. Training Loop Demonstration
    print("Running training loop (1 epoch)...")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)

    avg_loss = train_one_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epoch=1,
    )

    print(f"Training finished. Average Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Training loss is NaN"
    assert avg_loss > 0, "Training loss should be positive"

    # 6. Validation Demonstration
    print("Running validation...")
    val_loss, val_map5 = validate(
        model=model, dataloader=val_loader, criterion=criterion, device=device
    )

    print(f"Validation finished. Loss: {val_loss:.4f}, MAP@5: {val_map5:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0.0 <= val_map5 <= 1.0, "MAP@5 score out of range"

    # 7. Submission Generation
    print("Generating submission...")
    output_csv = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    generate_submission(
        model=model,
        dataloader=test_loader,
        test_df=test_df,
        class_map=class_map,
        device=device,
        output_path=output_csv,
    )

    # Verify Submission File
    assert os.path.exists(output_csv), "Submission file was not created"

    sub_df = pd.read_csv(output_csv)
    print(f"Submission loaded. Rows: {len(sub_df)}")
    print(sub_df.head())

    assert len(sub_df) == len(test_df), "Submission row count mismatch"
    assert (
        "image" in sub_df.columns and "hotel_id" in sub_df.columns
    ), "Submission columns incorrect"

    # Check format of prediction string (space delimited)
    example_pred = sub_df.iloc[0]["hotel_id"]
    assert isinstance(example_pred, str), "Prediction is not a string"
    assert len(example_pred.split(" ")) == 5, "Prediction does not contain 5 hotel IDs"

    print("\nDemonstration completed successfully.")
