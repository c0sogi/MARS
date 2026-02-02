import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Import library modules
from library.config import SEED, DEVICE, NUM_CLASSES, INPUT_SIZE, seed_everything
from library.dataset import (
    CdiscountDataset,
    get_transforms,
    train_collate_fn,
    eval_collate_fn,
)
from library.model import get_model
from library.engine import train_model, make_predictions


def main():
    print("==== Cdiscount Classification Pipeline Demo ====")

    # 1. Setup Environment
    seed_everything(SEED)
    print(f"Device set to: {DEVICE}")

    # Define a small debug size for rapid execution
    DEBUG_SIZE = 50
    BATCH_SIZE = 8

    # 2. Dataset Instantiation and Verification
    print("\n[1/5] Initializing Datasets...")

    # Train Dataset
    train_dataset = CdiscountDataset(
        mode="train", transform=get_transforms("train"), debug_size=DEBUG_SIZE
    )

    # Validation Dataset
    val_dataset = CdiscountDataset(
        mode="val", transform=get_transforms("val"), debug_size=DEBUG_SIZE
    )

    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size: {len(val_dataset)}")

    # Verify __getitem__ structure
    sample_imgs, sample_target, sample_pid = train_dataset[0]

    # Assertions for Dataset
    assert len(train_dataset) == DEBUG_SIZE, "Dataset size does not match debug_size."
    assert isinstance(sample_imgs, list), "Dataset should return a list of images."
    assert len(sample_imgs) > 0, "Product must have at least one image."
    assert isinstance(
        sample_imgs[0], torch.Tensor
    ), "Images should be converted to Tensors."
    assert sample_imgs[0].shape == (
        3,
        INPUT_SIZE,
        INPUT_SIZE,
    ), f"Image shape mismatch. Expected (3, {INPUT_SIZE}, {INPUT_SIZE}), got {sample_imgs[0].shape}"
    assert isinstance(sample_target, int), "Target should be an integer."

    print("Dataset verification passed.")

    # 3. DataLoader Verification
    print("\n[2/5] Initializing DataLoaders...")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=0,  # 0 workers for simple debug script
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=eval_collate_fn,
        num_workers=0,
    )

    # Verify Train Batch
    images_batch, targets_batch = next(iter(train_loader))
    # Note: Dimension 0 is Sum(num_images_per_product), not necessarily BATCH_SIZE
    assert images_batch.dim() == 4, "Batch images should be 4D tensor."
    assert targets_batch.dim() == 1, "Batch targets should be 1D tensor."
    assert images_batch.shape[1] == 3, "Batch images should have 3 channels."

    # Verify Val Batch
    v_imgs, v_targets, v_pids, v_nums = next(iter(val_loader))
    assert (
        v_imgs.shape[0] == v_nums.sum().item()
    ), "Validation batch flattening mismatch."

    print("DataLoader verification passed.")

    # 4. Model Initialization
    print("\n[3/5] Initializing Model...")
    # Use pretrained=False to avoid downloading weights during demo
    model = get_model(pretrained=False)
    model = model.to(DEVICE)

    # Verify Output Shape
    dummy_input = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        2,
        NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (2, {NUM_CLASSES}), got {output.shape}"

    print("Model verification passed.")

    # 5. Training Loop Integration
    print("\n[4/5] Running Training Loop (1 Epoch)...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Run training using the engine
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=1,
        device=DEVICE,
        patience=1,
    )

    print("Training loop execution successful.")

    # 6. Prediction Generation
    print("\n[5/5] Generating Predictions...")

    # Setup Test Loader
    test_dataset = CdiscountDataset(
        mode="test", transform=get_transforms("test"), debug_size=20
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=eval_collate_fn,
        num_workers=0,
    )

    output_csv = "./working/demo_submission.csv"
    if os.path.exists(output_csv):
        os.remove(output_csv)

    make_predictions(test_loader, trained_model, device=DEVICE, output_file=output_csv)

    # Verify Submission File
    assert os.path.exists(output_csv), "Submission file was not created."

    df_sub = pd.read_csv(output_csv)
    print(f"Submission file loaded. Rows: {len(df_sub)}")
    print(df_sub.head())

    assert len(df_sub) == 20, "Submission file row count does not match test set size."
    assert list(df_sub.columns) == [
        "_id",
        "category_id",
    ], "Submission file columns mismatch."

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    main()
