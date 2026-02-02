import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Import provided library components
from library.data_utils import load_and_preprocess_data
from library.model_arch import DualViewDCNResNet
from library.train_eval import train_one_epoch, evaluate

# ------------------------------------------------------------------------------
# Configuration & Setup
# ------------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 1024  # Large batch size for speed on A100
DEMO_SUBSET_SIZE = 5000  # Number of samples to use for training demo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print(f"Starting demonstration on device: {DEVICE}")
    set_seeds(SEED)

    # --------------------------------------------------------------------------
    # 1. Data Loading & Preprocessing
    # --------------------------------------------------------------------------
    print("\n[1/4] Loading and preprocessing data...")
    # We allow loading from cache if available to speed up repeated runs
    train_dataset, val_dataset, test_dataset, input_dim, num_classes, test_ids = (
        load_and_preprocess_data(load_cached_data=True)
    )

    # Verification
    print(f"      Input Dimensions: {input_dim}")
    print(f"      Number of Classes: {num_classes}")
    print(f"      Train Dataset Size: {len(train_dataset)}")

    assert input_dim > 0, "Input dimension must be positive."
    assert num_classes == 7, f"Expected 7 classes, got {num_classes}."
    assert len(train_dataset) > 0, "Training dataset is empty."
    assert len(test_ids) == len(
        test_dataset
    ), "Mismatch between test IDs and test dataset size."

    # --------------------------------------------------------------------------
    # 2. Model Instantiation & Architecture Check
    # --------------------------------------------------------------------------
    print("\n[2/4] Initializing model and verifying architecture...")
    model = DualViewDCNResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        dcn_rank=4,
        dcn_layers=2,  # Reduced for demo speed
        resnet_blocks=2,  # Reduced for demo speed
        resnet_dim=256,  # Reduced for demo speed
        dropout_rate=0.2,
    ).to(DEVICE)

    # Dummy forward pass to verify shape
    dummy_input = torch.randn(2, input_dim).to(DEVICE)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    assert dummy_output.shape == (
        2,
        num_classes,
    ), f"Model output shape mismatch. Expected (2, {num_classes}), got {dummy_output.shape}"
    print("      Model architecture verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Training Loop Demonstration (Subset)
    # --------------------------------------------------------------------------
    print(
        f"\n[3/4] Running training demonstration (Subset: {DEMO_SUBSET_SIZE} samples)..."
    )

    # Create subsets for rapid demonstration
    train_subset_indices = np.random.choice(
        len(train_dataset), DEMO_SUBSET_SIZE, replace=False
    )
    val_subset_indices = np.random.choice(
        len(val_dataset), DEMO_SUBSET_SIZE // 5, replace=False
    )

    train_subset = Subset(train_dataset, train_subset_indices)
    val_subset = Subset(val_dataset, val_subset_indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    # Setup Optimizer and Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    # Run 1 Epoch
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE
    )
    val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

    print(f"      Epoch 1 Summary:")
    print(f"      Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"      Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

    # Assertions to ensure learning mechanics are working
    assert not np.isnan(train_loss), "Training loss is NaN."
    assert 0 <= train_acc <= 1, "Training accuracy out of bounds."
    assert 0 <= val_acc <= 1, "Validation accuracy out of bounds."

    # --------------------------------------------------------------------------
    # 4. Inference & Submission Generation
    # --------------------------------------------------------------------------
    print("\n[4/4] Generating submission for test set...")

    test_loader = DataLoader(
        test_dataset, batch_size=4096, shuffle=False, num_workers=2
    )
    model.eval()

    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.append(preds.cpu().numpy())

    predictions = np.concatenate(predictions)

    # Map 0-6 back to 1-7 (Cover_Type classes)
    final_predictions = predictions + 1

    # Create submission DataFrame
    submission_df = pd.DataFrame({"Id": test_ids, "Cover_Type": final_predictions})

    # Verify submission shape and content
    assert (
        len(submission_df) == 400000
    ), f"Submission rows mismatch. Expected 400000, got {len(submission_df)}"
    assert submission_df["Cover_Type"].min() >= 1, "Invalid class label < 1 found."
    assert submission_df["Cover_Type"].max() <= 7, "Invalid class label > 7 found."

    # Save
    submission_path = "./submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"      Submission saved to {submission_path}")
    print(f"      Head:\n{submission_df.head()}")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
