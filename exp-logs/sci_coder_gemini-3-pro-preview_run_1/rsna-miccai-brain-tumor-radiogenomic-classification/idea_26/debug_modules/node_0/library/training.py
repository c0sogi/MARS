import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from library.utils import get_device, save_checkpoint, load_checkpoint, set_seed
from library.dataset import get_datasets
from library.model import (
    SIA_DS_EfficientNet,
    train_one_epoch,
    validate,
    generate_submission,
)


def run_fold(
    model,
    train_loader,
    val_loader,
    num_epochs=20,
    patience=5,
    save_path="./working/best_model.pth",
):
    """
    Executes the training loop for a single fold (or split).

    Args:
        model (nn.Module): The SIA-DS EfficientNet model.
        train_loader (DataLoader): Loader for training data.
        val_loader (DataLoader): Loader for validation data.
        num_epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        save_path (str): Path to save the best model checkpoint.

    Returns:
        float: The best AUC score achieved.
    """
    device = get_device()
    model = model.to(device)

    # Optimizer configuration per SIA-DS strategy:
    # AdamW with Learning Rate 1e-4 and Weight Decay 1e-2
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # Loss function for binary classification
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(num_epochs):
        # Train
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss:.10f}, Train AUC: {train_auc:.10f} - "
            f"Val Loss: {val_loss:.10f}, Val AUC: {val_auc:.10f}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0

            # Save the best model state
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auc": best_auc,
                },
                save_path,
            )

            print(f"New best model saved with AUC: {best_auc:.10f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return best_auc


def train_and_predict(
    num_epochs=20,
    batch_size=32,
    patience=5,
    limit_size=None,
    submission_path="./submission/submission.csv",
):
    """
    Main pipeline function to setup datasets, run training, and generate submission.

    Args:
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoaders.
        patience (int): Early stopping patience.
        limit_size (int, optional): Limit dataset size for debugging.
        submission_path (str): Path to save the submission CSV.
    """
    # Ensure reproducibility
    set_seed(42)

    # 1. Load Datasets
    # Uses library.dataset to handle metadata processing and caching
    train_ds, val_ds, test_ds = get_datasets(limit_size=limit_size)

    # 2. Integrity Check
    # Explicitly verify dataset size matches expectation (~523 subjects)
    total_labeled = len(train_ds) + len(val_ds)
    print(
        f"Dataset Integrity Check: Loaded {total_labeled} labeled subjects (Train + Val)."
    )

    if limit_size is None and total_labeled < 500:
        print(
            "WARNING: Dataset size is significantly lower than the expected ~523 cases. "
            "Check metadata generation or exclusion lists."
        )

    # 3. Create DataLoaders
    # Num_workers set to 4 for efficient data loading
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # 4. Initialize Model
    # SIA-DS EfficientNet with dropout rate 0.3 as per strategy
    model = SIA_DS_EfficientNet(num_classes=1, drop_rate=0.3)

    # 5. Run Training (Fold execution)
    save_path = "./working/best_model.pth"
    run_fold(model, train_loader, val_loader, num_epochs, patience, save_path)

    # 6. Inference and Submission
    print("Loading best model for inference...")

    # Re-initialize model to ensure clean state and load best weights
    inference_model = SIA_DS_EfficientNet(num_classes=1, drop_rate=0.3)
    load_checkpoint(save_path, inference_model)

    # Generate predictions for the test set
    generate_submission(inference_model, test_loader, submission_path)
