import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from unittest.mock import patch

# Import provided library modules
from library.utils import seed_everything, AverageMeter, weight_soup
from library.data_loader import get_dataloaders
from library.model import HierarchicalEfficientNet
from library.trainer import train_one_epoch, validate_one_epoch
from library.inference import generate_submission

# Configuration
WORKING_DIR = "./working/demo_run"
os.makedirs(WORKING_DIR, exist_ok=True)


class TruncatedDataLoader:
    """Wrapper to limit the number of batches yielded by a DataLoader."""

    def __init__(self, dataloader, max_batches=2):
        self.dataloader = dataloader
        self.max_batches = max_batches

    def __iter__(self):
        for i, batch in enumerate(self.dataloader):
            if i >= self.max_batches:
                break
            yield batch

    def __len__(self):
        return min(len(self.dataloader), self.max_batches)


def test_utils():
    print("\n=== Testing Utils ===")

    # 1. Test AverageMeter
    meter = AverageMeter()
    meter.update(10, n=1)
    meter.update(20, n=1)
    assert meter.avg == 15.0, f"AverageMeter failed: expected 15.0, got {meter.avg}"
    assert meter.count == 2, "AverageMeter count failed"
    print("AverageMeter: OK")

    # 2. Test weight_soup
    ckpt1_path = os.path.join(WORKING_DIR, "ckpt1.pth")
    ckpt2_path = os.path.join(WORKING_DIR, "ckpt2.pth")

    # Create dummy state dicts
    w1 = {"weight": torch.tensor([1.0, 2.0])}
    w2 = {"weight": torch.tensor([3.0, 4.0])}

    torch.save(w1, ckpt1_path)
    torch.save(w2, ckpt2_path)

    soup = weight_soup([ckpt1_path, ckpt2_path])
    expected = torch.tensor([2.0, 3.0])

    assert torch.allclose(
        soup["weight"], expected
    ), "weight_soup failed to average correctly"
    print("weight_soup: OK")


def test_data_loader_and_model():
    print("\n=== Testing Data Loader and Model ===")

    # 1. Get DataLoaders
    # We use a small batch size for speed
    batch_size = 4
    train_loader, val_loader, test_loader, num_diag = get_dataloaders(
        batch_size=batch_size,
        image_size=256,  # Smaller image size for speed
        num_workers=0,  # Avoid multiprocessing overhead for this small test
        load_cached_data=True,
    )

    # 2. Inspect one batch
    batch = next(iter(train_loader))
    images = batch["image"]
    meta = batch["meta"]
    targets = batch["target"]
    diagnosis = batch["diagnosis"]

    print(f"Batch keys: {list(batch.keys())}")
    print(f"Image shape: {images.shape}")
    print(f"Meta shape: {meta.shape}")

    assert images.shape == (batch_size, 3, 256, 256), "Incorrect image shape"
    assert "image_name" in batch, "Missing image_name in batch"

    # 3. Initialize Model
    # Use pretrained=False to avoid downloading weights and speed up init
    n_meta_features = meta.shape[1]
    model = HierarchicalEfficientNet(
        model_name="efficientnet_b3",
        pretrained=False,
        n_meta_features=n_meta_features,
        n_diagnosis_classes=num_diag,
        num_classes=1,
    )

    # 4. Forward Pass
    device = "cpu"  # Use CPU for simple logic verification
    model.to(device)
    images = images.to(device)
    meta = meta.to(device)

    primary_logits, aux_logits = model(images, meta)

    print(f"Primary logits shape: {primary_logits.shape}")
    print(f"Aux logits shape: {aux_logits.shape}")

    assert primary_logits.shape == (batch_size, 1), "Incorrect primary logits shape"
    assert aux_logits.shape == (batch_size, num_diag), "Incorrect aux logits shape"

    return model, train_loader, val_loader, test_loader, num_diag, n_meta_features


def test_training_loop(model, train_loader, val_loader, num_diag):
    print("\n=== Testing Training Loop ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Setup Optimizer and Criterion
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    # Dummy scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    criterion_primary = nn.BCEWithLogitsLoss()
    criterion_aux = nn.CrossEntropyLoss()

    # Use Truncated Loaders to run only a few steps
    truncated_train = TruncatedDataLoader(train_loader, max_batches=2)
    truncated_val = TruncatedDataLoader(val_loader, max_batches=2)

    # Train one epoch
    print("Running train_one_epoch...")
    train_loss = train_one_epoch(
        model,
        truncated_train,
        optimizer,
        scheduler,
        criterion_primary,
        criterion_aux,
        device,
    )
    print(f"Train Loss: {train_loss}")
    assert isinstance(train_loss, float), "Train loss should be a float"

    # Validate one epoch
    print("Running validate_one_epoch...")
    val_loss, val_auc = validate_one_epoch(
        model, truncated_val, criterion_primary, criterion_aux, device
    )
    print(f"Val Loss: {val_loss}, Val AUC: {val_auc}")
    assert isinstance(val_loss, float), "Val loss should be a float"
    assert 0.0 <= val_auc <= 1.0, "AUC should be between 0 and 1"


def test_inference(model, test_loader, num_diag):
    print("\n=== Testing Inference ===")

    # Save the current model as 'model_best.pth' to simulate a trained model
    model_path = os.path.join(WORKING_DIR, "model_best.pth")
    torch.save(model.state_dict(), model_path)

    output_csv = os.path.join(WORKING_DIR, "submission.csv")

    # We mock get_dataloaders in library.inference to return our truncated test loader
    # This avoids loading the full test set during the inference test
    truncated_test = TruncatedDataLoader(test_loader, max_batches=2)

    # The signature of get_dataloaders returns (train, val, test, num_diag)
    # We return None for train/val as they aren't used in inference
    mock_return_val = (None, None, truncated_test, num_diag)

    print("Running generate_submission with mocked dataloader...")
    with patch("library.inference.get_dataloaders", return_value=mock_return_val):
        generate_submission(
            model_path=model_path,
            output_path=output_csv,
            batch_size=4,
            device="cuda" if torch.cuda.is_available() else "cpu",
            load_cached_data=True,
        )

    assert os.path.exists(output_csv), "Submission file was not created"

    df_sub = pd.read_csv(output_csv)
    print(f"Submission rows: {len(df_sub)}")
    print(df_sub.head())

    assert list(df_sub.columns) == [
        "image_name",
        "target",
    ], "Incorrect submission columns"
    # We expect 2 batches * 4 images = 8 rows (or less if last batch is smaller)
    assert len(df_sub) > 0, "Submission file is empty"


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)

    # 2. Utils
    test_utils()

    # 3. Data & Model
    model, train_loader, val_loader, test_loader, num_diag, n_meta = (
        test_data_loader_and_model()
    )

    # 4. Training
    test_training_loop(model, train_loader, val_loader, num_diag)

    # 5. Inference
    test_inference(model, test_loader, num_diag)

    print("\nAll tests passed successfully.")
