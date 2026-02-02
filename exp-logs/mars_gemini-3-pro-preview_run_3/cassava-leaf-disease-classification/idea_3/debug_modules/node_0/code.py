import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import CFG, seed_everything
from library.dataset import CassavaDataset, get_transforms, Mixup
from library.model import CassavaViT
from library.engine import train_one_epoch, valid_one_epoch
from library.inference import run_inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def demo_pipeline():
    print("=== Starting Cassava Leaf Disease Classification Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # -------------------------------------------------------------------------
    print("[1/6] Configuring environment and overrides...")

    # Set seed for reproducibility
    seed_everything(CFG.seed)

    # Override CFG for speed and demonstration purposes
    CFG.model_name = "resnet18"  # Use a lighter model than ViT for demo
    CFG.img_size = 224  # Smaller image size for speed
    CFG.batch_size = 8  # Small batch size
    CFG.epochs = 1  # Only 1 epoch
    CFG.num_workers = 0  # Avoid multiprocessing overhead for small data
    CFG.print_freq = 5  # Frequent logging
    CFG.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define working paths
    demo_model_path = os.path.join(CFG.output_dir, "demo_model.pth")
    demo_submission_path = os.path.join(CFG.output_dir, "submission.csv")
    temp_test_csv = os.path.join(CFG.output_dir, "temp_test.csv")

    print(f"    Device: {CFG.device}")
    print(f"    Model: {CFG.model_name}")
    print(f"    Image Size: {CFG.img_size}")

    # -------------------------------------------------------------------------
    # 2. Dataset and Transforms Verification
    # -------------------------------------------------------------------------
    print("\n[2/6] Verifying Dataset and Transforms...")

    # Load metadata
    df_train = pd.read_csv(CFG.train_csv)
    df_val = pd.read_csv(CFG.val_csv)

    # Create tiny subsets
    subset_size = 16
    df_train_sub = df_train.head(subset_size).copy()
    df_val_sub = df_val.head(subset_size).copy()

    # Initialize Datasets
    train_dataset = CassavaDataset(
        df_train_sub, transform=get_transforms(data="train"), output_label=True
    )
    val_dataset = CassavaDataset(
        df_val_sub, transform=get_transforms(data="valid"), output_label=True
    )

    # Verify Train Item
    img, label = train_dataset[0]
    assert isinstance(img, torch.Tensor), "Image is not a Tensor"
    assert img.shape == (
        3,
        CFG.img_size,
        CFG.img_size,
    ), f"Unexpected image shape: {img.shape}"
    assert isinstance(label, torch.Tensor), "Label is not a Tensor"
    print("    Dataset item check passed.")

    # Initialize DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
    )

    # -------------------------------------------------------------------------
    # 3. Mixup/CutMix Verification
    # -------------------------------------------------------------------------
    print("\n[3/6] Verifying Mixup Augmentation...")

    mixup_fn = Mixup(prob=1.0, num_classes=CFG.num_classes)  # Force mixup/cutmix

    # Get a batch
    images, labels = next(iter(train_loader))
    images = images.to(CFG.device)
    labels = labels.to(CFG.device)

    # Apply mixup
    mixed_images, mixed_labels = mixup_fn(images, labels)

    # Assertions
    assert mixed_images.shape == images.shape, "Mixed images shape mismatch"
    assert mixed_labels.shape == (
        CFG.batch_size,
        CFG.num_classes,
    ), "Mixed labels shape mismatch (should be one-hot/soft)"
    assert mixed_labels.dtype == torch.float, "Mixed labels should be float"
    print("    Mixup application check passed.")

    # -------------------------------------------------------------------------
    # 4. Model Initialization and Training Loop
    # -------------------------------------------------------------------------
    print("\n[4/6] Initializing Model and Running Training Loop...")

    model = CassavaViT(
        model_name=CFG.model_name, pretrained=True, num_classes=CFG.num_classes
    )
    model.to(CFG.device)

    # Loss and Optimizer
    # Note: SoftTargetCrossEntropy is ideal for Mixup, but standard CrossEntropy works if targets are class indices.
    # Since Mixup returns soft labels [B, C], we need a loss that handles probabilities.
    # Standard nn.CrossEntropyLoss expects class indices, or (in newer PyTorch) class probabilities.
    # We'll use a simple implementation compatible with soft labels for this demo.
    class SoftTargetCrossEntropy(nn.Module):
        def forward(self, x, target):
            loss = torch.sum(
                -target * torch.nn.functional.log_softmax(x, dim=-1), dim=-1
            )
            return loss.mean()

    criterion = SoftTargetCrossEntropy()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)

    # Run one epoch of training
    avg_loss, avg_acc = train_one_epoch(
        epoch=0,
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=CFG.device,
        mixup_fn=mixup_fn,
    )

    print(f"    Train Result - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
    assert not np.isnan(avg_loss), "Training loss is NaN"

    # -------------------------------------------------------------------------
    # 5. Validation Loop
    # -------------------------------------------------------------------------
    print("\n[5/6] Running Validation Loop...")

    # For validation, we use standard CrossEntropyLoss with hard labels
    val_criterion = nn.CrossEntropyLoss()

    val_loss, val_acc = valid_one_epoch(
        epoch=0,
        model=model,
        dataloader=val_loader,
        criterion=val_criterion,
        device=CFG.device,
    )

    print(f"    Valid Result - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"

    # Save the model for inference step
    torch.save(model.state_dict(), demo_model_path)
    print(f"    Model saved to {demo_model_path}")

    # -------------------------------------------------------------------------
    # 6. Inference Pipeline
    # -------------------------------------------------------------------------
    print("\n[6/6] Verifying Inference Pipeline...")

    # Prepare a temporary test csv subset
    df_test_full = pd.read_csv(CFG.test_csv)
    df_test_sub = df_test_full.head(subset_size).copy()
    df_test_sub.to_csv(temp_test_csv, index=False)

    # Temporarily point CFG to this new test file
    original_test_csv_path = CFG.test_csv
    CFG.test_csv = temp_test_csv

    try:
        # Run inference
        run_inference(
            model_checkpoint_path=demo_model_path, output_csv_path=demo_submission_path
        )

        # Verify submission file
        assert os.path.exists(demo_submission_path), "Submission file not created"
        sub_df = pd.read_csv(demo_submission_path)
        assert len(sub_df) == len(df_test_sub), "Submission row count mismatch"
        assert (
            "image_id" in sub_df.columns and "label" in sub_df.columns
        ), "Submission columns missing"
        assert sub_df["label"].dtype == np.int64, "Label column should be integer"

        print("    Inference successful. Submission file verified.")

    finally:
        # Restore CFG and cleanup
        CFG.test_csv = original_test_csv_path
        if os.path.exists(temp_test_csv):
            os.remove(temp_test_csv)
        if os.path.exists(demo_model_path):
            os.remove(demo_model_path)
        if os.path.exists(demo_submission_path):
            os.remove(demo_submission_path)

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    demo_pipeline()
