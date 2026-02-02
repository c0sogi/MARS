import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import shutil

# Ensure the current directory is in the path to import library modules
sys.path.append(".")

from library.config import CFG
from library.utils import seed_everything, get_logger, save_checkpoint
from library.dataset import get_loaders
from library.model import CassavaConvNeXt
from library.engine import train_one_epoch, valid_one_epoch, inference_fn


def create_subset_metadata(source_path, dest_path, n_samples=100):
    """
    Reads the source metadata CSV, samples n_samples, and saves to dest_path.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source metadata not found: {source_path}")

    df = pd.read_csv(source_path)
    # Sample or take head if n_samples < len(df)
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=CFG.seed).reset_index(drop=True)

    df.to_csv(dest_path, index=False)
    print(f"Created subset {dest_path} with {len(df)} samples.")
    return len(df)


def run_demo():
    # 1. Setup
    seed_everything(CFG.seed)

    # Define a working directory for this demo
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)

    # 2. Override Configuration for Speed and Demo purposes
    print("--- Configuring Demo Settings ---")
    CFG.output_dir = demo_dir
    CFG.model_name = "resnet18"  # Use a lighter model for speed
    CFG.img_size_base = 224  # Smaller image size
    CFG.batch_size = 16
    CFG.valid_batch_size = 32
    CFG.epochs_base = 1  # Only 1 epoch
    CFG.epochs_warmup = 0  # Skip warmup
    CFG.num_workers = 2  # Reduce workers for small data
    CFG.print_freq = 5

    # 3. Create Data Subsets
    print("\n--- Creating Data Subsets ---")
    subset_train_path = os.path.join(demo_dir, "train_subset.csv")
    subset_val_path = os.path.join(demo_dir, "val_subset.csv")
    subset_test_path = os.path.join(demo_dir, "test_subset.csv")

    # We use the existing metadata in ./metadata to create subsets in ./working
    create_subset_metadata(CFG.train_csv, subset_train_path, n_samples=100)
    create_subset_metadata(CFG.val_csv, subset_val_path, n_samples=50)
    create_subset_metadata(CFG.test_csv, subset_test_path, n_samples=20)

    # Point CFG to these new subset files
    CFG.train_csv = subset_train_path
    CFG.val_csv = subset_val_path
    CFG.test_csv = subset_test_path

    # 4. Initialize DataLoaders
    print("\n--- Initializing DataLoaders ---")
    # load_cached_data=False ensures we read the new CSVs we just created
    train_loader, val_loader, test_loader = get_loaders(
        img_size=CFG.img_size_base, load_cached_data=False
    )

    # Verify DataLoader
    batch_img, batch_target = next(iter(train_loader))
    print(f"Train Batch Shape: {batch_img.shape}")
    assert batch_img.shape == (CFG.batch_size, 3, CFG.img_size_base, CFG.img_size_base)
    # Targets should be (Batch, Num_Classes) due to MixupCollate
    assert batch_target.shape == (CFG.batch_size, CFG.num_classes)

    # 5. Initialize Model
    print("\n--- Initializing Model ---")
    device = torch.device(CFG.device)
    model = CassavaConvNeXt(model_name=CFG.model_name, pretrained=True)
    model.to(device)

    # 6. Setup Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )
    # Simple scheduler for demo
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs_base)

    # 7. Training Loop
    print("\n--- Starting Training ---")
    avg_loss = train_one_epoch(
        epoch=1,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        accum_iter=CFG.accum_iter,
    )

    print(f"Training finished. Avg Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Training loss is NaN"

    # 8. Validation Loop
    print("\n--- Starting Validation ---")
    val_loss, val_acc = valid_one_epoch(
        epoch=1, model=model, val_loader=val_loader, device=device
    )
    print(f"Validation finished. Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    assert 0 <= val_acc <= 1.0, "Accuracy out of bounds"

    # Save the model (simulating the best model save)
    save_checkpoint(
        {"state_dict": model.state_dict()},
        is_best=True,
        output_dir=CFG.output_dir,
        filename="demo_best_model.pth",
    )
    assert os.path.exists(os.path.join(CFG.output_dir, "demo_best_model.pth"))

    # 9. Inference
    print("\n--- Starting Inference ---")
    predictions = inference_fn(
        model=model,
        test_loader=test_loader,
        device=device,
        tta_steps=2,  # Use 2 steps (Original + HFlip) for demo
    )

    print(f"Inference finished. Predictions shape: {predictions.shape}")
    assert (
        len(predictions) == 20
    ), "Number of predictions does not match test subset size"

    # 10. Generate Submission
    print("\n--- Generating Submission ---")
    test_df = pd.read_csv(subset_test_path)
    test_df["label"] = predictions

    # Keep only required columns
    submission_df = test_df[["image_id", "label"]]

    # Save to working directory
    sub_path = os.path.join(CFG.output_dir, "submission.csv")
    submission_df.to_csv(sub_path, index=False)

    print(f"Submission saved to {sub_path}")
    print(submission_df.head())

    # Final check
    assert os.path.exists(sub_path)
    print("\nDemo completed successfully.")


if __name__ == "__main__":
    run_demo()
