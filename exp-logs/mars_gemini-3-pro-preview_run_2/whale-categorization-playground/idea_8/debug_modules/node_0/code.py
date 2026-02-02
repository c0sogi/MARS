import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil
from torch.utils.data import DataLoader

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from provided library files
from library.config import CFG
from library.utils import seed_everything, get_logger
from library.dataset import process_data, WhaleDataset, get_transforms
from library.modeling import WhaleModel
from library.engine import train_fn, valid_fn
from library.inference import generate_submission


def run_demo():
    print("===============================================================")
    print("      Whale Species Prediction - Library Usage Demo            ")
    print("===============================================================")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup (Override for Speed)
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for demo...")

    # Modify CFG in-place to run a lightweight version
    CFG.working_dir = "./working/demo_execution"
    CFG.model_a_name = "resnet18"  # Use a small model for speed
    CFG.model_b_name = "resnet18"
    CFG.image_size_p1 = 128  # Small resolution
    CFG.image_size_p2 = 128
    CFG.batch_size = 4
    CFG.val_batch_size = 4
    CFG.epochs_p1 = 1  # Only 1 epoch
    CFG.epochs_p2 = 0
    CFG.num_workers = 2
    CFG.print_freq = 1

    # Clean working directory if it exists
    if os.path.exists(CFG.working_dir):
        shutil.rmtree(CFG.working_dir)
    os.makedirs(CFG.working_dir, exist_ok=True)

    seed_everything(CFG.seed)
    device = CFG.device
    print(f"  Device: {device}")
    print(f"  Working Directory: {CFG.working_dir}")

    # -------------------------------------------------------------------------
    # 2. Data Processing & Subsampling
    # -------------------------------------------------------------------------
    print("\n[Step 2] Processing and subsampling data...")

    # Initial processing (reads metadata, encodes labels)
    train_df, val_df, test_df, num_classes = process_data(load_cached_data=False)

    # SUBSAMPLE: Keep only a tiny fraction for the demo to run in seconds
    # We must ensure we keep enough classes to avoid errors, but for the demo
    # we just take the top N rows.
    demo_samples = 20
    train_df_small = train_df.head(demo_samples).copy()
    val_df_small = val_df.head(demo_samples).copy()
    test_df_small = test_df.head(demo_samples).copy()

    # Overwrite the cache files so subsequent library calls use the small data
    train_cache = os.path.join(CFG.working_dir, "train_processed.parquet")
    val_cache = os.path.join(CFG.working_dir, "val_processed.parquet")
    test_cache = os.path.join(CFG.working_dir, "test_processed.parquet")

    train_df_small.to_parquet(train_cache, index=False)
    val_df_small.to_parquet(val_cache, index=False)
    test_df_small.to_parquet(test_cache, index=False)

    print(f"  Subsampled Train: {len(train_df_small)}")
    print(f"  Subsampled Val:   {len(val_df_small)}")
    print(f"  Subsampled Test:  {len(test_df_small)}")
    print(f"  Num Classes:      {num_classes}")

    # -------------------------------------------------------------------------
    # 3. Dataset & DataLoader Verification
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Dataset and DataLoader...")

    transforms = get_transforms(data="train", image_size=CFG.image_size_p1)
    dataset = WhaleDataset(
        train_df_small, transform=transforms, label_col="label_idx", id_col="Id"
    )
    loader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=True)

    # Fetch one batch
    batch = next(iter(loader))
    images = batch["image"]
    labels = batch["label"]

    print(f"  Batch Image Shape: {images.shape}")
    print(f"  Batch Label Shape: {labels.shape}")

    # Assertions
    assert images.shape == (
        CFG.batch_size,
        3,
        CFG.image_size_p1,
        CFG.image_size_p1,
    ), "Incorrect image tensor shape"
    assert labels.shape == (CFG.batch_size,), "Incorrect label tensor shape"
    assert "id" in batch, "Batch missing string IDs"

    # -------------------------------------------------------------------------
    # 4. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[Step 4] Initializing Model and checking Forward pass...")

    model = WhaleModel(CFG.model_a_name, num_classes=num_classes, pretrained=True)
    model.to(device)

    # Test Training Forward (Returns Logits)
    images = images.to(device)
    labels = labels.to(device)
    logits = model(images, labels)

    print(f"  Logits Shape: {logits.shape}")
    assert logits.shape == (CFG.batch_size, num_classes), "Logits shape mismatch"

    # Test Inference Forward (Returns Embeddings)
    embeddings = model(images, labels=None)
    print(f"  Embeddings Shape: {embeddings.shape}")
    assert embeddings.shape == (
        CFG.batch_size,
        CFG.embedding_size,
    ), "Embeddings shape mismatch"

    # -------------------------------------------------------------------------
    # 5. Training Loop Demo
    # -------------------------------------------------------------------------
    print("\n[Step 5] Running Training Loop (1 Epoch)...")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)

    avg_loss = train_fn(loader, model, criterion, optimizer, device, epoch=0)
    print(f"  Epoch 1 Loss: {avg_loss:.4f}")

    assert not np.isnan(avg_loss), "Training loss is NaN"

    # -------------------------------------------------------------------------
    # 6. Validation Demo
    # -------------------------------------------------------------------------
    print("\n[Step 6] Running Validation (MAP@5)...")

    # Setup loaders for validation
    # Gallery = Train (Reference), Query = Val
    val_transforms = get_transforms(data="valid", image_size=CFG.image_size_p1)

    gallery_ds = WhaleDataset(train_df_small, transform=val_transforms, id_col="Id")
    query_ds = WhaleDataset(val_df_small, transform=val_transforms, id_col="Id")

    gallery_loader = DataLoader(
        gallery_ds, batch_size=CFG.val_batch_size, shuffle=False
    )
    query_loader = DataLoader(query_ds, batch_size=CFG.val_batch_size, shuffle=False)

    score = valid_fn(gallery_loader, query_loader, model, device)
    print(f"  Validation Score: {score:.4f}")

    # -------------------------------------------------------------------------
    # 7. Inference & Submission Generation
    # -------------------------------------------------------------------------
    print("\n[Step 7] Generating Submission (Dual-Ensemble Mock)...")

    # Save the current model as both Model A and Model B for the demo
    model_path_a = os.path.join(CFG.working_dir, "model_a.pth")
    model_path_b = os.path.join(CFG.working_dir, "model_b.pth")

    torch.save(model.state_dict(), model_path_a)
    torch.save(model.state_dict(), model_path_b)

    submission_path = os.path.join(CFG.working_dir, "submission.csv")

    # Generate submission
    # This function will reload data from the cache (which we subsampled)
    generate_submission(
        model_a_path=model_path_a,
        model_b_path=model_path_b,
        output_path=submission_path,
    )

    # Verify Submission
    if os.path.exists(submission_path):
        sub_df = pd.read_csv(submission_path)
        print(f"  Submission File Created: {submission_path}")
        print(f"  Rows: {len(sub_df)}")
        print(f"  Columns: {list(sub_df.columns)}")

        assert len(sub_df) == len(test_df_small), "Submission row count mismatch"
        assert list(sub_df.columns) == ["Image", "Id"], "Submission columns mismatch"

        # Check format of Id column
        sample_pred = sub_df.iloc[0]["Id"]
        assert isinstance(sample_pred, str), "Prediction Id is not a string"
        print("  Verification Successful.")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n===============================================================")
    print("      Demo Completed Successfully                              ")
    print("===============================================================")


if __name__ == "__main__":
    run_demo()
