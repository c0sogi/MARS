import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil
import time

# Import library modules
from library.config import Config
from library.utils import seed_everything, map_at_5
from library.dataset import get_loaders, WhaleDataset
from library.model import WhaleModel
from library.loss import ArcFaceLoss
from library.engine import train_fn, eval_fn
from library.rerank import re_ranking
from library.inference import inference_pipeline


def create_mini_dataset(input_csv, output_csv, n_samples=20, filter_new_whale=False):
    """
    Creates a smaller version of the dataset CSV for rapid testing.
    """
    df = pd.read_csv(input_csv)

    if filter_new_whale:
        # Ensure we have known whales for training
        df = df[df["Id"] != "new_whale"]

    # Sample data
    n = min(n_samples, len(df))
    df_mini = df.sample(n=n, random_state=42).reset_index(drop=True)

    # Save to working directory
    df_mini.to_csv(output_csv, index=False)
    print(f"Created mini dataset at {output_csv} with {len(df_mini)} samples.")
    return len(df_mini)


def test_reranking_logic():
    print("\n=== Testing Re-ranking Logic ===")
    # Simulate 5 queries and 10 gallery items with 512-dim features
    num_query = 5
    num_gallery = 10
    dim = 512

    # Random normalized features
    query_feats = np.random.rand(num_query, dim).astype(np.float32)
    gallery_feats = np.random.rand(num_gallery, dim).astype(np.float32)

    # Run re-ranking
    # k1 and k2 must be smaller than gallery size for this small test
    dist_matrix = re_ranking(query_feats, gallery_feats, k1=4, k2=2, lambda_value=0.3)

    # Check shape: (num_query, num_gallery)
    assert dist_matrix.shape == (
        num_query,
        num_gallery,
    ), f"Re-ranking output shape mismatch. Expected {(num_query, num_gallery)}, got {dist_matrix.shape}"

    print("Re-ranking logic verified. Output shape correct.")


def main():
    print("Starting Whale Identification Library Demo...")
    seed_everything(42)

    # ---------------------------------------------------------
    # 1. Patch Configuration for Speed
    # ---------------------------------------------------------
    print("\n=== Configuring Environment for Rapid Demo ===")

    # Create temporary directory for mini metadata
    mini_meta_dir = "./working/mini_metadata"
    os.makedirs(mini_meta_dir, exist_ok=True)

    # Define paths for mini CSVs
    mini_train_path = os.path.join(mini_meta_dir, "train.csv")
    mini_val_path = os.path.join(mini_meta_dir, "val.csv")
    mini_test_path = os.path.join(mini_meta_dir, "test.csv")

    # Create mini datasets
    # Note: We filter new_whale from train to ensure we have valid classes for the model
    create_mini_dataset(
        Config.train_csv_path, mini_train_path, n_samples=30, filter_new_whale=True
    )
    create_mini_dataset(Config.val_csv_path, mini_val_path, n_samples=10)
    create_mini_dataset(Config.test_csv_path, mini_test_path, n_samples=10)

    # Override Config attributes
    Config.train_csv_path = mini_train_path
    Config.val_csv_path = mini_val_path
    Config.test_csv_path = mini_test_path

    # Point caches to working dir to avoid loading full dataset caches if they exist
    Config.train_images_cache = "./working/mini_train_images.npy"
    Config.val_images_cache = "./working/mini_val_images.npy"
    Config.test_images_cache = "./working/mini_test_images.npy"
    Config.test_ids_cache = "./working/mini_test_ids.npy"

    # Reduce computational load
    Config.image_size = 128  # Small resolution for speed
    Config.batch_size = 4
    Config.num_workers = 0  # Avoid multiprocessing overhead for tiny data
    Config.backbone = "resnet18"  # Use a lighter backbone if supported by timm, else keep efficientnet but small
    # Note: The library uses 'tf_efficientnet_b4'. We'll stick to it but use small image size.
    # To be safe with the provided code, we won't change the backbone string as it might be hardcoded elsewhere,
    # but reducing image size is sufficient.

    Config.epochs = 1
    Config.model_save_path = "./working/demo_model.pth"
    Config.submission_path = "./working/demo_submission.csv"

    print("Configuration patched successfully.")

    # ---------------------------------------------------------
    # 2. Test Data Loading
    # ---------------------------------------------------------
    print("\n=== Testing Data Loading ===")

    # get_loaders will now use our mini CSVs and create new caches
    (
        train_loader,
        val_loader,
        gallery_loader,
        test_loader,
        num_classes,
        label_encoder,
    ) = get_loaders(
        load_cached_data=False  # Force re-creation of cache for mini data
    )

    print(f"Num classes in mini-train: {num_classes}")

    # Verify Train Loader
    images, labels = next(iter(train_loader))
    print(f"Train Batch Shape: Images {images.shape}, Labels {labels.shape}")

    assert images.shape == (
        Config.batch_size,
        3,
        Config.image_size,
        Config.image_size,
    ), "Train image batch shape incorrect"
    assert len(labels) == Config.batch_size, "Train label batch size incorrect"

    # Verify Test Loader (no labels)
    test_images = next(iter(test_loader))
    print(f"Test Batch Shape: {test_images.shape}")
    assert test_images.shape[0] == Config.batch_size or test_images.shape[0] <= len(
        test_loader.dataset
    ), "Test batch size incorrect"

    # ---------------------------------------------------------
    # 3. Test Model & Loss
    # ---------------------------------------------------------
    print("\n=== Testing Model & Loss ===")

    device = Config.device

    # Instantiate Model
    model = WhaleModel(
        embedding_size=Config.embedding_size, pretrained=False
    )  # No need to download weights for demo
    model.to(device)

    # Forward Pass
    dummy_input = images.to(device)
    embeddings = model(dummy_input)

    print(f"Embeddings Shape: {embeddings.shape}")
    assert embeddings.shape == (
        Config.batch_size,
        Config.embedding_size,
    ), "Model output shape mismatch"

    # Instantiate ArcFace Loss
    criterion = ArcFaceLoss(
        num_classes=num_classes, embedding_size=Config.embedding_size
    )
    criterion.to(device)

    # Calculate Loss
    dummy_labels = labels.to(device)
    loss = criterion(embeddings, dummy_labels)

    print(f"Calculated Loss: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    # ---------------------------------------------------------
    # 4. Test Re-ranking
    # ---------------------------------------------------------
    test_reranking_logic()

    # ---------------------------------------------------------
    # 5. Test Training Loop (Engine)
    # ---------------------------------------------------------
    print("\n=== Testing Training Engine ===")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Run 1 epoch of training
    epoch_loss = train_fn(train_loader, model, criterion, optimizer, device)
    print(f"Training Epoch Loss: {epoch_loss:.4f}")

    # Run evaluation
    # Note: eval_fn uses map_at_5 and re_ranking internally
    val_score = eval_fn(val_loader, gallery_loader, model, device, label_encoder)
    print(f"Validation MAP@5: {val_score:.4f}")

    # Save this model to simulate a trained state
    torch.save(model.state_dict(), Config.model_save_path)
    print(f"Saved demo model to {Config.model_save_path}")

    # ---------------------------------------------------------
    # 6. Test Inference Pipeline
    # ---------------------------------------------------------
    print("\n=== Testing Inference Pipeline ===")

    # We use the inference_pipeline function which handles loading, model init, and submission generation.
    # We pass debug_limit to ensure it uses our patched logic or just runs fast.
    # Since we already patched Config to point to mini datasets, we don't strictly need debug_limit,
    # but using it exercises that code path.

    # Note: inference_pipeline re-initializes the model and loads weights from Config.model_save_path
    inference_pipeline(load_cached_data=True, debug_limit=10)

    # Verify submission file exists
    if os.path.exists(Config.submission_path):
        sub_df = pd.read_csv(Config.submission_path)
        print(f"Submission generated with {len(sub_df)} rows.")
        print(sub_df.head())
        assert len(sub_df) > 0, "Submission file is empty"
        assert (
            "Image" in sub_df.columns and "Id" in sub_df.columns
        ), "Submission columns mismatch"
    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.submission_path}"
        )

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
