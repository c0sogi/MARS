import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, load_metadata, save_submission
from library.dataset import SiameseWhaleDataset, WhaleInferenceDataset
from library.model import EmbeddingNet, SiameseNet
from library.loss import ContrastiveLoss
from library.engine import train_one_epoch, extract_embeddings
from library.inference import predict_knn


def main():
    # ---------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # ---------------------------------------------------------
    print(">>> Setting up configuration for demonstration...")

    # Set reproducibility
    seed_everything(Config.SEED)

    # Override Config for speed and demonstration purposes
    Config.IMG_SIZE = 64  # Reduce image size for speed
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_EPOCHS = 1  # Only 1 epoch
    Config.EMBEDDING_DIM = 16  # Small embedding dimension

    # Define a demo working directory
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Device: {Config.DEVICE}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Data Preparation (Subsetting)
    # ---------------------------------------------------------
    print("\n>>> Preparing data subsets...")

    # Load full metadata
    df_train_full = load_metadata("train")
    df_test_full = load_metadata("test")

    # Create a small training subset ensuring at least 2 different IDs
    # This is crucial for the SiameseDataset negative sampling logic
    unique_ids = df_train_full["Id"].unique()
    assert len(unique_ids) >= 2, "Training data must have at least 2 classes."

    # Select top 20 samples, but ensure we have mixed classes
    df_train_subset = df_train_full.head(20).copy()

    # If by chance the head(20) has only 1 ID, force inject another one
    if df_train_subset["Id"].nunique() < 2:
        diff_id_row = df_train_full[
            df_train_full["Id"] != df_train_subset.iloc[0]["Id"]
        ].iloc[0]
        df_train_subset = pd.concat(
            [df_train_subset, pd.DataFrame([diff_id_row])], ignore_index=True
        )

    print(f"Training subset size: {len(df_train_subset)}")
    print(f"Unique IDs in subset: {df_train_subset['Id'].nunique()}")

    # Create a small test subset
    df_test_subset = df_test_full.head(5).copy()
    print(f"Test subset size: {len(df_test_subset)}")

    # ---------------------------------------------------------
    # 3. Dataset & DataLoader Verification
    # ---------------------------------------------------------
    print("\n>>> Verifying SiameseWhaleDataset...")

    # Instantiate dataset (disable loading from existing cache to force processing subset)
    train_dataset = SiameseWhaleDataset(df_train_subset, load_cached_data=False)

    # Verify length
    assert len(train_dataset) == len(
        df_train_subset
    ), f"Dataset length mismatch. Expected {len(df_train_subset)}, got {len(train_dataset)}"

    # Verify item structure
    (img1, img2), label = train_dataset[0]

    # Check tensor shapes: (3, IMG_SIZE, IMG_SIZE)
    expected_shape = (3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert img1.shape == expected_shape, f"Image 1 shape mismatch. Got {img1.shape}"
    assert img2.shape == expected_shape, f"Image 2 shape mismatch. Got {img2.shape}"

    # Check label type (scalar tensor)
    assert isinstance(label, torch.Tensor), "Label should be a tensor"
    assert label.ndim == 0, "Label should be a scalar (0-dim tensor)"

    print("SiameseWhaleDataset verification passed.")

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Use 0 workers for simple debug to avoid multiprocessing overhead
    )

    # ---------------------------------------------------------
    # 4. Model Instantiation
    # ---------------------------------------------------------
    print("\n>>> Instantiating Model...")

    embedding_net = EmbeddingNet()
    model = SiameseNet(embedding_net)
    model.to(Config.DEVICE)

    # Verify forward pass
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(Config.DEVICE)
    out1, out2 = model(dummy_input, dummy_input)

    assert out1.shape == (
        2,
        Config.EMBEDDING_DIM,
    ), f"Output shape mismatch. Got {out1.shape}"
    print("Model forward pass verification passed.")

    # ---------------------------------------------------------
    # 5. Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n>>> Running Training Loop (1 Epoch)...")

    criterion = ContrastiveLoss(margin=Config.MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Run one epoch
    loss = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)

    print(f"Epoch 1 Loss: {loss:.4f}")
    assert not np.isnan(loss), "Training loss is NaN."

    # ---------------------------------------------------------
    # 6. Inference Pipeline Demonstration
    # ---------------------------------------------------------
    print("\n>>> Running Inference Pipeline...")

    # 6a. Extract Reference Embeddings (Train Subset)
    # We use WhaleInferenceDataset which yields single images
    ref_dataset = WhaleInferenceDataset(
        df_train_subset, load_cached_data=False, cache_name="demo_ref_cache.npy"
    )
    ref_loader = DataLoader(
        ref_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    ref_results = extract_embeddings(model, ref_loader, Config.DEVICE)
    train_embs = ref_results["embeddings"]
    train_ids = np.array(ref_results["ids"])
    train_imgs = np.array(ref_results["images"])

    assert train_embs.shape == (len(df_train_subset), Config.EMBEDDING_DIM)
    print(f"Reference embeddings extracted: {train_embs.shape}")

    # 6b. Extract Query Embeddings (Test Subset)
    query_dataset = WhaleInferenceDataset(
        df_test_subset, load_cached_data=False, cache_name="demo_query_cache.npy"
    )
    query_loader = DataLoader(
        query_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    query_results = extract_embeddings(model, query_loader, Config.DEVICE)
    test_embs = query_results["embeddings"]
    test_imgs = np.array(query_results["images"])

    assert test_embs.shape == (len(df_test_subset), Config.EMBEDDING_DIM)
    print(f"Query embeddings extracted: {test_embs.shape}")

    # 6c. KNN Prediction
    # Note: k must be <= number of reference samples.
    # Since we have ~20 samples, k=5 is safe.
    print("Running KNN Prediction...")
    predictions = predict_knn(
        test_embeddings=test_embs,
        train_embeddings=train_embs,
        train_labels=train_ids,
        test_filenames=test_imgs,
        threshold=0.5,  # Arbitrary threshold for demo
        k=min(5, len(train_ids)),
    )

    # Verify predictions structure
    assert len(predictions) == len(df_test_subset)
    assert (
        len(predictions[0]) == 5
    ), f"Expected 5 predictions per image, got {len(predictions[0])}"
    print(f"Sample prediction for {test_imgs[0]}: {predictions[0]}")

    # ---------------------------------------------------------
    # 7. Submission Generation
    # ---------------------------------------------------------
    print("\n>>> Generating Submission File...")

    sub_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    save_submission(test_imgs, predictions, filename=sub_path)

    assert os.path.exists(sub_path), "Submission file was not created."

    # Check file content
    df_sub = pd.read_csv(sub_path)
    print(df_sub.head())
    assert df_sub.shape == (len(df_test_subset), 2), "Submission shape mismatch."
    assert list(df_sub.columns) == ["Image", "Id"], "Submission columns mismatch."

    print("\n=======================================================")
    print("   Demonstration Completed Successfully")
    print("=======================================================")


if __name__ == "__main__":
    main()
