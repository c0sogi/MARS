import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_loaders
from library.model import WhaleModel
from library.train import run_training
from library.evaluate import validate, inference
from library.rerank import re_ranking


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print(">>> Step 1: Configuring environment for fast demonstration...")

    # Override Config for speed and debugging
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 images
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.IMAGE_SIZE = 224  # Smaller image size for speed
    Config.NUM_WORKERS = 2  # Moderate workers

    # Clean up previous working directory to ensure a fresh run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print("Configuration updated for debug mode.")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n>>> Step 2: Verifying Data Loaders...")

    # Load data (load_cached_data=False ensures we process the debug subset freshly)
    train_loader, val_loader, test_loader, num_classes = get_loaders(
        load_cached_data=False
    )

    # Verify Train Loader
    images, labels = next(iter(train_loader))
    print(f"Train Batch - Images: {images.shape}, Labels: {labels.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect image batch shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label batch shape"
    assert num_classes > 0, "Number of classes should be positive"

    print("Data Loaders verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Model Verification
    # -------------------------------------------------------------------------
    print("\n>>> Step 3: Verifying Model Architecture...")

    device = Config.DEVICE
    model = WhaleModel(num_classes=num_classes)
    model = model.to(device)

    # Dummy Input
    dummy_input = torch.randn(
        Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE
    ).to(device)
    dummy_labels = torch.randint(0, num_classes, (Config.BATCH_SIZE,)).to(device)

    # Test Forward (Training Mode - Returns Logits via CurricularFace)
    model.train()
    logits = model(dummy_input, dummy_labels)
    assert logits.shape == (
        Config.BATCH_SIZE,
        num_classes,
    ), f"Expected logits shape {(Config.BATCH_SIZE, num_classes)}, got {logits.shape}"

    # Test Forward (Inference Mode - Returns Embeddings)
    model.eval()
    embeddings = model(dummy_input, label=None)
    assert embeddings.shape == (
        Config.BATCH_SIZE,
        Config.EMBEDDING_DIM,
    ), f"Expected embedding shape {(Config.BATCH_SIZE, Config.EMBEDDING_DIM)}, got {embeddings.shape}"

    print("Model architecture verified successfully.")

    # -------------------------------------------------------------------------
    # 4. Training Execution
    # -------------------------------------------------------------------------
    print("\n>>> Step 4: Running Training Loop...")

    # run_training uses the Config settings we modified
    # It will initialize its own model and loaders, but that's fine for the demo
    run_training()

    assert os.path.exists(Config.MODEL_PATH), "Model checkpoint was not saved."
    print("Training completed and model saved.")

    # -------------------------------------------------------------------------
    # 5. Evaluation & Inference
    # -------------------------------------------------------------------------
    print("\n>>> Step 5: Running Evaluation and Inference...")

    # Load the best model
    checkpoint = torch.load(Config.MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # Run Validation (MAP@5)
    # We disable caching here to ensure it runs through the extraction logic
    map5_score = validate(
        model, train_loader, val_loader, device, load_cached_data=False
    )
    print(f"Validation MAP@5: {map5_score:.4f}")
    assert 0.0 <= map5_score <= 1.0, "MAP@5 score out of range"

    # Run Inference (Submission Generation)
    inference(model, train_loader, test_loader, device, load_cached_data=False)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."
    print("Inference completed.")

    # -------------------------------------------------------------------------
    # 6. Re-ranking Logic Verification
    # -------------------------------------------------------------------------
    print("\n>>> Step 6: Verifying Re-ranking Logic...")

    # Create synthetic features
    # Query: 5 samples, Gallery: 10 samples, Dim: 128
    n_query = 5
    n_gallery = 10
    dim = 128

    query_feats = np.random.rand(n_query, dim).astype(np.float32)
    gallery_feats = np.random.rand(n_gallery, dim).astype(np.float32)

    # Run re-ranking
    dist_mat = re_ranking(query_feats, gallery_feats, k1=3, k2=2, lambda_value=0.3)

    # Check shape
    assert dist_mat.shape == (
        n_query,
        n_gallery,
    ), f"Re-ranking output shape mismatch. Expected {(n_query, n_gallery)}, got {dist_mat.shape}"

    # Check values (distances should be non-negative)
    assert np.all(dist_mat >= -1e-5), "Distance matrix contains negative values"

    print("Re-ranking logic verified.")

    # -------------------------------------------------------------------------
    # 7. Submission File Verification
    # -------------------------------------------------------------------------
    print("\n>>> Step 7: Verifying Submission File Format...")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    assert list(df_sub.columns) == ["Image", "Id"], "Submission columns mismatch"

    # Check row count (Should match debug sample size)
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission row count {len(df_sub)} does not match debug size {Config.DEBUG_SAMPLE_SIZE}"

    # Check prediction format
    sample_pred = df_sub.iloc[0]["Id"]
    assert isinstance(sample_pred, str), "Prediction Id is not a string"
    assert (
        len(sample_pred.split()) == 5
    ), f"Prediction does not contain 5 labels: {sample_pred}"

    print("Submission file format verified.")
    print("\n>>> All demonstration steps completed successfully!")


if __name__ == "__main__":
    main()
