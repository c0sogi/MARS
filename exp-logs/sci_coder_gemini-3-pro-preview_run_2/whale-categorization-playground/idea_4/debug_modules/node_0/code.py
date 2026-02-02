import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, map_at_5
from library.dataset import get_dataloaders
from library.model import WhaleModel
from library.loss import ArcFaceLoss
from library.rerank import re_ranking
from library.engine import WhaleEngine
from library.inference import generate_predictions


def run_demo():
    print("============================================================")
    print("      Whale Species Prediction: Library Demo Script")
    print("============================================================")

    # ------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # ------------------------------------------------------------------
    # We modify the Config class attributes in-place to create a lightweight
    # environment for this demonstration.

    print("\n[1] Configuring environment for demo...")

    # Set reproducible seeds
    seed_everything(42)

    # Override Config for speed and debug purposes
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 30  # Small subset for quick execution
    Config.IMAGE_SIZE = 128  # Reduced resolution for speed
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_EPOCHS = 1  # Single epoch training
    Config.MODEL_NAME = "tf_efficientnet_b0"  # Lighter backbone for demo

    # Setup working directory for demo outputs
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update paths to point to the demo working directory
    # This prevents overwriting actual experiment caches
    Config.CACHE_TRAIN_IMAGES = os.path.join(
        Config.WORKING_DIR, "debug_train_images_128.npy"
    )
    Config.CACHE_VAL_IMAGES = os.path.join(
        Config.WORKING_DIR, "debug_val_images_128.npy"
    )
    Config.CACHE_TEST_IMAGES = os.path.join(
        Config.WORKING_DIR, "debug_test_images_128.npy"
    )
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")

    # ------------------------------------------------------------------
    # 2. Data Loading & Preprocessing
    # ------------------------------------------------------------------
    print("\n[2] Testing Data Loading & Preprocessing...")

    # Generate dataloaders (this triggers image loading, resizing, and caching)
    # load_cached_data=False forces re-creation of the cache for this demo run
    train_loader, val_loader, test_loader, label_encoder, num_classes = get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Verify DataLoaders
    print(f"    Num Classes: {num_classes}")
    print(f"    Train Batches: {len(train_loader)}")

    # Fetch one batch to verify shapes
    images, labels = next(iter(train_loader))
    print(f"    Batch Shape - Images: {images.shape}, Labels: {labels.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect image tensor shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label tensor shape"
    assert images.dtype == torch.float32, "Images should be float32"

    # ------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # ------------------------------------------------------------------
    print("\n[3] Testing Model Architecture...")

    model = WhaleModel(num_classes=num_classes, model_name=Config.MODEL_NAME)
    model.to(Config.DEVICE)
    model.eval()

    # Move batch to device
    images = images.to(Config.DEVICE)

    # Test Inference Mode (No Labels) -> Should return Embeddings
    with torch.no_grad():
        embeddings = model(images, labels=None)

    print(f"    Embeddings Shape: {embeddings.shape}")
    assert embeddings.shape == (
        Config.BATCH_SIZE,
        Config.EMBEDDING_SIZE,
    ), "Incorrect embedding shape"

    # Test Training Mode (With Labels) -> Should return ArcFace Logits
    # Note: ArcFace head requires gradients usually, but we just check shape here
    model.train()
    # Create dummy labels on device
    dummy_labels = torch.randint(0, num_classes, (Config.BATCH_SIZE,)).to(Config.DEVICE)
    logits = model(images, dummy_labels)

    print(f"    Logits Shape: {logits.shape}")
    assert logits.shape == (Config.BATCH_SIZE, num_classes), "Incorrect logits shape"

    # ------------------------------------------------------------------
    # 4. Loss Function Verification
    # ------------------------------------------------------------------
    print("\n[4] Testing ArcFace Loss...")

    criterion = ArcFaceLoss()
    loss = criterion(logits, dummy_labels)

    print(f"    Loss Value: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    # ------------------------------------------------------------------
    # 5. Re-ranking Module Verification
    # ------------------------------------------------------------------
    print("\n[5] Testing Re-ranking Module...")

    # Simulate Query (N=5) and Gallery (M=20) features
    N, M, D = 5, 20, 512
    query_feats = torch.randn(N, D).numpy()
    gallery_feats = torch.randn(M, D).numpy()

    # Run re-ranking
    dist_matrix = re_ranking(query_feats, gallery_feats, k1=5, k2=3, lambda_value=0.5)

    print(f"    Distance Matrix Shape: {dist_matrix.shape}")
    assert dist_matrix.shape == (N, M), "Distance matrix shape mismatch"
    assert np.all(dist_matrix >= 0), "Distances should be non-negative"

    # ------------------------------------------------------------------
    # 6. Training Engine Execution
    # ------------------------------------------------------------------
    print("\n[6] Running Training Loop (1 Epoch)...")

    # Initialize Engine
    engine = WhaleEngine(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        label_encoder=label_encoder,
    )

    # Run Fit
    engine.fit(num_epochs=Config.NUM_EPOCHS)

    # Verify model checkpoint creation
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"    Success: Model saved to {Config.MODEL_SAVE_PATH}")
    else:
        # If validation score didn't improve (possible in random init), manually save for next step
        print(
            "    Note: Best model not saved by loop (score didn't improve). Saving manually."
        )
        torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # ------------------------------------------------------------------
    # 7. Inference Pipeline Execution
    # ------------------------------------------------------------------
    print("\n[7] Running Inference Pipeline...")

    # We call the high-level inference function
    # Note: This function re-loads data internally. We rely on the cache we just built.
    generate_predictions(load_cached_data=True, debug=True)

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"    Success: Submission generated with {len(df_sub)} rows.")
        print(f"    Sample:\n{df_sub.head(2)}")

        # Verify format
        assert (
            "Image" in df_sub.columns and "Id" in df_sub.columns
        ), "Missing columns in submission"
        assert len(df_sub) > 0, "Submission file is empty"
    else:
        raise FileNotFoundError("Submission file was not generated.")

    # ------------------------------------------------------------------
    # 8. Metric Utility Verification
    # ------------------------------------------------------------------
    print("\n[8] Verifying MAP@5 Metric Logic...")

    # Case 1: Perfect prediction
    preds_perfect = [["w_1", "w_2", "w_3", "w_4", "w_5"]]
    truth_perfect = ["w_1"]
    score_1 = map_at_5(preds_perfect, truth_perfect)

    # Case 2: Correct at rank 2
    preds_rank2 = [["w_2", "w_1", "w_3", "w_4", "w_5"]]
    truth_rank2 = ["w_1"]
    score_2 = map_at_5(preds_rank2, truth_rank2)

    # Case 3: Not in top 5
    preds_fail = [["w_2", "w_3", "w_4", "w_5", "w_6"]]
    truth_fail = ["w_1"]
    score_3 = map_at_5(preds_fail, truth_fail)

    print(f"    Score (Rank 1): {score_1} (Expected 1.0)")
    print(f"    Score (Rank 2): {score_2} (Expected 0.5)")
    print(f"    Score (Fail):   {score_3} (Expected 0.0)")

    assert score_1 == 1.0
    assert score_2 == 0.5
    assert score_3 == 0.0

    print("\n============================================================")
    print("      Demo Completed Successfully")
    print("============================================================")


if __name__ == "__main__":
    run_demo()
