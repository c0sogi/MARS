import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import seed_everything, save_submission
from library.data import get_dataloaders
from library.feature_extraction import DualStreamExtractor
from library.ensemble import BaggedLDAPipeline


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("Initializing Configuration...")

    # Modify Config for rapid demonstration
    Config.MAX_SAMPLES = 40  # Limit dataset size significantly for speed
    Config.N_ESTIMATORS = 2  # Reduce ensemble size to minimal
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script

    # Set seeds for reproducibility
    seed_everything(42)

    print(f"Device: {Config.DEVICE}")
    print(f"Max Samples: {Config.MAX_SAMPLES}")
    print(f"Estimators: {Config.N_ESTIMATORS}")

    # ==========================================
    # 2. Data Loading Demonstration
    # ==========================================
    print("\n[Step 1] Loading Data...")

    # Explicitly pass max_samples to override default arg evaluation
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        max_samples=Config.MAX_SAMPLES
    )

    print(f"Number of classes: {len(classes)}")
    print(f"Train batches: {len(train_loader)}")

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))
    stacked_views, tabular, label, image_id = batch

    # Verify Shapes
    # stacked_views: (B, 4, 3, 224, 224) -> (Batch, Views, Channels, Height, Width)
    assert stacked_views.dim() == 5, f"Expected 5D views, got {stacked_views.shape}"
    assert stacked_views.shape[1] == 4, "Expected 4 rotated views"
    assert stacked_views.shape[2] == 3, "Expected 3 RGB channels"
    assert stacked_views.shape[3] == Config.IMAGE_SIZE

    # tabular: (B, 192) -> (Batch, Features)
    assert tabular.dim() == 2
    assert tabular.shape[1] == 192, f"Expected 192 features, got {tabular.shape[1]}"

    print("Data structure verified successfully.")

    # ==========================================
    # 3. Feature Extraction Demonstration
    # ==========================================
    print("\n[Step 2] Extracting Features (DualStreamExtractor)...")

    # Initialize the extractor (loads DINOv2 and ConvNeXt)
    extractor = DualStreamExtractor()

    # Extract from the small training subset
    # This runs the forward pass and aggregates features across the 4 views
    X_dino, X_conv, X_tab, y, ids = extractor.extract_features(train_loader)

    print(f"Extracted DINOv2 features: {X_dino.shape}")
    print(f"Extracted ConvNeXt features: {X_conv.shape}")
    print(f"Extracted Tabular features: {X_tab.shape}")
    print(f"Labels shape: {y.shape}")

    # Verify consistency
    n_samples = len(y)
    assert X_dino.shape[0] == n_samples
    assert X_conv.shape[0] == n_samples
    assert X_tab.shape[0] == n_samples
    assert X_dino.ndim == 2

    # ==========================================
    # 4. Pipeline Training Demonstration
    # ==========================================
    print("\n[Step 3] Training Ensemble (BaggedLDAPipeline)...")

    pipeline = BaggedLDAPipeline()

    # For demonstration purposes, we map the 99 classes to a smaller set (e.g., 5 classes).
    # With only 40 samples, many of the 99 classes would be singletons or missing,
    # which can cause instability in LDA covariance estimation during this tiny demo.
    # In a full run, we would use the real 'y'.
    print("Mapping targets to 5 pseudo-classes for stable demo training...")
    y_demo = y % 5

    pipeline.fit(X_dino, X_conv, X_tab, y_demo)

    # Verify pipeline structure
    assert len(pipeline.estimators) == Config.N_ESTIMATORS
    print("Pipeline fitted successfully.")

    # ==========================================
    # 5. Prediction Demonstration
    # ==========================================
    print("\n[Step 4] Generating Predictions...")

    # Predict on the same training data for verification
    probs = pipeline.predict_proba(X_dino, X_conv, X_tab)

    print(f"Probabilities shape: {probs.shape}")
    print(f"Sample probabilities (first row): {probs[0]}")

    # Verify Probabilities
    # 1. Should sum to ~1 (tolerance for float precision)
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1"

    # 2. Should be within [0, 1]
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0

    # 3. Output dimensions should match number of classes seen during fit (5 in this demo)
    assert probs.shape[1] == 5

    # ==========================================
    # 6. Submission File Generation
    # ==========================================
    print("\n[Step 5] Saving Submission...")

    # Create a dummy submission for the demo
    # We use the real class names list to match the required format
    # We create random probabilities for 5 test IDs

    demo_test_ids = [101, 102, 103, 104, 105]
    n_test = len(demo_test_ids)
    n_classes = len(classes)

    # Random probs
    demo_probs = np.random.rand(n_test, n_classes)
    # Normalize to sum to 1
    demo_probs = demo_probs / demo_probs.sum(axis=1, keepdims=True)

    output_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    save_submission(demo_probs, demo_test_ids, classes, output_path=output_path)

    # Verify file existence and format
    assert os.path.exists(output_path)
    print(f"Submission file created at {output_path}")

    # Read back to check format
    df_sub = pd.read_csv(output_path)
    print(f"Submission DataFrame shape: {df_sub.shape}")
    assert df_sub.shape[1] == n_classes + 1  # classes + id column
    assert "id" in df_sub.columns

    print("\nAll demonstration steps completed successfully!")


if __name__ == "__main__":
    run_demo()
