import os
import shutil
import numpy as np
import pandas as pd
import torch

# Import provided library components
from library.config import Config
from library.utils import seed_everything
from library.data_loader import load_tabular_data, get_data_loaders
from library.preprocessing import FeaturePipeline
from library.ensemble_model import RandomSubspaceLDA, train_evaluate_predict


def run_demo():
    print("==================================================")
    print("   Leaf Species Classification: Pipeline Demo     ")
    print("==================================================")

    # ---------------------------------------------------------
    # 1. Configuration Setup
    # ---------------------------------------------------------
    # Define a unique directory for this demo to avoid cache conflicts
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config parameters for the demonstration
    print(f"Setting up configuration in {demo_dir}...")
    Config.CACHE_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission_demo.csv")

    # Reduce complexity for speed
    Config.N_ESTIMATORS = 5  # Reduced from 50
    Config.PCA_VARIANCE = 0.95  # Slightly lower variance retention for faster PCA

    # Ensure reproducibility
    seed_everything(Config.SEED)
    print(f"Device: {Config.DEVICE}")

    # ---------------------------------------------------------
    # 2. Data Loading Verification
    # ---------------------------------------------------------
    print("\n[Step 1] Verifying Data Loading...")

    # Test Tabular Data Loading
    print("  Loading tabular training data...")
    X_tab, y_tab, ids_tab = load_tabular_data("train", load_cached_data=False)

    print(f"    Tabular shape: {X_tab.shape}")
    # Verify shapes: 712 samples, 192 features (64 margin + 64 shape + 64 texture)
    assert X_tab.shape == (712, 192), f"Expected (712, 192), got {X_tab.shape}"
    assert len(y_tab) == 712

    # Test Image Data Loading
    print("  Initializing DataLoaders...")
    # Use small batch size for verification
    train_loader, _, _ = get_data_loaders(
        batch_size=4, num_workers=0, load_cached_data=False
    )

    # Fetch a single batch
    images, labels, ids = next(iter(train_loader))
    print(f"    Image batch shape: {images.shape}")

    # Verify shapes: (Batch, 4 Views, 3 Channels, 224 Height, 224 Width)
    assert images.shape == (
        4,
        4,
        3,
        224,
        224,
    ), f"Expected (4, 4, 3, 224, 224), got {images.shape}"
    assert labels.shape == (4,)

    # ---------------------------------------------------------
    # 3. Feature Pipeline Verification
    # ---------------------------------------------------------
    print("\n[Step 2] Verifying Feature Pipeline (Extraction + Preprocessing)...")
    pipeline = FeaturePipeline()

    # We request processed validation data.
    # The pipeline logic ensures transformers are fitted on training data first.
    # This will trigger:
    #   1. Extraction of Train features (DINOv2 + ConvNeXt)
    #   2. Fitting of PCA and QuantileTransformer
    #   3. Extraction of Val features
    #   4. Transformation and Fusion
    print("  Processing validation data (triggers training extraction/fit)...")
    X_val_proc, y_val_proc, ids_val_proc = pipeline.get_processed_data(
        "val", load_cached_data=False
    )

    print(f"    Processed Val Shape: {X_val_proc.shape}")

    # Verify dimensions
    # Rows: 179 validation samples
    # Cols: 192 tabular + PCA components (variable)
    assert X_val_proc.shape[0] == 179
    assert X_val_proc.shape[1] > 192, "Features should include fused image embeddings"

    # Retrieve processed training data (should use the cache generated in the previous step)
    print("  Retrieving processed training data (from cache)...")
    X_train_proc, y_train_proc, ids_train_proc = pipeline.get_processed_data(
        "train", load_cached_data=True
    )

    assert X_train_proc.shape[0] == 712
    assert (
        X_train_proc.shape[1] == X_val_proc.shape[1]
    ), "Feature dimension mismatch between train and val"

    # ---------------------------------------------------------
    # 4. Model Training Verification
    # ---------------------------------------------------------
    print("\n[Step 3] Verifying Random Subspace LDA Model...")

    model = RandomSubspaceLDA(
        n_estimators=Config.N_ESTIMATORS,
        subspace_fraction=0.5,
        random_state=Config.SEED,
    )

    print(f"  Fitting ensemble with {Config.N_ESTIMATORS} estimators...")
    model.fit(X_train_proc, y_train_proc)

    print("  Predicting on validation set...")
    probs = model.predict_proba(X_val_proc)
    preds = model.predict(X_val_proc)

    print(f"    Probabilities shape: {probs.shape}")

    # Verify outputs
    assert probs.shape == (
        179,
        99,
    ), "Probability matrix shape incorrect (N_samples, N_classes)"
    assert preds.shape == (179,), "Predictions shape incorrect"

    # Check probability validity
    assert np.all(probs >= 0) and np.all(
        probs <= 1.0 + 1e-9
    ), "Probabilities out of [0, 1] range"

    # ---------------------------------------------------------
    # 5. End-to-End Execution
    # ---------------------------------------------------------
    print("\n[Step 4] Running Full Pipeline (Train -> Eval -> Predict)...")

    # This function orchestrates the entire flow and saves the submission file.
    # We use load_cached_data=True to leverage the features we just extracted.
    train_evaluate_predict(load_cached_data=True)

    # Verify Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"\nSubmission generated successfully at: {Config.SUBMISSION_PATH}")
        print(df_sub.head(3))

        # Check submission format
        assert df_sub.shape == (99, 100), f"Expected (99, 100), got {df_sub.shape}"
        assert df_sub.columns[0] == "id", "First column must be 'id'"
        assert df_sub.iloc[:, 1:].min().min() >= 0, "Probabilities must be non-negative"
        assert df_sub.iloc[:, 1:].max().max() <= 1, "Probabilities must be <= 1"
        print("Verification Passed: Submission format is correct.")
    else:
        raise FileNotFoundError("Submission file was not created!")

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    run_demo()
