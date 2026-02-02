import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import joblib

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.data import get_class_names, get_data_loaders, DogDataset
from library.modeling import load_feature_extractor
from library.processing import process_stream, extract_features_for_stream
from library.training import run_training, train_classifier, optimize_ensemble_weights


def run_demo():
    print("Starting Demo Run...")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Define demo directories
    demo_work_dir = "./working/demo_run"
    demo_meta_dir = "./working/demo_metadata"
    os.makedirs(demo_work_dir, exist_ok=True)
    os.makedirs(demo_meta_dir, exist_ok=True)

    # Override Config paths to isolate the demo
    Config.WORKING_DIR = demo_work_dir
    Config.SUBMISSION_DIR = demo_work_dir
    Config.SUBMISSION_PATH = os.path.join(demo_work_dir, "submission.csv")

    # Override Cache Paths to prevent loading existing full-dataset caches
    Config.STREAM_A_TRAIN_EMB = os.path.join(demo_work_dir, "stream_a_train_emb.npy")
    Config.STREAM_A_VAL_EMB = os.path.join(demo_work_dir, "stream_a_val_emb.npy")
    Config.STREAM_A_TEST_EMB = os.path.join(demo_work_dir, "stream_a_test_emb.npy")
    Config.STREAM_A_MODEL = os.path.join(demo_work_dir, "stream_a_logreg.joblib")

    Config.STREAM_B_TRAIN_EMB = os.path.join(demo_work_dir, "stream_b_train_emb.npy")
    Config.STREAM_B_VAL_EMB = os.path.join(demo_work_dir, "stream_b_val_emb.npy")
    Config.STREAM_B_TEST_EMB = os.path.join(demo_work_dir, "stream_b_test_emb.npy")
    Config.STREAM_B_MODEL = os.path.join(demo_work_dir, "stream_b_logreg.joblib")

    Config.TRAIN_LABELS_CACHE = os.path.join(demo_work_dir, "train_labels.npy")
    Config.VAL_LABELS_CACHE = os.path.join(demo_work_dir, "val_labels.npy")
    Config.TEST_IDS_CACHE = os.path.join(demo_work_dir, "test_ids.npy")

    # Override Model Configs for Speed
    # Using 'resnet18' as a lightweight proxy for the heavy models
    Config.MODEL_A_NAME = "resnet18"
    Config.MODEL_B_NAME = "resnet18"

    # Override Hyperparameters for rapid execution
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2
    Config.LOGREG_PARAMS["max_iter"] = 10
    Config.LOGREG_PARAMS["Cs"] = 2
    Config.LOGREG_PARAMS["cv"] = 2

    # Create Subset Metadata
    print("    Creating subset metadata...")
    # Read original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Filter for top 5 breeds to ensure we have enough samples for CV
    top_breeds = orig_train["breed"].value_counts().head(5).index.tolist()

    # Sample data: 4 samples per breed for train, 2 for val
    subset_train = (
        orig_train[orig_train["breed"].isin(top_breeds)]
        .groupby("breed")
        .head(4)
        .reset_index(drop=True)
    )
    subset_val = (
        orig_val[orig_val["breed"].isin(top_breeds)]
        .groupby("breed")
        .head(2)
        .reset_index(drop=True)
    )
    # Take 10 random test images
    subset_test = orig_test.head(10).reset_index(drop=True)

    # Save subsets
    demo_train_path = os.path.join(demo_meta_dir, "train.csv")
    demo_val_path = os.path.join(demo_meta_dir, "val.csv")
    demo_test_path = os.path.join(demo_meta_dir, "test.csv")

    subset_train.to_csv(demo_train_path, index=False)
    subset_val.to_csv(demo_val_path, index=False)
    subset_test.to_csv(demo_test_path, index=False)

    # Update Config to point to these new subset files
    Config.TRAIN_METADATA_PATH = demo_train_path
    Config.VAL_METADATA_PATH = demo_val_path
    Config.TEST_METADATA_PATH = demo_test_path

    print(
        f"    Subset sizes: Train={len(subset_train)}, Val={len(subset_val)}, Test={len(subset_test)}"
    )

    # -------------------------------------------------------------------------
    # 2. Demonstrate Data Loading
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Check class names retrieval
    class_names = get_class_names()
    print(f"    Number of classes in subset: {len(class_names)}")
    assert len(class_names) == 5, f"Expected 5 classes, got {len(class_names)}"

    # Get DataLoaders for Stream A
    train_loader, val_loader, test_loader, class_to_idx = get_data_loaders(
        "stream_a", batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # Check one batch
    batch = next(iter(train_loader))
    print("    Batch keys:", batch.keys())
    assert "global" in batch and "standard" in batch and "local" in batch
    assert "label" in batch
    assert "id" in batch

    # Verify tensor shapes
    # ResNet18 inputs should be (B, 3, H, W). Batch size is 8.
    img_global = batch["global"]
    assert img_global.shape == (
        8,
        3,
        224,
        224,
    ), f"Unexpected global shape: {img_global.shape}"
    print("    Data Loading verification passed.")

    # -------------------------------------------------------------------------
    # 3. Demonstrate Modeling (Feature Extractor)
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Feature Extractor Initialization...")

    model = load_feature_extractor(Config.MODEL_A_NAME, Config.DEVICE)

    # Check if parameters are frozen
    params_requiring_grad = [p for p in model.parameters() if p.requires_grad]
    assert (
        len(params_requiring_grad) == 0
    ), "Model parameters should be frozen (requires_grad=False)"

    # Check forward pass output shape
    # ResNet18 feature dim is 512
    dummy_input = torch.randn(2, 3, 224, 224).to(Config.DEVICE)
    with torch.no_grad():
        features = model(dummy_input)

    print(f"    Output feature shape: {features.shape}")
    assert features.shape == (2, 512), f"Expected (2, 512), got {features.shape}"
    print("    Modeling verification passed.")

    # Clean up model to free memory
    del model
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 4. Demonstrate Processing (Feature Extraction Loop)
    # -------------------------------------------------------------------------
    print("\n[4] Running Feature Extraction (Stream A & B)...")

    # Process Stream A
    # We force recomputation (load_cached_data=False) to test the extraction logic
    data_stream_a = process_stream("stream_a", load_cached_data=False)

    train_emb_a, train_lbl_a = data_stream_a["train"]
    print(f"    Stream A Train Embeddings: {train_emb_a.shape}")

    # Expected shape: (N_train, 512 * 3) = (20, 1536)
    # 3 views (global, standard, local) concatenated
    expected_dim = 512 * 3
    assert (
        train_emb_a.shape[1] == expected_dim
    ), f"Expected dim {expected_dim}, got {train_emb_a.shape[1]}"
    assert len(train_lbl_a) == len(subset_train)

    # Process Stream B
    data_stream_b = process_stream("stream_b", load_cached_data=False)
    train_emb_b, _ = data_stream_b["train"]
    print(f"    Stream B Train Embeddings: {train_emb_b.shape}")
    assert train_emb_b.shape[1] == expected_dim

    print("    Feature Extraction verification passed.")

    # -------------------------------------------------------------------------
    # 5. Demonstrate Training & Submission
    # -------------------------------------------------------------------------
    print("\n[5] Running Training and Submission Generation...")

    # Run the full training pipeline (Train LogReg, Optimize Ensemble, Predict)
    run_training(data_stream_a, data_stream_b, load_cached_models=False)

    # Verify Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission file shape: {sub_df.shape}")

    # Check rows = test set size
    assert len(sub_df) == len(
        subset_test
    ), f"Expected {len(subset_test)} rows, got {len(sub_df)}"

    # Check columns = id + classes
    expected_cols = ["id"] + class_names
    assert (
        list(sub_df.columns) == expected_cols
    ), "Submission columns do not match expected classes."

    # Check probabilities sum to 1 (approx)
    prob_sums = sub_df[class_names].sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1."

    print("    Training and Submission verification passed.")

    print("\nDemo Run Completed Successfully!")


if __name__ == "__main__":
    run_demo()
