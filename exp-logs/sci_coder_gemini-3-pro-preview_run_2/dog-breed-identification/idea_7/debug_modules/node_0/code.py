import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import provided library components
from library.config import Config
from library.data_utils import get_dataloaders
from library.model_factory import create_backbone
from library.embedding_engine import extract_embeddings
from library.classifier_engine import train_and_evaluate, generate_submission


def main():
    print("Starting Dog Breed Prediction Demo...")

    # 1. Configure for Speed (Demo Mode)
    # We modify the Config class attributes directly to affect downstream modules
    print("\n[1] Configuring environment for fast demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Small subset for speed
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2
    Config.CV_FOLDS = 2  # Minimal folds for Cross-Validation
    Config.LOGREG_MAX_ITER = 50  # Limit iterations for speed

    # Setup a specific working directory for this demo to avoid conflicts
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    Config.WORKING_DIR = demo_dir

    # Define cache paths for this run
    train_feat_path = os.path.join(demo_dir, "train_emb.npy")
    train_lbl_path = os.path.join(demo_dir, "train_lbl.npy")
    val_feat_path = os.path.join(demo_dir, "val_emb.npy")
    val_lbl_path = os.path.join(demo_dir, "val_lbl.npy")
    test_feat_path = os.path.join(demo_dir, "test_emb.npy")
    test_ids_path = os.path.join(demo_dir, "test_ids.npy")
    submission_path = os.path.join(demo_dir, "submission.csv")

    # 2. Data Loading
    print("\n[2] Loading DataLoaders...")
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        debug=Config.DEBUG
    )

    # Verification
    print(f"    Train batches: {len(train_loader)}")
    print(f"    Val batches:   {len(val_loader)}")
    print(f"    Test batches:  {len(test_loader)}")
    assert len(train_loader) > 0, "Train loader is empty"

    # Peek at a batch to verify structure
    sample_imgs, sample_targets = next(iter(train_loader))
    assert isinstance(sample_imgs, list), "Batch images should be a list (PIL images)"
    assert isinstance(sample_targets, torch.Tensor), "Batch targets should be a Tensor"
    print("    Data loading verification passed.")

    # 3. Model Initialization
    print("\n[3] Initializing Backbone Model...")
    device = Config.DEVICE
    print(f"    Device: {device}")
    model = create_backbone()
    model.to(device)

    # Verification
    assert isinstance(model, torch.nn.Module), "Model is not a PyTorch Module"
    print("    Model initialized successfully.")

    # 4. Feature Extraction
    # The pipeline extracts features from 3 views (Global, Standard, Local)
    # and 2 stages (Stage 3, Stage 4) of ConvNeXt Large.
    # Expected Dimension:
    #   Global:   Stage4(1536) + Stage3(768)
    #   Standard: Stage4(1536) + Stage3(768)
    #   Local:    Stage4(1536) + Stage3(768)
    #   Total:    (1536 + 768) * 3 = 2304 * 3 = 6912
    EXPECTED_DIM = 6912

    print("\n[4] Extracting Features (Train)...")
    train_feats, train_lbls = extract_embeddings(
        train_loader,
        model,
        device,
        train_feat_path,
        train_lbl_path,
        load_cached_data=False,
    )
    assert (
        train_feats.shape[1] == EXPECTED_DIM
    ), f"Expected {EXPECTED_DIM} features, got {train_feats.shape[1]}"
    assert len(train_feats) == len(train_lbls)

    print("\n[4] Extracting Features (Val)...")
    val_feats, val_lbls = extract_embeddings(
        val_loader, model, device, val_feat_path, val_lbl_path, load_cached_data=False
    )
    assert val_feats.shape[1] == EXPECTED_DIM

    print("\n[4] Extracting Features (Test)...")
    # Note: test_loader yields (images, ids), so the second return value is ids
    test_feats, test_ids = extract_embeddings(
        test_loader,
        model,
        device,
        test_feat_path,
        test_ids_path,
        load_cached_data=False,
    )
    assert test_feats.shape[1] == EXPECTED_DIM
    print("    Feature extraction verification passed.")

    # 5. Classifier Training
    print("\n[5] Training Classifier...")
    # We use the extracted features to train a Logistic Regression model
    clf_model, val_metric = train_and_evaluate(
        train_feats, train_lbls, val_feats, val_lbls
    )

    print(f"    Validation Log Loss: {val_metric:.4f}")
    assert isinstance(val_metric, float), "Metric should be a float"
    # Log loss is non-negative
    assert val_metric >= 0, "Log loss cannot be negative"

    # 6. Submission Generation
    print("\n[6] Generating Submission...")
    generate_submission(clf_model, test_feats, test_ids, submission_path)

    # Verification of output file
    assert os.path.exists(submission_path), "Submission file was not created"

    sub_df = pd.read_csv(submission_path)
    print(f"    Submission shape: {sub_df.shape}")

    # Check columns
    # 120 breeds + 1 'id' column = 121 columns
    assert sub_df.shape[1] == 121, f"Expected 121 columns, found {sub_df.shape[1]}"
    assert "id" in sub_df.columns, "Submission missing 'id' column"

    # Check rows
    # Should match Config.DEBUG_SAMPLE_SIZE (50)
    assert (
        len(sub_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} rows, found {len(sub_df)}"

    print("\nDemo completed successfully. All logic verified.")


if __name__ == "__main__":
    main()
