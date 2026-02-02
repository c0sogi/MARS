import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer

# Import from provided library
from library.config import GlobalConfig, MPNET_CONFIG
from library.utils import seed_everything, compute_spearman_metric
from library.dataset import StackExchangeDataset, get_dataloader
from library.modeling import SegmentAwareCrossEncoder
from library.engine import train_one_epoch, validate, extract_all_features
from library.ridge_head import train_ridge_head, predict_ridge


def run_demonstration():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    print(">>> Step 1: Setup & Configuration")
    seed_everything(42)

    # Define paths for demo data
    demo_dir = "./working/demo_run"
    os.makedirs(demo_dir, exist_ok=True)

    demo_train_path = os.path.join(demo_dir, "train_demo.csv")
    demo_test_path = os.path.join(demo_dir, "test_demo.csv")

    # Create a tiny subset of data for demonstration purposes (Speed Optimization)
    # We read from the metadata files provided in the environment
    print(f"Creating demo datasets in {demo_dir}...")
    full_train_df = pd.read_csv(GlobalConfig.TRAIN_METADATA_PATH)
    full_test_df = pd.read_csv(GlobalConfig.TEST_METADATA_PATH)

    # Take top 16 rows for train and 8 for test
    demo_train_df = full_train_df.head(16).copy()
    demo_test_df = full_test_df.head(8).copy()

    demo_train_df.to_csv(demo_train_path, index=False)
    demo_test_df.to_csv(demo_test_path, index=False)

    # Configuration overrides for speed
    BATCH_SIZE = 4
    MAX_LENGTH = 64  # Reduced from 512 for speed
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {DEVICE}")

    # --------------------------------------------------------------------------
    # 2. Dataset & DataLoader
    # --------------------------------------------------------------------------
    print("\n>>> Step 2: Dataset & DataLoader Verification")

    tokenizer = AutoTokenizer.from_pretrained(MPNET_CONFIG.model_name)

    # Instantiate Dataset
    train_dataset = StackExchangeDataset(
        data_path=demo_train_path,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        is_test=False,
    )

    # Verify __getitem__
    sample = train_dataset[0]
    print("Sample keys:", sample.keys())

    # Assertions for shapes
    assert sample["input_ids"].shape == (MAX_LENGTH,), "Input IDs shape mismatch"
    assert sample["attention_mask"].shape == (
        MAX_LENGTH,
    ), "Attention Mask shape mismatch"
    assert sample["segment_mask"].shape == (MAX_LENGTH,), "Segment Mask shape mismatch"
    assert sample["targets"].shape == (
        30,
    ), "Targets shape mismatch (should be 30 labels)"

    # Verify Segment Mask Logic
    # 0: Pad/Special, 1: Question, 2: Answer
    unique_segments = torch.unique(sample["segment_mask"]).tolist()
    print(f"Unique segment IDs in sample: {unique_segments}")
    # We expect at least some 0s (special tokens) and 1s (question).
    # 2s (answer) might be missing if answer text is empty, but usually present.
    assert 0 in unique_segments, "Segment mask missing special/pad tokens (0)"
    assert 1 in unique_segments, "Segment mask missing question tokens (1)"

    # Instantiate DataLoader
    train_loader = get_dataloader(
        data_path=demo_train_path,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        is_test=False,
        shuffle=True,
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    print(
        f"Batch shapes -> Input IDs: {batch['input_ids'].shape}, Targets: {batch['targets'].shape}"
    )
    assert batch["input_ids"].shape == (BATCH_SIZE, MAX_LENGTH)
    assert batch["targets"].shape == (BATCH_SIZE, 30)

    # --------------------------------------------------------------------------
    # 3. Modeling (Backbone)
    # --------------------------------------------------------------------------
    print("\n>>> Step 3: Model Instantiation & Forward Pass")

    model = SegmentAwareCrossEncoder(model_name=MPNET_CONFIG.model_name, num_labels=30)
    model.to(DEVICE)

    # Move batch to device
    b_input_ids = batch["input_ids"].to(DEVICE)
    b_att_mask = batch["attention_mask"].to(DEVICE)
    b_seg_mask = batch["segment_mask"].to(DEVICE)

    # Forward pass
    logits, features = model(b_input_ids, b_att_mask, b_seg_mask)

    print(f"Logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")

    # Assertions
    assert logits.shape == (BATCH_SIZE, 30), "Logits shape incorrect"
    # Feature dim is 4 * hidden_size. MPNet hidden size is 768. 4*768 = 3072.
    expected_feat_dim = 4 * model.config.hidden_size
    assert features.shape == (
        BATCH_SIZE,
        expected_feat_dim,
    ), f"Features shape incorrect. Expected {expected_feat_dim}"

    # --------------------------------------------------------------------------
    # 4. Training Engine (Train & Validate)
    # --------------------------------------------------------------------------
    print("\n>>> Step 4: Training Engine Demo")

    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    # Train one epoch (on the tiny demo dataset)
    avg_loss = train_one_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        scheduler=None,
        device=DEVICE,
        criterion=criterion,
    )
    print(f"Training Loss (1 epoch): {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Training loss is NaN"

    # Validate
    # We use the same loader for validation just to demonstrate the function
    val_loss, val_spearman = validate(
        model=model, dataloader=train_loader, device=DEVICE, criterion=criterion
    )
    print(f"Validation Loss: {val_loss:.4f}, Spearman: {val_spearman:.4f}")
    assert isinstance(val_spearman, float), "Spearman score is not a float"

    # --------------------------------------------------------------------------
    # 5. Feature Extraction
    # --------------------------------------------------------------------------
    print("\n>>> Step 5: Feature Extraction")

    # Extract features for the demo train set
    extracted_features = extract_all_features(model, train_loader, DEVICE)
    print(f"Extracted Features Shape: {extracted_features.shape}")

    assert extracted_features.shape[0] == len(
        demo_train_df
    ), "Number of extracted samples mismatch"
    assert (
        extracted_features.shape[1] == expected_feat_dim
    ), "Feature dimension mismatch"

    # --------------------------------------------------------------------------
    # 6. Ridge Head (Training & Prediction)
    # --------------------------------------------------------------------------
    print("\n>>> Step 6: Ridge Regression Head")

    # Prepare targets
    train_targets = demo_train_df[GlobalConfig.TARGET_COLS].values

    # Train Ridge
    ridge_model_path = os.path.join(demo_dir, "ridge_model.joblib")
    ridge_model = train_ridge_head(
        train_features=extracted_features,
        train_targets=train_targets,
        model_path=ridge_model_path,
        alphas=[0.1, 1.0],  # Reduced alphas for speed
    )

    # Predict (using the same features for demonstration)
    preds = predict_ridge(extracted_features, ridge_model_path)
    print(f"Predictions Shape: {preds.shape}")
    print(f"Predictions Sample (First row, first 5 cols): {preds[0, :5]}")

    # Assertions
    assert preds.shape == train_targets.shape, "Prediction shape mismatch"
    assert (preds >= 0.0).all() and (
        preds <= 1.0
    ).all(), "Predictions outside [0, 1] range"

    # --------------------------------------------------------------------------
    # 7. Metric Verification
    # --------------------------------------------------------------------------
    print("\n>>> Step 7: Metric Verification")

    # Create synthetic ground truth and predictions
    y_true = np.random.rand(10, 30)
    y_pred = y_true + np.random.normal(0, 0.1, (10, 30))

    score = compute_spearman_metric(y_true, y_pred)
    print(f"Calculated Spearman Score: {score:.4f}")
    assert -1.0 <= score <= 1.0, "Spearman score out of bounds"

    print("\n>>> Demonstration Completed Successfully!")


if __name__ == "__main__":
    run_demonstration()
