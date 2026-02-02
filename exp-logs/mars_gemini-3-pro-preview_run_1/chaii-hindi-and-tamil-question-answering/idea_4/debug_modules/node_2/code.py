import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Import library modules
from library.config import Config
from library.utils import set_seed, jaccard, compute_average_jaccard, cleanup
from library.data_loader import (
    get_train_data,
    get_val_data,
    get_test_data,
    get_tokenizer,
    prepare_test_features,
    QADataset,
)
from library.model_arch import get_model
from library.train_runner import run_training
from library.inference_engine import (
    get_test_features_cached,
    get_fold_logits,
    ensemble_and_postprocess,
)


def main():
    print("==== Starting Demonstration Script ====")

    # --------------------------------------------------------------------------
    # 1. Configuration Setup for Speed & Demo
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Modify Config class attributes directly to optimize for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Use very small subset
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 4
    Config.N_FOLDS = 1  # Only demonstrate 1 fold
    Config.IDEA_NAME = "demo_run"
    Config.WORKING_DIR = os.path.join("./working", Config.IDEA_NAME)
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up previous demo run if exists to ensure fresh start
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    Config.setup()
    set_seed(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # --------------------------------------------------------------------------
    # 2. Data Loading & Processing Verification
    # --------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading Pipeline...")

    # Load Training Data
    print("Loading training data (subset)...")
    train_dataset = get_train_data(load_cached_data=False, debug=True)

    # Verify Dataset properties
    assert isinstance(
        train_dataset, QADataset
    ), "get_train_data should return a QADataset"
    assert len(train_dataset) > 0, "Training dataset should not be empty"

    # Verify a single sample
    sample = train_dataset[0]
    required_keys = ["input_ids", "attention_mask", "start_positions", "end_positions"]
    for key in required_keys:
        assert key in sample, f"Sample missing key: {key}"
        assert torch.is_tensor(sample[key]), f"{key} should be a tensor"

    print(f"Train dataset size: {len(train_dataset)}")
    print("Sample input_ids shape:", sample["input_ids"].shape)

    # Load Validation Data
    print("Loading validation data (subset)...")
    val_dataset = get_val_data(load_cached_data=False, debug=True)
    assert len(val_dataset) > 0
    print(f"Val dataset size: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues in simple script
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # --------------------------------------------------------------------------
    # 3. Model Initialization & Training Loop Verification
    # --------------------------------------------------------------------------
    print("\n[3] Verifying Model and Training Runner...")

    model = get_model()
    assert model is not None, "Model initialization failed"

    # Setup Optimizer and Scheduler for the demo
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    save_path = os.path.join(Config.WORKING_DIR, "fold_0_best_model.pth")

    print("Starting training loop (1 epoch)...")
    trained_model = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,  # Note: val_loader here is processed as inference data (no labels usually)
        # However, run_training expects labels for validation loss.
        # In this library structure, get_val_data returns 'eval' mode without labels
        # if using prepare_test_features.
        # To make run_training work for this demo, we need to ensure val_loader has labels
        # or we skip validation loss check if it crashes.
        # Actually, looking at library code: prepare_test_features does NOT return start/end positions.
        # validate_one_epoch checks `if "start_positions" in batch`.
        # So validate_one_epoch will return 0.0 loss if no labels.
        # This is fine for the demo execution flow.
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        epochs=Config.EPOCHS,
        save_path=save_path,
        patience=1,
    )

    assert os.path.exists(save_path), "Model checkpoint was not saved!"
    print(f"Model successfully saved to {save_path}")

    # Free memory
    del model, trained_model, optimizer, scheduler
    cleanup()

    # --------------------------------------------------------------------------
    # 4. Inference Pipeline Verification
    # --------------------------------------------------------------------------
    print("\n[4] Verifying Inference Pipeline...")

    # Load raw test data (subset)
    test_df_raw = pd.read_csv(Config.TEST_CSV).head(10)
    tokenizer = get_tokenizer()

    # Process features
    print("Processing test features...")
    # We force reprocessing to test the logic
    test_features = get_test_features_cached(
        test_df_raw, tokenizer, load_cached_data=False
    )

    assert "input_ids" in test_features.columns
    assert "offset_mapping" in test_features.columns
    assert "example_id" in test_features.columns

    # Compute Logits using the saved model from step 3
    print("Computing logits...")
    start_logits, end_logits = get_fold_logits(
        fold_idx=0, features_df=test_features, device=Config.DEVICE
    )

    assert (
        start_logits is not None and end_logits is not None
    ), "Logits computation failed"
    assert start_logits.shape == end_logits.shape
    assert len(start_logits) == len(test_features)
    print(f"Logits shape: {start_logits.shape}")

    # Post-processing
    print("Running ensemble and post-processing...")
    # We pass [0] as fold_indices because we only have fold_0 model
    submission_df = ensemble_and_postprocess(
        test_df_raw, test_features, fold_indices=[0]
    )

    # Verify Submission Format
    print("Verifying submission format...")
    assert isinstance(submission_df, pd.DataFrame)
    assert list(submission_df.columns) == ["id", "PredictionString"]
    assert len(submission_df) == len(test_df_raw)

    # Check content
    print("Sample Predictions:")
    print(submission_df.head())

    # Ensure predictions are strings and quoted (heuristic check)
    sample_pred = submission_df.iloc[0]["PredictionString"]
    assert isinstance(sample_pred, str)
    # The prompt requires quoted text, e.g., "answer".
    # Our post-processing adds quotes.
    assert sample_pred.startswith('"') and sample_pred.endswith(
        '"'
    ), f"Prediction should be quoted, got: {sample_pred}"

    # --------------------------------------------------------------------------
    # 5. Metric Verification
    # --------------------------------------------------------------------------
    print("\n[5] Verifying Metric Functions...")

    # Test Jaccard
    s1 = "India is a country"
    s2 = "India country"
    score = jaccard(s1, s2)
    # intersection: {india, country} (size 2)
    # union: {india, is, a, country} (size 4)
    # jaccard: 2/4 = 0.5
    print(f"Jaccard('{s1}', '{s2}') = {score}")
    assert abs(score - 0.5) < 1e-6, "Jaccard calculation incorrect"

    s3 = "Different text"
    score_zero = jaccard(s1, s3)
    assert score_zero == 0.0, "Jaccard should be 0 for disjoint sets"

    # Test Average Jaccard
    gts = ["a b", "c"]
    preds = ["a b", "d"]
    # 1.0 + 0.0 / 2 = 0.5
    avg_score = compute_average_jaccard(gts, preds)
    assert abs(avg_score - 0.5) < 1e-6, "Average Jaccard incorrect"

    print("\n==== Demonstration Complete: All checks passed ====")


if __name__ == "__main__":
    main()
