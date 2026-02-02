import os
import pandas as pd
import torch
import warnings
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import Dataset as HFDataset

# Import from the provided library
from library.config import Config
from library.utils import (
    set_seed,
    postprocess_qa_predictions,
    compute_metrics,
    save_submission,
)
from library.data import load_processed_data, QADataset
from library.model import get_tokenizer, get_model
from library.engine import train_fn, eval_fn

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Configuration Setup
    # Override Config for a fast demonstration run
    Config.DEBUG = True  # Use small subset (100 samples)
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.EVAL_BATCH_SIZE = 8

    # Ensure reproducibility
    set_seed(Config.SEED)
    print(f"Device: {Config.DEVICE}")
    print(f"Debug Mode: {Config.DEBUG}")

    # 2. Data Loading & Processing
    print("\n[1/5] Loading and Processing Data...")
    tokenizer = get_tokenizer()

    # Load processed features (tokenized and chunked)
    # forcing load_cached_data=False to ensure we process the debug subset freshly
    train_features_df = load_processed_data(
        tokenizer, split="train", load_cached_data=False
    )
    val_features_df = load_processed_data(
        tokenizer, split="val", load_cached_data=False
    )
    test_features_df = load_processed_data(
        tokenizer, split="test", load_cached_data=False
    )

    # Create PyTorch Datasets
    train_dataset = QADataset(train_features_df, mode="train")
    val_dataset = QADataset(
        val_features_df, mode="val"
    )  # mode='val' implies no labels needed for Dataset __getitem__
    test_dataset = QADataset(test_features_df, mode="val")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for simple script execution
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.EVAL_BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.EVAL_BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(f"Train features: {len(train_features_df)}")
    print(f"Val features: {len(val_features_df)}")

    # 3. Model Initialization
    print("\n[2/5] Initializing Model...")
    model = get_model(pretrained=True)
    model.to(Config.DEVICE)

    # Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    num_train_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
        num_training_steps=num_train_steps,
    )

    # 4. Training Loop
    print("\n[3/5] Starting Training...")
    for epoch in range(Config.EPOCHS):
        print(f"Epoch {epoch + 1}/{Config.EPOCHS}")
        train_loss = train_fn(train_loader, model, optimizer, Config.DEVICE, scheduler)
        print(f"Epoch {epoch + 1} Loss: {train_loss:.4f}")

    # 5. Validation & Evaluation
    print("\n[4/5] Validating...")
    # Get raw logits from model
    val_preds_tuple = eval_fn(val_loader, model, Config.DEVICE)

    # Load original validation data to map back to text
    # In Debug mode, we need to slice the original dataframe to match the subset logic
    # The load_processed_data function in Debug mode takes .head(DEBUG_SAMPLE_SIZE)
    df_val_raw = pd.read_csv(Config.VAL_PATH)
    if Config.DEBUG:
        df_val_raw = df_val_raw.head(Config.DEBUG_SAMPLE_SIZE)

    # Convert to HF Dataset for compatibility with postprocess_qa_predictions
    val_examples = HFDataset.from_pandas(df_val_raw)

    # Convert features DataFrame to list of dicts
    val_features_list = val_features_df.to_dict("records")

    # Post-process to get text answers
    final_val_predictions = postprocess_qa_predictions(
        examples=val_examples,
        features=val_features_list,
        predictions=val_preds_tuple,
        n_best_size=20,
        max_answer_length=30,
    )

    # Compute Metric
    val_score = compute_metrics(final_val_predictions, df_val_raw)
    print(f"Validation Jaccard Score: {val_score:.4f}")

    # Assertion to verify metric calculation
    assert isinstance(val_score, float), "Score must be a float"
    assert 0 <= val_score <= 1.0, "Jaccard score must be between 0 and 1"

    # 6. Inference on Test Set
    print("\n[5/5] Generating Test Predictions...")
    test_preds_tuple = eval_fn(test_loader, model, Config.DEVICE)

    # Load original test data
    df_test_raw = pd.read_csv(Config.TEST_PATH)
    if Config.DEBUG:
        df_test_raw = df_test_raw.head(Config.DEBUG_SAMPLE_SIZE)

    test_examples = HFDataset.from_pandas(df_test_raw)
    test_features_list = test_features_df.to_dict("records")

    final_test_predictions = postprocess_qa_predictions(
        examples=test_examples,
        features=test_features_list,
        predictions=test_preds_tuple,
    )

    # Save Submission
    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    save_submission(final_test_predictions, submission_path)
    print(f"Submission saved to: {submission_path}")

    # Verify Submission
    assert os.path.exists(submission_path), "Submission file was not created"
    sub_df = pd.read_csv(submission_path)
    print(f"Submission shape: {sub_df.shape}")
    print("First 3 predictions:")
    print(sub_df.head(3))

    expected_len = len(df_test_raw)
    assert (
        len(sub_df) == expected_len
    ), f"Submission length {len(sub_df)} mismatch with test set {expected_len}"
    assert (
        "id" in sub_df.columns and "PredictionString" in sub_df.columns
    ), "Invalid submission format"

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
