import sys
import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

# Import provided library modules
from library.config import Config, set_seed
from library.utils import get_logger
from library.transformations import TransformationRegistry
from library.label_manager import LabelEngineer
from library.dataset import NormalizationDataset
from library.model import TransformerTokenClassifier
from library.trainer import run_training_pipeline, generate_submission


def main():
    # ==========================================
    # 1. Setup and Configuration
    # ==========================================
    # Set seed for reproducibility
    set_seed(42)

    # Initialize logger
    logger = get_logger("demo_execution")
    logger.info("Starting library demonstration script...")

    # Override Config for rapid demonstration
    logger.info("Overriding configuration for speed...")
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VAL_BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ==========================================
    # 2. Verify Transformation Logic
    # ==========================================
    logger.info("Step 1: Verifying TransformationRegistry...")
    registry = TransformationRegistry()

    # Test Case 1: Cardinal Number
    # Logic in transformations.py: 123 -> "one hundred twenty three" (space separated)
    raw_cardinal = "123"
    expected_cardinal = "one hundred twenty three"
    trans_cardinal = registry.apply("TRANS_CARDINAL", raw_cardinal)
    assert (
        trans_cardinal == expected_cardinal
    ), f"TRANS_CARDINAL failed. Expected '{expected_cardinal}', got '{trans_cardinal}'"

    # Test Case 2: Date Year
    # Logic: 2010 -> "twenty ten"
    raw_date = "2010"
    expected_date = "twenty ten"
    trans_date = registry.apply("TRANS_DATE_YEAR", raw_date)
    assert (
        trans_date == expected_date
    ), f"TRANS_DATE_YEAR failed. Expected '{expected_date}', got '{trans_date}'"

    # Test Case 3: Inverse Label Engineering
    # Should detect that "123" -> "one hundred twenty three" is a CARDINAL transformation
    detected_label = registry.find_best_transform(
        raw_cardinal, expected_cardinal, "CARDINAL"
    )
    assert (
        detected_label == "TRANS_CARDINAL"
    ), f"find_best_transform failed. Expected 'TRANS_CARDINAL', got '{detected_label}'"

    logger.info("Transformation logic verified successfully.")

    # ==========================================
    # 3. Verify Label Manager
    # ==========================================
    logger.info("Step 2: Verifying LabelEngineer...")
    label_engineer = LabelEngineer()

    # Process training data in debug mode (loads first 10k rows)
    # This generates 'label_id' and 'label_name' columns
    df_train_processed = label_engineer.process_dataset("train", debug=True)

    # Assertions
    assert (
        "label_id" in df_train_processed.columns
    ), "Processed dataframe missing 'label_id'"
    assert (
        "label_name" in df_train_processed.columns
    ), "Processed dataframe missing 'label_name'"
    assert not df_train_processed.empty, "Processed dataframe is empty"

    # Check if label encoder was created
    assert os.path.exists(Config.LABEL_ENCODER_PATH), "Label encoder file not found"

    logger.info(f"LabelEngineer processed {len(df_train_processed)} rows.")

    # ==========================================
    # 4. Verify Dataset and Tokenization
    # ==========================================
    logger.info("Step 3: Verifying NormalizationDataset...")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Initialize Datasets (Debug mode loads subset)
    train_dataset = NormalizationDataset("train", tokenizer, debug=True)
    val_dataset = NormalizationDataset("val", tokenizer, debug=True)

    # CRITICAL OPTIMIZATION: Manually slice dataset lists to minimal size
    # This ensures the training loop runs almost instantly
    subset_size = 10
    train_dataset.sentences = train_dataset.sentences[:subset_size]
    train_dataset.labels = train_dataset.labels[:subset_size]
    train_dataset.submission_ids = train_dataset.submission_ids[:subset_size]

    val_dataset.sentences = val_dataset.sentences[:subset_size]
    val_dataset.labels = val_dataset.labels[:subset_size]
    val_dataset.submission_ids = val_dataset.submission_ids[:subset_size]

    logger.info(f"Datasets sliced to {subset_size} samples for demo speed.")

    # Verify __getitem__ structure
    sample_item = train_dataset[0]
    required_keys = [
        "input_ids",
        "attention_mask",
        "labels",
        "word_ids",
        "raw_tokens",
        "submission_ids",
    ]
    for key in required_keys:
        assert key in sample_item, f"Dataset item missing key: {key}"

    # Verify tensor shapes
    assert sample_item["input_ids"].shape == (
        Config.MAX_LEN,
    ), f"Incorrect input_ids shape: {sample_item['input_ids'].shape}"

    logger.info("Dataset structure verified.")

    # ==========================================
    # 5. Verify Model Architecture
    # ==========================================
    logger.info("Step 4: Verifying TransformerTokenClassifier...")
    model = TransformerTokenClassifier(pretrained_model_name=Config.MODEL_NAME)

    # Move sample to device for forward pass check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create a dummy batch (unsqueeze to add batch dim)
    input_ids = sample_item["input_ids"].unsqueeze(0).to(device)
    attention_mask = sample_item["attention_mask"].unsqueeze(0).to(device)
    labels = sample_item["labels"].unsqueeze(0).to(device)

    # Forward pass
    output = model(input_ids, attention_mask, labels=labels)

    # Assertions
    assert output.loss is not None, "Model output missing loss"
    assert output.logits.shape == (
        1,
        Config.MAX_LEN,
        model.num_labels,
    ), f"Incorrect logits shape: {output.logits.shape}"

    logger.info("Model forward pass verified.")

    # ==========================================
    # 6. Verify Training Pipeline
    # ==========================================
    logger.info("Step 5: Running Training Pipeline (Demo)...")

    # Run training
    # This uses the sliced datasets, so it will finish very quickly
    trainer = run_training_pipeline(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=1,
        batch_size=Config.TRAIN_BATCH_SIZE,
        val_batch_size=Config.VAL_BATCH_SIZE,
    )

    # Verify checkpoint creation
    assert os.path.exists(
        Config.MODEL_CHECKPOINT_DIR
    ), "Checkpoint directory not created"
    # Note: HuggingFace save_pretrained saves config.json and model.safetensors/pytorch_model.bin
    has_model_file = any(
        f.endswith(".bin") or f.endswith(".safetensors")
        for f in os.listdir(Config.MODEL_CHECKPOINT_DIR)
    )
    assert has_model_file, "Model weights file not found in checkpoint directory"

    logger.info("Training pipeline completed successfully.")

    # ==========================================
    # 7. Verify Submission Generation
    # ==========================================
    logger.info("Step 6: Generating Submission...")

    # Initialize Test Dataset
    test_dataset = NormalizationDataset("test", tokenizer, debug=True)

    # Slice test dataset for speed
    test_dataset.sentences = test_dataset.sentences[:subset_size]
    test_dataset.submission_ids = test_dataset.submission_ids[:subset_size]

    # Define output path
    demo_submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Generate submission
    generate_submission(
        test_dataset, output_path=demo_submission_path, batch_size=Config.VAL_BATCH_SIZE
    )

    # Verify file content
    assert os.path.exists(demo_submission_path), "Submission file not created"
    df_sub = pd.read_csv(demo_submission_path)

    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert "after" in df_sub.columns, "Submission missing 'after' column"
    assert len(df_sub) > 0, "Submission file is empty"

    # Check that we have rows for the tokens in our subset
    # Note: The number of rows depends on the number of tokens in the first 10 sentences
    logger.info(f"Submission generated with {len(df_sub)} rows.")

    logger.info("All library components verified successfully.")


if __name__ == "__main__":
    main()
