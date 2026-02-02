import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, Timer
from library.data_processing import TextVectorizer, TagEncoder, prepare_loaders
from library.model import LinearTaggingModel, FocalLoss
from library.engine import train_model, validate
from library.inference import find_best_threshold, generate_submission


def configure_demo_settings():
    """
    Overrides Config parameters to ensure the demo runs quickly and efficiently.
    """
    print("Configuring settings for demonstration run...")

    # Enable Debug mode to use a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SIZE = 5000  # Process only 5000 samples

    # Reduce model complexity for speed
    Config.VOCAB_SIZE = 1000  # Smaller vocabulary
    Config.NUM_TAGS = 50  # Predict only top 50 tags
    Config.INPUT_DIM = Config.VOCAB_SIZE
    Config.OUTPUT_DIM = Config.NUM_TAGS

    # Training settings
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 128
    Config.NUM_WORKERS = 2

    # Set a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_run"
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update submission path to be inside the demo working dir or standard location
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")


def verify_components():
    """
    Verifies the logic of TextVectorizer and TagEncoder with synthetic data.
    """
    print("\n=== Verifying Components ===")

    # 1. Verify TextVectorizer
    print("Testing TextVectorizer...")
    texts = [
        "python is great for data science",
        "c++ is fast and powerful",
        "java is verbose but reliable",
    ]
    vectorizer = TextVectorizer()
    # Temporarily override config for this micro-test
    original_vocab = Config.VOCAB_SIZE
    Config.VOCAB_SIZE = 10

    try:
        vectorizer.fit(pd.Series(texts))
        features = vectorizer.transform(pd.Series(texts))

        assert features.shape == (
            3,
            10,
        ), f"Expected shape (3, 10), got {features.shape}"
        assert features.dtype == np.float32, "Feature dtype should be float32"
        print("TextVectorizer verification passed.")
    finally:
        Config.VOCAB_SIZE = original_vocab

    # 2. Verify TagEncoder
    print("Testing TagEncoder...")
    tags = pd.Series(["python java", "c++", "python c++ java"])
    encoder = TagEncoder()
    # Temporarily override num tags
    original_num_tags = Config.NUM_TAGS
    Config.NUM_TAGS = 3

    try:
        encoder.fit(tags)
        binary_matrix = encoder.transform(tags)

        # Check shape: 3 samples, 3 unique tags (python, java, c++)
        assert binary_matrix.shape == (
            3,
            3,
        ), f"Expected shape (3, 3), got {binary_matrix.shape}"

        # Check inverse transform
        reconstructed = encoder.inverse_transform(binary_matrix)
        assert (
            len(reconstructed) == 3
        ), "Inverse transform should return list of length 3"

        # Check correctness of first sample (python java)
        # The order in reconstructed tuple depends on internal sorting, but both should be present
        first_sample_tags = set(reconstructed[0])
        assert (
            "python" in first_sample_tags and "java" in first_sample_tags
        ), "Tag encoding/decoding failed"

        print("TagEncoder verification passed.")
    finally:
        Config.NUM_TAGS = original_num_tags


def run_pipeline_demo():
    """
    Runs the full pipeline: Data Loading -> Training -> Inference.
    """
    print("\n=== Running Full Pipeline Demo ===")

    # 1. Data Preparation
    # load_cached_data=False forces the script to process the raw CSVs (sliced by DEBUG_SIZE)
    print("Step 1: Preparing DataLoaders...")
    train_loader, val_loader, test_loader, encoder, test_ids = prepare_loaders(
        load_cached_data=False
    )

    # Verify DataLoaders
    x_batch, y_batch = next(iter(train_loader))
    print(f"Train Batch Shape - Inputs: {x_batch.shape}, Targets: {y_batch.shape}")

    assert (
        x_batch.shape[1] == Config.VOCAB_SIZE
    ), f"Input dim mismatch. Expected {Config.VOCAB_SIZE}, got {x_batch.shape[1]}"
    assert (
        y_batch.shape[1] == Config.NUM_TAGS
    ), f"Output dim mismatch. Expected {Config.NUM_TAGS}, got {y_batch.shape[1]}"

    # 2. Model Initialization
    print("\nStep 2: Initializing Model and Loss...")
    device = torch.device(Config.DEVICE)
    model = LinearTaggingModel(
        input_dim=Config.INPUT_DIM, output_dim=Config.OUTPUT_DIM
    ).to(device)
    criterion = FocalLoss()

    # Verify Forward Pass
    dummy_input = torch.randn(2, Config.INPUT_DIM).to(device)
    dummy_output = model(dummy_input)
    assert dummy_output.shape == (2, Config.OUTPUT_DIM), "Model output shape mismatch"
    print("Model initialized and forward pass verified.")

    # 3. Training
    print("\nStep 3: Training Model (1 Epoch)...")
    trained_model = train_model(model, train_loader, val_loader)

    # 4. Inference Preparation
    print("\nStep 4: Optimizing Threshold...")
    # Get validation probabilities
    val_loss, val_probs, val_targets = validate(
        trained_model, val_loader, criterion, device
    )

    # Find best threshold
    best_threshold = find_best_threshold(val_targets, val_probs)

    # 5. Submission Generation
    print("\nStep 5: Generating Submission...")
    submission_df = generate_submission(
        trained_model, test_loader, test_ids, encoder, best_threshold
    )

    # 6. Validate Submission File
    print("\nStep 6: Validating Submission File...")
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_check = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(df_check.columns) == ["Id", "Tags"], "Submission columns are incorrect."
    assert len(df_check) > 0, "Submission file is empty."

    # Check format of tags (should be string, space delimited or empty)
    if len(df_check) > 0:
        sample_tags = df_check.iloc[0]["Tags"]
        assert isinstance(sample_tags, str) or pd.isna(
            sample_tags
        ), "Tags column format incorrect."

    print(f"Submission generated successfully with {len(df_check)} rows.")
    print(f"File location: {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # Configure for speed
    configure_demo_settings()

    # Run verifications
    verify_components()

    # Run pipeline
    run_pipeline_demo()

    print("\nDemo completed successfully.")
