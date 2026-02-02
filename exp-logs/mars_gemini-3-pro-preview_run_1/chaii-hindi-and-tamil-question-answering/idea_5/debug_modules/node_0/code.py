import os
import sys
import pandas as pd
import torch
import shutil
import numpy as np

# Import provided library modules
from library.configuration import Config
from library.utilities import set_seed, jaccard, compute_average_jaccard
from library.data_processing import get_train_data, get_test_data
from library.modeling import MultiTaskQAModel
from library.training_inference import train_fn, inference_fn


def create_demo_data(config):
    """
    Creates a small subset of the data for demonstration purposes.
    """
    print(f"Creating demo data in {config.metadata_dir}...")

    # Ensure directories exist
    os.makedirs(config.metadata_dir, exist_ok=True)

    # Load original metadata
    # Note: In a real scenario, we read from ./metadata, but for this demo
    # we want to create specific small files referenced by our DemoConfig.
    # The provided metadata is in ./metadata.
    orig_train_path = "./metadata/train.csv"
    orig_val_path = "./metadata/val.csv"
    orig_test_path = "./metadata/test.csv"

    # Read and sample (take top 20 rows for speed)
    train_df = pd.read_csv(orig_train_path).head(20)
    val_df = pd.read_csv(orig_val_path).head(10)
    test_df = pd.read_csv(orig_test_path).head(10)

    # Save to the paths defined in DemoConfig
    train_df.to_csv(config.train_meta_path, index=False)
    val_df.to_csv(config.val_meta_path, index=False)
    test_df.to_csv(config.test_meta_path, index=False)

    print("Demo data created successfully.")


class DemoConfig(Config):
    """
    Configuration optimized for a quick demonstration run.
    """

    def __init__(self):
        super().__init__()

        # Use a separate working directory for the demo
        self.working_dir = "./working/demo_run"
        self.submission_dir = os.path.join(self.working_dir, "submission")

        # Override metadata paths to point to our temporary demo files
        # We will create these files in the working directory to avoid touching ./metadata
        self.demo_data_dir = os.path.join(self.working_dir, "demo_data")
        self.train_meta_path = os.path.join(self.demo_data_dir, "train.csv")
        self.val_meta_path = os.path.join(self.demo_data_dir, "val.csv")
        self.test_meta_path = os.path.join(self.demo_data_dir, "test.csv")

        # Override cache paths
        self.train_features_path = os.path.join(
            self.working_dir, "train_features.parquet"
        )
        self.test_features_path = os.path.join(
            self.working_dir, "test_features_inference.parquet"
        )

        # Model & Submission paths
        self.best_model_path = os.path.join(self.working_dir, "fold_0_best_model.pth")
        self.submission_path = os.path.join(self.submission_dir, "submission.csv")

        # Optimization for Speed
        # Use 'base' instead of 'large' to reduce download/load time and memory
        self.model_name = "xlm-roberta-base"
        self.epochs = 1
        self.batch_size = 2
        self.max_length = 128  # Shorter sequence length for speed
        self.doc_stride = 32
        self.n_best_size = 5

        # Re-create directories for this specific config
        self._create_directories()
        os.makedirs(self.demo_data_dir, exist_ok=True)


def verify_metrics():
    """
    Verifies the Jaccard metric implementation.
    """
    print("\n=== Verifying Metrics ===")
    s1 = "India is a country"
    s2 = "India country"
    # Intersection: {india, country} (len 2)
    # Union: {india, is, a, country} (len 4)
    # Jaccard: 0.5
    score = jaccard(s1, s2)
    print(f"Jaccard('{s1}', '{s2}') = {score}")

    assert abs(score - 0.5) < 1e-6, "Jaccard calculation is incorrect"

    avg_score = compute_average_jaccard([s1, s1], [s2, s1])
    # (0.5 + 1.0) / 2 = 0.75
    print(f"Average Jaccard = {avg_score}")
    assert abs(avg_score - 0.75) < 1e-6, "Average Jaccard calculation is incorrect"
    print("Metrics verified.")


def verify_data_processing(config):
    """
    Verifies dataset loading and feature processing.
    """
    print("\n=== Verifying Data Processing ===")
    # Load training data using the library function
    # This triggers tokenization, sliding window, and caching
    train_dataset = get_train_data(config, load_cached_data=False)

    print(f"Processed Train Dataset Size: {len(train_dataset)}")
    assert len(train_dataset) > 0, "Train dataset is empty"

    # Check a single item
    item = train_dataset[0]
    required_keys = [
        "input_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "relevance_labels",
    ]
    for key in required_keys:
        assert key in item, f"Missing key {key} in dataset item"

    # Check shapes
    input_ids = item["input_ids"]
    assert (
        input_ids.shape[0] == config.max_length
    ), f"Input IDs shape mismatch. Expected {config.max_length}, got {input_ids.shape[0]}"

    print("Data processing verified.")
    return train_dataset


def verify_model(config, dataset):
    """
    Verifies model initialization and forward pass.
    """
    print("\n=== Verifying Model ===")
    model = MultiTaskQAModel(config)
    model.to(config.device)
    model.eval()

    # Create a dummy batch
    batch_size = 2
    input_ids = torch.stack([dataset[i]["input_ids"] for i in range(batch_size)]).to(
        config.device
    )
    attention_mask = torch.stack(
        [dataset[i]["attention_mask"] for i in range(batch_size)]
    ).to(config.device)

    print(f"Running forward pass with batch size {batch_size}...")
    with torch.no_grad():
        start_logits, end_logits, relevance_logits = model(input_ids, attention_mask)

    # Check output shapes
    # Start/End logits: (batch_size, seq_len)
    assert start_logits.shape == (
        batch_size,
        config.max_length,
    ), f"Start logits shape mismatch: {start_logits.shape}"
    assert end_logits.shape == (
        batch_size,
        config.max_length,
    ), f"End logits shape mismatch: {end_logits.shape}"

    # Relevance logits: (batch_size,)
    assert relevance_logits.shape == (
        batch_size,
    ), f"Relevance logits shape mismatch: {relevance_logits.shape}"

    print("Model forward pass verified.")


def run_pipeline(config):
    """
    Runs the full training and inference pipeline.
    """
    print("\n=== Running Training Pipeline ===")
    # 1. Train
    # This will use the demo data created earlier
    train_fn(config)

    assert os.path.exists(
        config.best_model_path
    ), "Model file was not saved after training"
    print("Training complete. Model saved.")

    print("\n=== Running Inference Pipeline ===")
    # 2. Inference
    inference_fn(config)

    assert os.path.exists(config.submission_path), "Submission file was not generated"

    # Verify Submission Format
    sub_df = pd.read_csv(config.submission_path)
    print(f"Submission rows: {len(sub_df)}")
    print(sub_df.head())

    expected_cols = ["id", "PredictionString"]
    for col in expected_cols:
        assert col in sub_df.columns, f"Missing column {col} in submission file"

    # Verify that we have predictions for the test IDs
    test_df = pd.read_csv(config.test_meta_path)
    expected_ids = set(test_df["id"].unique())
    submitted_ids = set(sub_df["id"].unique())

    # Check if all test IDs are present in submission
    missing_ids = expected_ids - submitted_ids
    assert not missing_ids, f"Missing predictions for IDs: {missing_ids}"

    print("Pipeline execution verified.")


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)

    # Initialize Demo Configuration
    config = DemoConfig()

    # Prepare small dataset for the demo
    create_demo_data(config)

    # 1. Verify Metrics
    verify_metrics()

    # 2. Verify Data Processing
    dataset = verify_data_processing(config)

    # 3. Verify Model Logic
    verify_model(config, dataset)

    # 4. Run Full Pipeline (Train + Inference)
    run_pipeline(config)

    print("\nAll demonstrations completed successfully.")
