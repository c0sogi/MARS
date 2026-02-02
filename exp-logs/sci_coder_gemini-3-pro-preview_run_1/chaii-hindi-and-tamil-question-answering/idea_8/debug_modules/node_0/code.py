import os
import torch
import pandas as pd
import numpy as np
import warnings
import shutil
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, logging as hf_logging

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, jaccard
from library.modeling import CustomXLMRoberta
from library.dataset import get_train_dataset, get_test_dataset
from library.engine import get_optimizer, get_scheduler, train_fn
from library.predict import predict

# Suppress warnings and progress bars for cleaner output
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_demo():
    print("==== Starting QA Task Demo ====")

    # 1. Configuration Setup
    # We override defaults to ensure the demo runs quickly (Debug mode, 1 epoch, small batch)
    # We also set specific paths for this demo run to avoid overwriting production artifacts
    demo_working_dir = "./working/demo_run"

    # Clean up previous demo run if exists
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)

    config = Config(
        debug=True,  # Enable debug to use a small subset of data
        debug_sample_size=20,  # Only use 20 samples
        epochs=1,  # Train for only 1 epoch
        train_batch_size=2,  # Small batch size
        eval_batch_size=2,
        ensemble_seeds=[42],  # Train/Predict with a single seed
        working_dir=demo_working_dir,
        output_dir=os.path.join(demo_working_dir, "output"),
        cache_dir=os.path.join(demo_working_dir, "cache"),
        submission_path=os.path.join(demo_working_dir, "submission", "submission.csv"),
        # Using a smaller model for the demo if possible would be faster,
        # but the code relies on xlm-roberta-large specific hidden sizes implicitly via AutoConfig.
        # We stick to the config default but rely on debug=True to limit compute time.
    )

    # Ensure directories exist
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.submission_path), exist_ok=True)

    print(f"Configuration initialized. Working directory: {config.working_dir}")

    # Set seed for reproducibility
    seed_everything(config.seed)

    # 2. Metric Verification
    # Verify the Jaccard function logic
    s1 = "India is a country"
    s2 = "India country"
    # Intersection: {india, country} (2), Union: {india, is, a, country} (4) -> 0.5
    score = jaccard(s1, s2)
    print(f"Jaccard Test ('{s1}', '{s2}'): {score}")
    assert abs(score - 0.5) < 1e-6, "Jaccard calculation is incorrect"

    # 3. Dataset & Tokenizer
    print("\n==== Loading Tokenizer and Datasets ====")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Load Training Dataset (Force reload to skip cache for demo purposes)
    train_ds = get_train_dataset(config, tokenizer, load_cached_data=False)
    print(f"Training Dataset Size (Windows): {len(train_ds)}")

    # Verify Train Dataset Item Structure
    sample_item = train_ds[0]
    required_train_keys = [
        "input_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "relevance_labels",
    ]
    for key in required_train_keys:
        assert key in sample_item, f"Missing key {key} in training dataset item"
    assert (
        sample_item["input_ids"].shape[0] == config.max_length
    ), "Incorrect sequence length"

    # Load Test Dataset
    test_ds = get_test_dataset(config, tokenizer, load_cached_data=False)
    print(f"Test Dataset Size (Windows): {len(test_ds)}")

    # 4. Model Initialization
    print("\n==== Initializing Model ====")
    device = config.device
    model = CustomXLMRoberta(config)
    model.to(device)

    # Verify Model Output Shapes
    # Create a dummy batch
    dummy_input_ids = sample_item["input_ids"].unsqueeze(0).to(device)
    dummy_mask = sample_item["attention_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        start_logits, end_logits, rel_logits = model(dummy_input_ids, dummy_mask)

    # Expected shapes: Start/End -> (Batch, SeqLen), Relevance -> (Batch)
    assert start_logits.shape == (
        1,
        config.max_length,
    ), f"Start logits shape mismatch: {start_logits.shape}"
    assert end_logits.shape == (
        1,
        config.max_length,
    ), f"End logits shape mismatch: {end_logits.shape}"
    assert rel_logits.shape == (
        1,
    ), f"Relevance logits shape mismatch: {rel_logits.shape}"
    print("Model forward pass verification successful.")

    # 5. Training Loop Simulation
    print("\n==== Running Training Loop (1 Epoch) ====")
    train_loader = DataLoader(
        train_ds,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for demo
    )

    optimizer = get_optimizer(model, config)
    # Calculate total steps for scheduler
    num_train_steps = len(train_loader) * config.epochs
    scheduler = get_scheduler(optimizer, num_train_steps, config)

    # Train
    avg_loss = train_fn(train_loader, model, optimizer, device, scheduler, config)
    print(f"Epoch 1 Completed. Average Loss: {avg_loss:.4f}")

    # Save Model (Simulating the saving of the best model)
    # The predict function expects models named 'model_seed_{seed}.pth'
    model_save_path = os.path.join(config.output_dir, f"model_seed_{config.seed}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    assert os.path.exists(model_save_path), "Model file was not created"

    # 6. Inference Pipeline
    print("\n==== Running Inference Pipeline ====")
    # The predict function handles loading the test dataset, loading the model(s),
    # running inference, post-processing, and saving the submission.

    # Clear memory before inference
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    predict(config)

    # 7. Submission Validation
    print("\n==== Validating Submission ====")
    if not os.path.exists(config.submission_path):
        raise FileNotFoundError(
            f"Submission file not found at {config.submission_path}"
        )

    submission_df = pd.read_csv(config.submission_path)
    print(f"Submission loaded. Rows: {len(submission_df)}")
    print(submission_df.head())

    # Check columns
    assert "id" in submission_df.columns, "Submission missing 'id' column"
    assert (
        "PredictionString" in submission_df.columns
    ), "Submission missing 'PredictionString' column"

    # Check if we have predictions for the test IDs (based on the debug subset)
    # Note: In debug mode, get_test_dataset loads head(debug_sample_size).
    # The submission should contain rows corresponding to those IDs.
    test_metadata = pd.read_csv(config.test_path).head(config.debug_sample_size)
    expected_ids = set(test_metadata["id"].tolist())
    submitted_ids = set(submission_df["id"].tolist())

    # Check overlap (should be complete overlap if logic holds, but predict handles missing IDs gracefully)
    missing_ids = expected_ids - submitted_ids
    if missing_ids:
        print(f"Warning: {len(missing_ids)} IDs missing from submission.")
    else:
        print("All expected IDs are present in the submission.")

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
