import os
import shutil
import pandas as pd
import torch
import numpy as np
import logging
from transformers import logging as hf_logging
from transformers import AutoTokenizer

# Import provided library components
from library.config import Config
from library.utils import seed_everything
from library.data import EssayDataset, Collate, get_mlm_data, get_test_data
from library.modeling import EssayScorer
from library.pretraining import run_mlm
from library.training import run_cross_validation
from library.inference import generate_predictions


def main():
    # 1. Setup Environment
    # ---------------------------------------------------------
    print("=== Setting up Demo Environment ===")
    seed_everything(42)

    # Suppress verbose logging for cleaner output
    hf_logging.set_verbosity_error()
    logging.basicConfig(level=logging.ERROR)

    # Define a working directory for this demo
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # 2. Data Preparation (Create Subsets)
    # ---------------------------------------------------------
    print("=== Preparing Subset Data ===")
    # Load a small slice of the provided metadata to ensure speed
    subset_size = 20

    # Read original metadata
    df_train_orig = pd.read_csv("./metadata/train.csv")
    df_val_orig = pd.read_csv("./metadata/val.csv")
    df_test_orig = pd.read_csv("./metadata/test.csv")

    # Create subsets
    df_train_sub = df_train_orig.head(subset_size).copy()
    df_val_sub = df_val_orig.head(subset_size).copy()
    df_test_sub = df_test_orig.head(subset_size).copy()

    # Save to demo directory
    train_sub_path = os.path.join(demo_dir, "train.csv")
    val_sub_path = os.path.join(demo_dir, "val.csv")
    test_sub_path = os.path.join(demo_dir, "test.csv")

    df_train_sub.to_csv(train_sub_path, index=False)
    df_val_sub.to_csv(val_sub_path, index=False)
    df_test_sub.to_csv(test_sub_path, index=False)

    print(f"Created subset data with {subset_size} samples each.")

    # 3. Configure Overrides
    # ---------------------------------------------------------
    print("=== Configuring Hyperparameters for Speed ===")
    # Point Config to the new subset files
    Config.train_path = train_sub_path
    Config.val_path = val_sub_path
    Config.test_path = test_sub_path
    Config.output_dir = demo_dir

    # Reduce computational load
    Config.max_length = 64  # Short sequences
    Config.train_batch_size = 4
    Config.valid_batch_size = 4
    Config.mlm_batch_size = 4
    Config.epochs = 1  # Single epoch
    Config.mlm_epochs = 1  # Single epoch for MLM
    Config.num_folds = 2  # Only 2 folds
    Config.gradient_accumulation_steps = 1
    Config.debug = True  # Enable debug mode (though we manually subsetted)

    print("Configuration updated.")

    # 4. Verify Data Loading
    # ---------------------------------------------------------
    print("\n=== Verifying Data Loading Components ===")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Test Dataset Class
    ds = EssayDataset(
        df_train_sub, tokenizer, max_length=Config.max_length, include_labels=True
    )
    sample = ds[0]

    # Assertions
    assert "input_ids" in sample, "Dataset item missing input_ids"
    assert "attention_mask" in sample, "Dataset item missing attention_mask"
    assert "labels" in sample, "Dataset item missing labels"
    assert isinstance(sample["labels"], torch.Tensor), "Labels should be a tensor"

    # Test Collate Function
    collate_fn = Collate(tokenizer)
    batch_size = 4
    batch_raw = [ds[i] for i in range(batch_size)]
    batch = collate_fn(batch_raw)

    # Assertions
    assert batch["input_ids"].shape[0] == batch_size, "Batch size mismatch"
    assert (
        batch["input_ids"].shape[1] <= Config.max_length
    ), "Sequence length exceeds max_length"
    assert "labels" in batch, "Batch missing labels"
    print("Data loading logic verified successfully.")

    # 5. Verify Modeling
    # ---------------------------------------------------------
    print("\n=== Verifying Model Architecture ===")
    model = EssayScorer(pretrained=True)
    model.to(Config.device)
    model.eval()

    # Prepare inputs
    input_ids = batch["input_ids"].to(Config.device)
    mask = batch["attention_mask"].to(Config.device)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, mask)

    # Assertions
    assert outputs.shape == (
        batch_size,
    ), f"Expected output shape ({batch_size},), got {outputs.shape}"
    print("Model forward pass verified successfully.")

    # Cleanup to save memory
    del model, input_ids, mask, outputs
    torch.cuda.empty_cache()

    # 6. Verify Pre-training (MLM)
    # ---------------------------------------------------------
    print("\n=== Running Domain Adaptive Pre-training (MLM) ===")
    # Run MLM on the subset data
    # load_cached_data=False ensures we process the new subset files
    mlm_model_path = run_mlm(load_cached_data=False)

    # Assertions
    assert os.path.isdir(mlm_model_path), "MLM output directory not found"
    assert os.path.exists(
        os.path.join(mlm_model_path, "pytorch_model.bin")
    ) or os.path.exists(
        os.path.join(mlm_model_path, "model.safetensors")
    ), "MLM model weights not saved"
    print(f"MLM training completed. Model saved to: {mlm_model_path}")

    # 7. Verify Supervised Training (Cross-Validation)
    # ---------------------------------------------------------
    print("\n=== Running Supervised Training (2 Folds) ===")
    # Run cross-validation using the MLM-adapted model
    model_paths = run_cross_validation(mlm_model_path=mlm_model_path)

    # Assertions
    assert (
        len(model_paths) == Config.num_folds
    ), f"Expected {Config.num_folds} model paths, got {len(model_paths)}"
    for path in model_paths:
        assert os.path.exists(path), f"Model checkpoint missing: {path}"
    print("Cross-validation training completed successfully.")

    # 8. Verify Inference
    # ---------------------------------------------------------
    print("\n=== Running Inference ===")
    # Generate predictions on the subset test data
    submission_path = generate_predictions(
        model_paths=model_paths, load_cached_data=False
    )

    # Assertions
    assert os.path.exists(submission_path), "Submission file not found"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    assert "essay_id" in df_sub.columns, "Submission missing essay_id column"
    assert "score" in df_sub.columns, "Submission missing score column"
    assert len(df_sub) == len(
        df_test_sub
    ), f"Submission row count mismatch. Expected {len(df_test_sub)}, got {len(df_sub)}"

    # Check score validity (1-6)
    valid_scores = df_sub["score"].between(1, 6).all()
    assert valid_scores, "Predictions contain scores outside the 1-6 range"

    print(f"Inference verified. Submission saved to: {submission_path}")

    print("\n=== All Demonstrations Passed Successfully ===")


if __name__ == "__main__":
    main()
