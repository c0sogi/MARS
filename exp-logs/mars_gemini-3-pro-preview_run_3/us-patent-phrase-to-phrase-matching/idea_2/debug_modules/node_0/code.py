import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
from transformers import AutoTokenizer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["transformers_verbosity"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import library components
from library.config import Config, seed_everything
from library.utils import compute_pearson_correlation
from library.dataset import load_and_process_data, PhraseDataset, CustomCollator
from library.model import CrossEncoderModel
from library.engine import train_fn


def run_demo():
    print("============================================================")
    print("      Phrase Matching Library Demonstration & Verification  ")
    print("============================================================")

    # ---------------------------------------------------------
    # 1. Setup & Configuration
    # ---------------------------------------------------------
    print("\n[1] Setting up Configuration...")
    seed_everything(42)

    # Override Config paths for this demo run to keep it isolated
    Config.working_dir = "./working/demo_run"
    Config.model_save_path = os.path.join(Config.working_dir, "model.pth")
    Config.submission_path = os.path.join(Config.working_dir, "submission.csv")

    # Update cache paths to be inside the demo directory
    Config.train_cache_path = os.path.join(Config.working_dir, "train.parquet")
    Config.val_cache_path = os.path.join(Config.working_dir, "val.parquet")
    Config.test_cache_path = os.path.join(Config.working_dir, "test.parquet")

    # Ensure directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    print(f"    Working Directory: {Config.working_dir}")
    print(f"    Model Save Path:   {Config.model_save_path}")

    # ---------------------------------------------------------
    # 2. Verify Utility Functions
    # ---------------------------------------------------------
    print("\n[2] Verifying Utilities...")

    # Test Pearson Correlation
    vec_a = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    vec_b = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Identical
    vec_c = np.array([0.5, 0.4, 0.3, 0.2, 0.1])  # Inverse

    corr_perfect = compute_pearson_correlation(vec_a, vec_b)
    corr_inverse = compute_pearson_correlation(vec_a, vec_c)

    print(f"    Pearson (Identical): {corr_perfect:.4f}")
    print(f"    Pearson (Inverse):   {corr_inverse:.4f}")

    assert np.isclose(
        corr_perfect, 1.0
    ), "Correlation of identical vectors should be 1.0"
    assert np.isclose(
        corr_inverse, -1.0
    ), "Correlation of inverse vectors should be -1.0"
    print("    -> Utility verification passed.")

    # ---------------------------------------------------------
    # 3. Verify Data Pipeline
    # ---------------------------------------------------------
    print("\n[3] Verifying Data Pipeline...")

    # Load Data
    df_train = load_and_process_data(Config.train_path, Config.train_cache_path)
    print(f"    Loaded Train Data: {df_train.shape}")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Create a small dataset subset
    subset_df = df_train.head(8).reset_index(drop=True)
    dataset = PhraseDataset(subset_df, tokenizer, max_length=64)

    # Check single item
    item = dataset[0]
    assert "input_ids" in item, "Dataset item missing input_ids"
    assert "attention_mask" in item, "Dataset item missing attention_mask"
    assert "labels" in item, "Dataset item missing labels"

    # Check Collator
    collator = CustomCollator(tokenizer)
    batch_list = [dataset[i] for i in range(len(dataset))]
    batch = collator(batch_list)

    print(f"    Batch Input Shape: {batch['input_ids'].shape}")
    print(f"    Batch Labels Shape: {batch['labels'].shape}")

    assert batch["input_ids"].shape[0] == 8, "Batch size mismatch"
    assert batch["labels"].shape[0] == 8, "Labels batch size mismatch"
    print("    -> Data pipeline verification passed.")

    # ---------------------------------------------------------
    # 4. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    model = CrossEncoderModel(model_name=Config.model_name, num_labels=1)
    model.to(Config.device)
    model.eval()

    # Move batch to device
    input_ids = batch["input_ids"].to(Config.device)
    attention_mask = batch["attention_mask"].to(Config.device)
    labels = batch["labels"].to(Config.device)

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    loss = outputs["loss"]
    logits = outputs["logits"]

    print(f"    Forward Pass Loss: {loss.item():.4f}")
    print(f"    Logits Shape: {logits.shape}")

    assert loss is not None, "Model did not return loss"
    assert logits.shape == (
        8,
        1,
    ), f"Logits shape mismatch. Expected (8, 1), got {logits.shape}"
    print("    -> Model verification passed.")

    # ---------------------------------------------------------
    # 5. End-to-End Engine Execution
    # ---------------------------------------------------------
    print("\n[5] Running End-to-End Engine (Debug Mode)...")

    # Run training function with debug=True to use small subsets
    # We use a very small batch size and 1 epoch for speed
    trained_model = train_fn(
        debug=True,
        epochs=1,
        batch_size=4,
        learning_rate=1e-5,
        patience=1,
        save_path=Config.model_save_path,
        submission_path=Config.submission_path,
    )

    print("    -> Engine execution completed.")

    # ---------------------------------------------------------
    # 6. Verify Artifacts
    # ---------------------------------------------------------
    print("\n[6] Verifying Output Artifacts...")

    # Check Model File
    if os.path.exists(Config.model_save_path):
        print(f"    [OK] Model file found: {Config.model_save_path}")
        file_size = os.path.getsize(Config.model_save_path) / (1024 * 1024)
        print(f"         Size: {file_size:.2f} MB")
    else:
        raise FileNotFoundError(f"Model file not found at {Config.model_save_path}")

    # Check Submission File
    if os.path.exists(Config.submission_path):
        print(f"    [OK] Submission file found: {Config.submission_path}")
        df_sub = pd.read_csv(Config.submission_path)
        print(f"         Shape: {df_sub.shape}")
        print(f"         Columns: {list(df_sub.columns)}")

        # Verify content
        assert "id" in df_sub.columns, "Submission missing 'id' column"
        assert "score" in df_sub.columns, "Submission missing 'score' column"
        assert (
            len(df_sub) == 50
        ), f"Expected 50 predictions (debug mode), got {len(df_sub)}"

        # Verify score range
        min_score = df_sub["score"].min()
        max_score = df_sub["score"].max()
        print(f"         Score Range: [{min_score:.4f}, {max_score:.4f}]")
        assert 0.0 <= min_score and max_score <= 1.0, "Scores out of range [0, 1]"
    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.submission_path}"
        )

    print("\n============================================================")
    print("      SUCCESS: All checks passed!                           ")
    print("============================================================")


if __name__ == "__main__":
    run_demo()
