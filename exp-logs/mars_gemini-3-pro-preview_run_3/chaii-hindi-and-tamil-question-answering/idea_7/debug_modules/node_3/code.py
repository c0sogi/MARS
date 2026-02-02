import os
import shutil
import pandas as pd
import torch
import warnings
import logging

# Suppress warnings and logs for clean output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import Library modules
from library.config import Config
from library.utils import set_seed, jaccard
from library.data import QADataset, qa_collate_fn, TAPTDataset
from library.model import WeightedTokenClassifier, get_class_weights
from library.tapt_engine import run_tapt
from library.qa_engine import run_training
from library.inference import run_inference


def main():
    print("=== Starting Hindi/Tamil QA Task Demonstration ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration & Environment Setup
    # -------------------------------------------------------------------------
    print("[1/7] Setting up demonstration environment...")

    # Define a specific directory for this demo run to avoid polluting main working dir
    DEMO_DIR = "./working/demo_run"
    DEMO_METADATA = os.path.join(DEMO_DIR, "metadata")

    # Clean up any previous demo runs
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_METADATA, exist_ok=True)

    # Create small data subsets to ensure the script runs quickly (within minutes)
    # We read the original metadata and take the top N rows.
    print("      Creating data subsets (Train: 20, Val: 10, Test: 10)...")
    try:
        train_full = pd.read_csv("./metadata/train.csv")
        val_full = pd.read_csv("./metadata/val.csv")
        test_full = pd.read_csv("./metadata/test.csv")

        train_subset = train_full.head(20)
        val_subset = val_full.head(10)
        test_subset = test_full.head(10)

        train_subset_path = os.path.join(DEMO_METADATA, "train.csv")
        val_subset_path = os.path.join(DEMO_METADATA, "val.csv")
        test_subset_path = os.path.join(DEMO_METADATA, "test.csv")

        train_subset.to_csv(train_subset_path, index=False)
        val_subset.to_csv(val_subset_path, index=False)
        test_subset.to_csv(test_subset_path, index=False)
    except FileNotFoundError as e:
        print(f"Error: Metadata files not found. Ensure ./metadata exists. {e}")
        return

    # Monkey-patch the Config class to use our demo settings
    # This overrides the default settings in library/config.py for this execution only.
    print("      Patching Config with demo parameters...")
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "qa_cache")
    Config.MODEL_OUTPUT_DIR = os.path.join(DEMO_DIR, "qa_models")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.TAPT_CACHE_DIR = os.path.join(DEMO_DIR, "tapt_cache")
    Config.TAPT_OUTPUT_DIR = os.path.join(DEMO_DIR, "tapt_model_finetuned")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Point Config to our subsets
    Config.TRAIN_CSV = train_subset_path
    Config.VAL_CSV = val_subset_path
    Config.TEST_CSV = test_subset_path

    # Reduce hyperparameters for speed
    Config.EPOCHS = 1
    Config.TAPT_EPOCHS = 1
    Config.SEEDS = [42]  # Run only one seed
    Config.BATCH_SIZE = 4
    Config.TAPT_BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Disable multiprocessing for small data to avoid overhead
    Config.LOAD_CACHED_DATA = False  # Force reprocessing of our new subsets

    # Create necessary directories based on new Config paths
    for d in [
        Config.WORKING_DIR,
        Config.CACHE_DIR,
        Config.MODEL_OUTPUT_DIR,
        Config.SUBMISSION_DIR,
        Config.TAPT_CACHE_DIR,
        Config.TAPT_OUTPUT_DIR,
    ]:
        os.makedirs(d, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Verify Utilities
    # -------------------------------------------------------------------------
    print("\n[2/7] Verifying Utilities...")
    set_seed(42)

    # Test Jaccard Score
    s1 = "This is a test answer"
    s2 = "This is a test"
    score = jaccard(s1, s2)
    print(f"      Jaccard('{s1}', '{s2}') = {score:.4f}")

    # Assertions
    assert 0.0 < score < 1.0, "Jaccard score calculation seems incorrect."
    assert jaccard("same", "same") == 1.0
    assert jaccard("a", "b") == 0.0
    print("      Utils verification passed.")

    # -------------------------------------------------------------------------
    # 3. Verify Data Loading
    # -------------------------------------------------------------------------
    print("\n[3/7] Verifying Data Loading...")

    # Instantiate Training Dataset
    # This triggers tokenization and sliding window creation
    train_ds = QADataset(mode="train", load_cached_data=False)
    print(f"      Train Dataset size (windows): {len(train_ds)}")
    assert len(train_ds) > 0, "Training dataset should not be empty."

    # Inspect a single item
    item = train_ds[0]
    required_keys = ["input_ids", "attention_mask", "labels", "example_id", "context"]
    for k in required_keys:
        assert k in item, f"Dataset item missing key: {k}"
    assert isinstance(item["input_ids"], torch.Tensor)

    # Verify Collate Function
    batch_list = [train_ds[0], train_ds[1]]
    batch = qa_collate_fn(batch_list)
    assert "input_ids" in batch
    assert batch["input_ids"].shape[0] == 2
    assert "metadata" in batch
    print("      Data loading and collation verification passed.")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4/7] Verifying Model Architecture...")

    # Calculate Class Weights
    weights = get_class_weights(train_ds)
    print(f"      Computed Class Weights: {weights.tolist()}")
    assert weights.shape[0] == Config.NUM_LABELS

    # Initialize Model
    model = WeightedTokenClassifier(class_weights=weights)
    model.to(Config.DEVICE)
    model.eval()

    # Run Forward Pass (Simulation)
    with torch.no_grad():
        inputs = batch["input_ids"].to(Config.DEVICE)
        mask = batch["attention_mask"].to(Config.DEVICE)
        labels = batch["labels"].to(Config.DEVICE)

        # Pass 1: With labels (Training mode -> returns loss)
        loss, logits = model(inputs, attention_mask=mask, labels=labels)
        assert loss is not None
        assert not torch.isnan(loss)
        print(f"      Forward pass (Training) Loss: {loss.item():.4f}")

        # Pass 2: Without labels (Inference mode -> returns logits)
        logits_only = model(inputs, attention_mask=mask)
        assert logits_only.shape == (2, Config.MAX_LENGTH, Config.NUM_LABELS)
        print("      Forward pass (Inference) shape verified.")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 5. Verify TAPT (Task-Adaptive Pretraining)
    # -------------------------------------------------------------------------
    print("\n[5/7] Running TAPT (Pretraining)...")

    # run_tapt() uses the TAPTDataset which reads from Config paths.
    # Since we patched Config, it uses our subsets.
    tapt_model_path = run_tapt()

    assert os.path.exists(tapt_model_path), "TAPT output directory missing."
    assert os.path.exists(
        os.path.join(tapt_model_path, "config.json")
    ), "TAPT model config missing."
    print(f"      TAPT completed. Model saved to: {tapt_model_path}")

    # -------------------------------------------------------------------------
    # 6. Verify QA Training Pipeline
    # -------------------------------------------------------------------------
    print("\n[6/7] Running QA Training Pipeline...")

    # run_training() iterates through Config.SEEDS (only [42] now).
    # It trains the model, saves it, and generates a submission.
    run_training()

    # Verify Model Artifacts
    expected_model_path = os.path.join(Config.MODEL_OUTPUT_DIR, "model_seed_42.pt")
    assert os.path.exists(
        expected_model_path
    ), f"Trained model not found at {expected_model_path}"
    print(f"      Model checkpoint verified at: {expected_model_path}")

    # -------------------------------------------------------------------------
    # 7. Verify Inference Pipeline
    # -------------------------------------------------------------------------
    print("\n[7/7] Verifying Inference Pipeline...")

    # Define a separate output path for this specific inference test
    inference_sub_path = os.path.join(
        Config.SUBMISSION_DIR, "submission_inference_test.csv"
    )

    # Run explicit inference
    run_inference(
        test_csv_path=Config.TEST_CSV,
        submission_path=inference_sub_path,
        model_dir=Config.MODEL_OUTPUT_DIR,
        seeds=Config.SEEDS,
        batch_size=Config.BATCH_SIZE,
        device=Config.DEVICE,
        num_workers=Config.NUM_WORKERS,
    )

    # Validate Submission File
    assert os.path.exists(
        inference_sub_path
    ), "Inference submission file not generated."

    # Use keep_default_na=False to ensure empty strings are read as strings, not NaN
    df_sub = pd.read_csv(inference_sub_path, keep_default_na=False)
    print(f"      Submission Shape: {df_sub.shape}")

    # Check columns
    assert "id" in df_sub.columns
    assert "PredictionString" in df_sub.columns

    # Check row count matches test subset
    assert len(df_sub) == len(
        test_subset
    ), f"Expected {len(test_subset)} predictions, got {len(df_sub)}"

    # Check content (PredictionString should be string)
    assert df_sub["PredictionString"].dtype == object
    print("      Inference output format verified.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
