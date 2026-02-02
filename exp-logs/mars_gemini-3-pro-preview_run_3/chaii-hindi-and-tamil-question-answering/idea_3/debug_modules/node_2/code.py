import os
import sys
import torch
import pandas as pd
import transformers
import warnings
import shutil

# Suppress warnings and verbose logs
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, jaccard, clean_text
from library.data_manager import (
    prepare_tapt_corpus,
    get_fold_dataloaders,
    get_test_dataloader,
    QADataset,
)
from library.model_factory import get_tokenizer, get_qa_model
from library.trainer import run_tapt, train_qa_fold
from library.inference import generate_submission


def main():
    print("Initializing Demonstration Script...")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed
    # -------------------------------------------------------------------------
    # We modify the Config class attributes directly to run a minimal version
    # of the pipeline suitable for a quick demonstration.
    print("Configuring hyperparameters for fast execution...")
    Config.SEED = 42
    Config.TAPT_EPOCHS = 1  # Reduce TAPT epochs
    Config.EPOCHS = 1  # Reduce QA Fine-tuning epochs
    Config.N_FOLDS = 1  # Run only 1 fold instead of 3
    Config.BATCH_SIZE = 8  # Small batch size
    Config.TAPT_BATCH_SIZE = 8
    Config.DEBUG = True  # Enable debug mode if applicable

    # Setup directories (creates working/idea_3/...)
    Config.setup()
    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Verify Utilities
    # -------------------------------------------------------------------------
    print("Verifying utility functions...")

    # Test Jaccard
    str1 = "The quick brown fox"
    str2 = "The quick brown"
    score = jaccard(str1, str2)
    # Intersection: {the, quick, brown} (3)
    # Union: {the, quick, brown, fox} (4)
    # Score: 3/4 = 0.75
    assert (
        abs(score - 0.75) < 1e-6
    ), f"Jaccard calculation failed. Expected 0.75, got {score}"

    # Test Clean Text
    raw_text = "  Hello   World  "
    cleaned = clean_text(raw_text)
    assert cleaned == "Hello   World", f"Text cleaning failed. Got '{cleaned}'"

    # -------------------------------------------------------------------------
    # 3. Task-Adaptive Pretraining (TAPT)
    # -------------------------------------------------------------------------
    print("\n--- Step 1: Task-Adaptive Pretraining (TAPT) ---")

    # The run_tapt function handles data prep and training
    # It saves the model to Config.TAPT_MODEL_PATH
    run_tapt()

    # Verify TAPT model was saved
    assert os.path.exists(
        Config.TAPT_MODEL_PATH
    ), "TAPT model directory not found after training."
    assert os.path.exists(
        os.path.join(Config.TAPT_MODEL_PATH, "config.json")
    ), "TAPT model config not found."
    print("TAPT completed and model saved successfully.")

    # -------------------------------------------------------------------------
    # 4. QA Model Training (Fold 0)
    # -------------------------------------------------------------------------
    print("\n--- Step 2: QA Model Training ---")

    tokenizer = get_tokenizer()

    # Initialize DataLoaders for Cross-Validation
    # We requested N_FOLDS=1, so the generator will yield once.
    fold_gen = get_fold_dataloaders(
        tokenizer, k_folds=Config.N_FOLDS, load_cached_data=False
    )

    try:
        train_loader, val_loader = next(fold_gen)
    except StopIteration:
        raise RuntimeError("DataLoader generator did not yield any folds.")

    # Verify DataLoaders
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    batch = next(iter(train_loader))
    assert "input_ids" in batch, "Batch missing input_ids"
    assert "labels" in batch, "Batch missing labels"
    assert batch["input_ids"].shape[0] <= Config.BATCH_SIZE, "Batch size mismatch"

    # Initialize Model
    # We load the weights from the TAPT step we just finished
    print(f"Loading model from TAPT checkpoint: {Config.TAPT_MODEL_PATH}")
    model = get_qa_model(model_path=Config.TAPT_MODEL_PATH)

    # Verify Model Architecture
    # XLM-RoBERTa base has 768 hidden size. Classifier should map to NUM_LABELS (3).
    assert (
        model.classifier.out_features == Config.NUM_LABELS
    ), "Model output dimension mismatch."

    # Train the fold
    # This saves the best model to Config.QA_MODEL_OUTPUT_DIR/model_fold_0.pt
    train_qa_fold(model, train_loader, val_loader, fold_idx=0)

    # Verify QA Model Checkpoint
    expected_model_path = os.path.join(Config.QA_MODEL_OUTPUT_DIR, "model_fold_0.pt")
    assert os.path.exists(
        expected_model_path
    ), f"QA model checkpoint not found at {expected_model_path}"
    print(f"QA training for Fold 0 completed. Model saved to {expected_model_path}")

    # -------------------------------------------------------------------------
    # 5. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n--- Step 3: Inference and Submission ---")

    # generate_submission handles loading the test set, predicting with all available folds,
    # ensemble voting, and saving the CSV.
    generate_submission()

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    # Check Columns
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert (
        "PredictionString" in df_sub.columns
    ), "Submission missing 'PredictionString' column"

    # Check Row Count (Test set has 112 rows)
    df_test_meta = pd.read_csv(Config.TEST_META_PATH)
    assert len(df_sub) == len(
        df_test_meta
    ), f"Submission row count mismatch. Expected {len(df_test_meta)}, got {len(df_sub)}"

    # Check content validity (PredictionString should be string)
    assert pd.api.types.is_string_dtype(
        df_sub["PredictionString"]
    ), "PredictionString column is not string type."

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
