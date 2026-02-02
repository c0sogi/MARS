import os
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.configuration import Config, seed_everything
from library.feature_engineering import FeatureEngineer
from library.dataset import get_tokenizer, load_supervised_data, Collate
from library.modeling import EssayModel
from library.pretraining import run_pretraining
from library.training import run_training
from library.meta_modeling import run_stacking


def run_demo():
    print("=== Starting End-to-End Pipeline Demonstration ===")

    # 1. Configure for Fast Execution (Monkey-patching Config)
    print("\n[1] Configuring environment for fast demonstration...")

    # Use a tiny model to ensure execution finishes in seconds/minutes
    Config.MODEL_BACKBONE = "prajjwal1/bert-tiny"
    Config.EXP_NAME = "demo_execution"
    Config.WORKING_DIR = f"./working/{Config.EXP_NAME}"

    # Update paths based on new working dir
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.MLM_CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "mlm_checkpoints")
    Config.SUBMISSION_DIR = f"./working/{Config.EXP_NAME}/submission"
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce hyperparameters for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Use only 20 samples
    Config.MAX_LENGTH = 64  # Short context
    Config.EPOCHS = 1  # 1 Epoch
    Config.MLM_EPOCHS = 1  # 1 Epoch MLM
    Config.NUM_FOLDS = 2  # 2 Folds (min for CV)
    Config.TRAIN_BATCH_SIZE = 4
    Config.EVAL_BATCH_SIZE = 4
    Config.MLM_BATCH_SIZE = 4
    Config.GRAD_ACCUM_STEPS = 1

    # Clean previous run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Setup directories
    Config.setup_directories()
    seed_everything(Config.SEED)

    print(f"    Backbone: {Config.MODEL_BACKBONE}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # 2. Feature Engineering Verification
    print("\n[2] Verifying Feature Engineering...")
    fe = FeatureEngineer()

    # Create dummy data
    dummy_df = pd.DataFrame(
        {
            "essay_id": ["001", "002"],
            "full_text": [
                "This is a test essay.",
                "Another test essay with more words in it.",
            ],
            "score": [3, 4],
        }
    )

    features_df = fe.extract_features(dummy_df)

    # Assertions
    expected_cols = [
        "char_count",
        "word_count",
        "sentence_count",
        "unique_word_count",
        "avg_word_len",
    ]
    for col in expected_cols:
        assert col in features_df.columns, f"Missing feature column: {col}"

    assert features_df.iloc[0]["word_count"] == 5, "Incorrect word count calculation"
    print("    Feature Engineering logic verified.")

    # 3. Dataset & Tokenizer Verification
    print("\n[3] Verifying Dataset and Tokenizer...")
    tokenizer = get_tokenizer()

    # Load supervised data (this uses the metadata files provided in the environment)
    ds = load_supervised_data("train", tokenizer, load_cached_data=False, debug=True)

    assert (
        len(ds) == Config.DEBUG_SAMPLE_SIZE
    ), f"Dataset size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(ds)}"

    sample_item = ds[0]
    assert "input_ids" in sample_item, "Missing input_ids in dataset item"
    assert "attention_mask" in sample_item, "Missing attention_mask in dataset item"
    assert "labels" in sample_item, "Missing labels in dataset item"

    # Test Collate function
    collate_fn = Collate(tokenizer)
    batch = [ds[0], ds[1]]
    collated_batch = collate_fn(batch)

    assert collated_batch["input_ids"].shape[0] == 2, "Batch size mismatch in collation"
    assert (
        collated_batch["input_ids"].shape[1] <= Config.MAX_LENGTH
    ), "Sequence length exceeds max length"
    print("    Dataset and Collate logic verified.")

    # 4. Model Architecture Verification
    print("\n[4] Verifying Model Architecture...")
    # Initialize model (random weights)
    model = EssayModel(pretrained=False)
    model.to(Config.DEVICE)
    model.eval()

    # Move batch to device
    inputs = collated_batch["input_ids"].to(Config.DEVICE)
    masks = collated_batch["attention_mask"].to(Config.DEVICE)

    with torch.no_grad():
        outputs = model(inputs, masks)

    assert outputs.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {outputs.shape}"
    print("    Model forward pass verified.")

    # Clean up model to free memory
    del model, inputs, masks, outputs
    torch.cuda.empty_cache()

    # 5. Pipeline Stage 1: Pretraining (MLM)
    print("\n[5] Running Stage 1: MLM Pretraining...")
    run_pretraining(debug=True, load_cached_data=False)

    # Verify MLM Checkpoints
    mlm_config_path = os.path.join(Config.MLM_CHECKPOINT_DIR, "config.json")
    assert os.path.exists(mlm_config_path), "MLM config not saved"

    # Check for model weights (safetensors or bin)
    has_safetensors = os.path.exists(
        os.path.join(Config.MLM_CHECKPOINT_DIR, "model.safetensors")
    )
    has_bin = os.path.exists(
        os.path.join(Config.MLM_CHECKPOINT_DIR, "pytorch_model.bin")
    )
    assert has_safetensors or has_bin, "MLM model weights not saved"
    print("    MLM Pretraining completed and verified.")

    # 6. Pipeline Stage 2: Supervised Training
    print("\n[6] Running Stage 2: Supervised Training (5-Fold CV - Debug Mode)...")
    oof_df = run_training(debug=True, load_cached_data=False)

    # Verify OOF Predictions
    assert not oof_df.empty, "OOF DataFrame is empty"
    assert "pred_score" in oof_df.columns, "OOF DataFrame missing 'pred_score'"

    # Verify Checkpoints for the 2 folds
    for fold in range(Config.NUM_FOLDS):
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"model_fold_{fold}.pth")
        assert os.path.exists(ckpt_path), f"Checkpoint for fold {fold} missing"

    print("    Supervised Training completed and verified.")

    # 7. Pipeline Stage 3: Meta-Modeling (Stacking)
    print("\n[7] Running Stage 3: Meta-Modeling (Stacking)...")
    submission_df = run_stacking(debug=True, load_cached_data=False)

    # Verify Submission
    assert not submission_df.empty, "Submission DataFrame is empty"
    assert "essay_id" in submission_df.columns, "Submission missing 'essay_id'"
    assert "score" in submission_df.columns, "Submission missing 'score'"

    # Verify Score Range (1-6) and Type (int)
    scores = submission_df["score"]
    assert pd.api.types.is_integer_dtype(scores), "Scores are not integers"
    assert scores.min() >= 1, "Scores contain values < 1"
    assert scores.max() <= 6, "Scores contain values > 6"

    print("    Meta-Modeling completed and verified.")

    print("\n=== Demonstration Completed Successfully ===")
    print(f"Final Submission File: {Config.SUBMISSION_FILE}")
    print(submission_df.head())


if __name__ == "__main__":
    run_demo()
