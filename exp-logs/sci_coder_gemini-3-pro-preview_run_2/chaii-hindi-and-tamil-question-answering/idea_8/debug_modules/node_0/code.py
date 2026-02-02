import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, jaccard
from library.dataset import prepare_train_features, QADataset
from library.model import CustomXLMRoberta
from library.engine import get_optimizer_grouped_parameters, train_fn, eval_fn
from library.inference import inference_fn


def run_demo():
    # Suppress verbose logging for cleaner output
    transformers.logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("=== Starting Demonstration of QA Pipeline ===")

    # -------------------------------------------------------------------------
    # 1. Runtime Configuration Overrides
    # -------------------------------------------------------------------------
    # We modify the global Config class to optimize for a quick demonstration.
    print("[1] Configuring environment for fast execution...")

    # Enable debug mode to use a tiny subset of data
    Config.debug = True
    Config.debug_sample_size = 5  # Process only 5 documents

    # Use a smaller model for demonstration speed (Base instead of Large)
    Config.model_name = "xlm-roberta-base"

    # Training hyperparameters for speed
    Config.epochs = 1
    Config.train_batch_size = 2
    Config.valid_batch_size = 2
    Config.accumulate_grad_batches = 1  # No accumulation needed for small batch

    # Set a distinct working directory for the demo
    Config.working_dir = "./working/demo_run"
    Config.submission_path = os.path.join(Config.working_dir, "submission.csv")

    # Set seeds to a single value for the loop in inference
    Config.seeds = [42]

    # Ensure clean state
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)
    os.makedirs(Config.working_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("[2] Verifying Utility Functions...")
    seed_everything(42)

    # Test Jaccard Metric
    str1 = "The quick brown fox"
    str2 = "The quick brown"
    score = jaccard(str1, str2)
    # Intersection: {the, quick, brown} (3)
    # Union: {the, quick, brown, fox} (4)
    # Score: 3/4 = 0.75
    assert (
        abs(score - 0.75) < 1e-6
    ), f"Jaccard calculation failed. Expected 0.75, got {score}"
    print("    - Jaccard metric verified.")

    # -------------------------------------------------------------------------
    # 3. Data Processing & Dataset
    # -------------------------------------------------------------------------
    print("[3] Verifying Data Processing...")

    # Load training metadata
    if not os.path.exists(Config.train_path):
        raise FileNotFoundError(f"Metadata file not found: {Config.train_path}")

    df_train = pd.read_csv(Config.train_path)
    print(
        f"    - Loaded training metadata: {len(df_train)} rows (will use {Config.debug_sample_size})"
    )

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Generate Features (cached_train_features_debug.parquet will be created)
    # We force load_cached_data=False to demonstrate the processing logic
    train_features = prepare_train_features(
        df_train, tokenizer=tokenizer, load_cached_data=False
    )

    # Validation of feature structure
    assert isinstance(train_features, list), "Features should be a list"
    assert len(train_features) > 0, "No features generated"
    sample_feat = train_features[0]
    required_keys = [
        "input_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "answerable_label",
    ]
    for key in required_keys:
        assert key in sample_feat, f"Missing key in features: {key}"

    print(f"    - Generated {len(train_features)} sliding window features.")

    # Create PyTorch Dataset and Loader
    train_dataset = QADataset(train_features, mode="train")
    train_loader = DataLoader(
        train_dataset, batch_size=Config.train_batch_size, shuffle=True, drop_last=False
    )

    # Validate Batch
    batch = next(iter(train_loader))
    assert batch["input_ids"].shape[0] <= Config.train_batch_size
    assert batch["input_ids"].shape[1] == Config.max_len
    print("    - Dataset and DataLoader verified.")

    # -------------------------------------------------------------------------
    # 4. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("[4] Verifying Model Architecture...")
    device = Config.device
    model = CustomXLMRoberta()
    model.to(device)

    # Run a dummy forward pass
    with torch.no_grad():
        s_logits, e_logits, a_logits = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )

    # Check output shapes
    batch_current_size = batch["input_ids"].shape[0]
    assert s_logits.shape == (batch_current_size, Config.max_len)
    assert e_logits.shape == (batch_current_size, Config.max_len)
    assert a_logits.shape == (batch_current_size, 1)
    print("    - Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 5. Training Loop (Engine)
    # -------------------------------------------------------------------------
    print("[5] Verifying Training Loop...")

    # Setup Optimizer
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, Config)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=Config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * Config.epochs,
    )

    # Run Training (1 Epoch)
    train_loss = train_fn(train_loader, model, optimizer, device, scheduler, Config)
    assert not np.isnan(train_loss), "Training loss is NaN"
    print(f"    - Training complete. Loss: {train_loss:.4f}")

    # -------------------------------------------------------------------------
    # 6. Validation Loop
    # -------------------------------------------------------------------------
    print("[6] Verifying Validation Loop...")
    # Using train_loader as validation just for demonstration
    val_loss = eval_fn(train_loader, model, device, Config)
    assert not np.isnan(val_loss), "Validation loss is NaN"
    print(f"    - Validation complete. Loss: {val_loss:.4f}")

    # -------------------------------------------------------------------------
    # 7. Inference Pipeline
    # -------------------------------------------------------------------------
    print("[7] Verifying Inference Pipeline...")

    # Save the model state dict to the location expected by inference_fn
    # inference_fn looks for 'best_model_seed_{seed}.pth' or 'best_model.pth'
    save_path = os.path.join(
        Config.working_dir, f"best_model_seed_{Config.seeds[0]}.pth"
    )
    torch.save(model.state_dict(), save_path)
    print(f"    - Model saved to {save_path}")

    # Run Inference
    # This will:
    # 1. Load test data (Config.test_path)
    # 2. Tokenize (prepare_test_features)
    # 3. Load the model we just saved
    # 4. Predict and save submission.csv
    inference_fn()

    # Verify Submission
    if not os.path.exists(Config.submission_path):
        raise FileNotFoundError("Submission file was not created.")

    sub_df = pd.read_csv(Config.submission_path)
    print(f"    - Submission created with {len(sub_df)} rows.")

    # Check format
    assert "id" in sub_df.columns
    assert "PredictionString" in sub_df.columns
    # Check if we have predictions for the debug samples
    assert len(sub_df) > 0

    print("=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
