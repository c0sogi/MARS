import os
import shutil
import pandas as pd
import torch
import numpy as np
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_optimizer_grouped_parameters
from library.data import prepare_train_features, prepare_test_features
from library.model import CustomXLMR
from library.engine import train_one_epoch, predict_test


def main():
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    # Suppress verbose output from transformers
    transformers.logging.set_verbosity_error()

    # Set seeds for reproducibility
    seed_everything(42)

    print("Initializing Demonstration...")

    # Define a specific config for this demo run to ensure speed
    class DemoConfig(Config):
        # Use a smaller model for faster download and execution
        MODEL_NAME = "xlm-roberta-base"

        # Use temporary directories to avoid messing with real inputs/outputs
        METADATA_DIR = "./working/demo_metadata"
        WORKING_DIR = "./working/demo_run"

        # Reduce training parameters
        EPOCHS = 1
        BATCH_SIZE = 4
        MAX_LENGTH = 128  # Shorter sequence length for speed
        DOC_STRIDE = 64

        # Ensure we use the GPU if available
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    config = DemoConfig()

    # Clean up any previous demo run
    if os.path.exists(config.METADATA_DIR):
        shutil.rmtree(config.METADATA_DIR)
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)

    os.makedirs(config.METADATA_DIR, exist_ok=True)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # 2. Data Subsetting (Mocking Data)
    # -------------------------------------------------------------------------
    print("Creating data subsets for speed...")

    # Load original metadata
    orig_train_path = "./metadata/train.csv"
    orig_test_path = "./metadata/test.csv"
    orig_val_path = "./metadata/val.csv"

    # Create small subsets (top 20 rows)
    # We need to ensure columns match what library.data expects
    if os.path.exists(orig_train_path):
        df_train = pd.read_csv(orig_train_path).head(20)
        df_train.to_csv(os.path.join(config.METADATA_DIR, "train.csv"), index=False)

    if os.path.exists(orig_val_path):
        df_val = pd.read_csv(orig_val_path).head(5)
        df_val.to_csv(os.path.join(config.METADATA_DIR, "val.csv"), index=False)

    if os.path.exists(orig_test_path):
        df_test = pd.read_csv(orig_test_path).head(10)
        df_test.to_csv(os.path.join(config.METADATA_DIR, "test.csv"), index=False)

    # 3. Feature Engineering
    # -------------------------------------------------------------------------
    print("Preparing features...")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # Prepare Train Features
    # Note: load_cached_data=False forces re-computation using our new small dataset
    train_dataset = prepare_train_features(config, tokenizer, load_cached_data=False)

    # Validation: Ensure dataset is populated
    print(f"Train dataset size: {len(train_dataset)}")
    if len(train_dataset) == 0:
        raise AssertionError("Train dataset is empty! Check data preparation logic.")

    # Prepare Test Features
    test_dataset = prepare_test_features(config, tokenizer)
    print(f"Test dataset size: {len(test_dataset)}")
    if len(test_dataset) == 0:
        raise AssertionError("Test dataset is empty! Check data preparation logic.")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 0 workers for simple debugging/demo
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # 4. Model Initialization
    # -------------------------------------------------------------------------
    print("Initializing Model...")
    model = CustomXLMR.from_pretrained(config.MODEL_NAME)
    model.to(config.DEVICE)

    # Validation: Check model structure
    if not hasattr(model, "qa_outputs"):
        raise AssertionError("Model missing 'qa_outputs' head.")
    if not hasattr(model, "classifier"):
        raise AssertionError("Model missing 'classifier' (relevance) head.")

    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("Starting Training Demo...")

    # Setup Optimizer with Differential Learning Rates
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, config)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # Setup Scheduler
    num_training_steps = len(train_loader) * config.EPOCHS
    num_warmup_steps = int(num_training_steps * config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Run one epoch
    avg_loss = train_one_epoch(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=train_loader,
        device=config.DEVICE,
        epoch=0,
        config=config,
    )

    print(f"Training completed. Average Loss: {avg_loss:.4f}")

    # Validation: Loss should be a valid number
    if np.isnan(avg_loss) or np.isinf(avg_loss):
        raise AssertionError("Training loss resulted in NaN or Inf.")

    # 6. Inference Demonstration
    # -------------------------------------------------------------------------
    print("Starting Inference Demo...")

    predictions = predict_test(model, test_loader, config.DEVICE)

    # Unpack predictions
    start_logits = predictions["start_logits"]
    end_logits = predictions["end_logits"]
    relevance_logits = predictions["relevance_logits"]
    example_ids = predictions["example_ids"]
    offset_mappings = predictions["offset_mappings"]

    print("Inference completed.")
    print(f"Start Logits Shape: {start_logits.shape}")
    print(f"Example IDs Count: {len(example_ids)}")

    # Validation: Check shapes
    # start_logits shape: (num_features, max_seq_len)
    if start_logits.shape[1] != config.MAX_LENGTH:
        raise AssertionError(
            f"Logits sequence length ({start_logits.shape[1]}) does not match config ({config.MAX_LENGTH})."
        )

    if len(example_ids) != start_logits.shape[0]:
        raise AssertionError("Mismatch between number of example IDs and logits rows.")

    # 7. Simple Post-Processing Check (Optional but good for verification)
    # -------------------------------------------------------------------------
    # Just verify that we can map back to an example ID
    sample_idx = 0
    sample_id = example_ids[sample_idx]
    sample_offsets = offset_mappings[sample_idx]

    # Find the best start/end for this sample window
    start_pred = np.argmax(start_logits[sample_idx])
    end_pred = np.argmax(end_logits[sample_idx])

    print(
        f"Sample Prediction - ID: {sample_id}, Start Token: {start_pred}, End Token: {end_pred}"
    )

    # 8. Cleanup
    # -------------------------------------------------------------------------
    print("Cleaning up temporary files...")
    if os.path.exists(config.METADATA_DIR):
        shutil.rmtree(config.METADATA_DIR)
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)

    print("Demonstration finished successfully.")


if __name__ == "__main__":
    main()
