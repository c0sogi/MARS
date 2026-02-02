import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import provided library modules
from library.config import Config, setup_environment
from library.utils import set_seed, get_device, init_logger
from library.data_processing import get_dataloaders
from library.model_components import SiameseDebertaHierarchical
from library.engine import train_one_epoch, validate, inference_fn


class DemoConfig(Config):
    """
    Configuration optimized for a fast demonstration run.
    """

    # Enable debug mode to use a tiny subset of data
    DEBUG = True
    DEBUG_SAMPLE_SIZE = 64  # Small enough for quick execution

    # Reduce training duration
    EPOCHS = 1

    # Adjust batch sizes for the demo (A100 can handle more, but we keep it small for speed/safety)
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 8

    # Working directory for this specific demo
    WORKING_DIR = "./working/demo_execution"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "demo_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Reduce workers to minimize overhead for small data
    NUM_WORKERS = 2


def run_demo():
    # 1. Setup Environment
    print("\n[1] Setting up environment...")
    setup_environment(DemoConfig)
    device = get_device()
    logger = init_logger(os.path.join(DemoConfig.WORKING_DIR, "demo.log"))
    logger.info(f"Device: {device}")

    # 2. Initialize Tokenizer
    print("\n[2] Initializing Tokenizer...")
    # Using the model name from config
    tokenizer = AutoTokenizer.from_pretrained(DemoConfig.MODEL_NAME)

    # 3. Data Loading
    print("\n[3] Generating Dataloaders...")
    # This will trigger data processing, caching, and loading
    train_loader, val_loader, test_loader = get_dataloaders(
        DemoConfig, tokenizer, load_cached_data=False
    )

    # --- Verification: Data Shapes ---
    print("    Verifying data shapes...")
    try:
        batch = next(iter(train_loader))
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        scalar_features = batch["scalar_features"]
        labels = batch["labels"]

        # Expected shapes based on Config
        # Input IDs: (Batch, 2, Seq_Len)
        expected_input_shape = (DemoConfig.TRAIN_BATCH_SIZE, 2, DemoConfig.MAX_LENGTH)
        # Scalar Features: (Batch, 3) -> [log_len_p, log_len_a, log_len_b]
        expected_scalar_shape = (DemoConfig.TRAIN_BATCH_SIZE, 3)
        # Labels: (Batch, 3)
        expected_label_shape = (DemoConfig.TRAIN_BATCH_SIZE, 3)

        assert (
            input_ids.shape == expected_input_shape
        ), f"Input IDs shape mismatch. Got {input_ids.shape}, expected {expected_input_shape}"
        assert (
            scalar_features.shape == expected_scalar_shape
        ), f"Scalar features shape mismatch. Got {scalar_features.shape}, expected {expected_scalar_shape}"
        assert (
            labels.shape == expected_label_shape
        ), f"Labels shape mismatch. Got {labels.shape}, expected {expected_label_shape}"

        print("    -> Data shapes verified successfully.")
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # 4. Model Initialization
    print("\n[4] Initializing Model...")
    model = SiameseDebertaHierarchical(DemoConfig)
    model.to(device)

    # --- Verification: Forward Pass ---
    print("    Verifying forward pass...")
    model.eval()
    with torch.no_grad():
        # Move batch to device
        b_input_ids = input_ids.to(device)
        b_mask = attention_mask.to(device)
        b_scalars = scalar_features.to(device)

        # Run forward
        logits = model(b_input_ids, b_mask, scalar_features=b_scalars)

        # Check output
        assert logits.shape == (
            DemoConfig.TRAIN_BATCH_SIZE,
            DemoConfig.NUM_CLASSES,
        ), f"Logits shape mismatch. Got {logits.shape}, expected {(DemoConfig.TRAIN_BATCH_SIZE, DemoConfig.NUM_CLASSES)}"

        print("    -> Forward pass successful. Logits shape correct.")

    # 5. Training Loop Demonstration
    print("\n[5] Running Training Loop (1 Epoch)...")

    # Setup Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=DemoConfig.LEARNING_RATE,
        weight_decay=DemoConfig.WEIGHT_DECAY,
    )

    num_train_steps = len(train_loader) * DemoConfig.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * DemoConfig.WARMUP_RATIO),
        num_training_steps=num_train_steps,
    )

    # Train
    avg_train_loss = train_one_epoch(
        model, train_loader, optimizer, scheduler, device, 0, DemoConfig
    )

    assert isinstance(avg_train_loss, float), "Train loss should be a float"
    assert avg_train_loss > 0, "Train loss should be positive"
    print(f"    -> Training complete. Avg Loss: {avg_train_loss:.4f}")

    # 6. Validation Demonstration
    print("\n[6] Running Validation...")
    avg_val_loss = validate(model, val_loader, device, DemoConfig)

    assert isinstance(avg_val_loss, float), "Val loss should be a float"
    print(f"    -> Validation complete. Avg Loss: {avg_val_loss:.4f}")

    # 7. Inference Demonstration
    print("\n[7] Running Inference on Test Set...")
    # Note: inference_fn includes TTA (Test Time Augmentation) if configured
    predictions = inference_fn(model, test_loader, device, DemoConfig)

    # --- Verification: Predictions ---
    # Check shape: (Num_Test_Samples, 3)
    # Note: DEBUG_SAMPLE_SIZE applies to test set too in data_processing.py logic provided
    expected_rows = min(
        len(pd.read_csv(DemoConfig.TEST_PATH)), DemoConfig.DEBUG_SAMPLE_SIZE
    )

    assert predictions.shape == (
        expected_rows,
        3,
    ), f"Predictions shape mismatch. Got {predictions.shape}, expected {(expected_rows, 3)}"

    # Check probability properties (Sum to 1)
    row_sums = np.sum(predictions, axis=1)
    # Allow small floating point error
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1.0"

    print(
        "    -> Inference successful. Predictions shape and probability constraints verified."
    )

    # 8. Generate Submission
    print("\n[8] Generating Submission File...")
    test_df = pd.read_csv(DemoConfig.TEST_PATH)
    if DemoConfig.DEBUG:
        test_df = test_df.head(DemoConfig.DEBUG_SAMPLE_SIZE)

    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            "winner_model_a": predictions[:, 0],
            "winner_model_b": predictions[:, 1],
            "winner_tie": predictions[:, 2],
        }
    )

    submission.to_csv(DemoConfig.SUBMISSION_PATH, index=False)

    assert os.path.exists(
        DemoConfig.SUBMISSION_PATH
    ), "Submission file was not created."
    print(f"    -> Submission saved to {DemoConfig.SUBMISSION_PATH}")
    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    run_demo()
