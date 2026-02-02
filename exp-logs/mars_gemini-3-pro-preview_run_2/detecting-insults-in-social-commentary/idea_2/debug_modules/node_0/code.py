import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import transformers
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

# Import from the provided library files
from library.config import Config
from library.data_utils import get_dataloaders
from library.model import TransformerClassifier
from library.train_utils import train_one_epoch, evaluate, predict


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Starting demonstration script...")

    # 1. Configuration Setup
    # Override Config parameters for a fast demonstration
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 16
    Config.VALID_BATCH_SIZE = 32

    # Ensure reproducibility
    set_seed(Config.SEED)
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {Config.MODEL_NAME}")

    # 2. Tokenizer Initialization
    # We need the tokenizer to pass to the data utility
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # 3. Data Loading
    # Generate dataloaders using the library function
    # We set load_cached_data=False to demonstrate processing logic,
    # though in practice True is faster for repeated runs.
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer, load_cached_data=False
    )

    # Verification: Check DataLoader sizes
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Verification: Inspect a single batch structure
    sample_batch = next(iter(train_loader))
    assert "input_ids" in sample_batch
    assert "attention_mask" in sample_batch
    assert "labels" in sample_batch
    assert sample_batch["input_ids"].shape[1] == Config.MAX_LEN
    print("Data loading verification passed.")

    # 4. Model Initialization
    print("Initializing model...")
    model = TransformerClassifier(model_name=Config.MODEL_NAME)
    model.to(Config.DEVICE)

    # Verification: Forward pass check
    # Ensure the model outputs the correct shape (batch_size, 1)
    with torch.no_grad():
        dummy_input = sample_batch["input_ids"].to(Config.DEVICE)
        dummy_mask = sample_batch["attention_mask"].to(Config.DEVICE)
        dummy_output = model(dummy_input, dummy_mask)
        assert dummy_output.shape == (
            dummy_input.shape[0],
            1,
        ), f"Expected output shape {(dummy_input.shape[0], 1)}, got {dummy_output.shape}"
    print("Model forward pass verification passed.")

    # 5. Training Setup
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    loss_fn = nn.BCEWithLogitsLoss()

    # Scheduler (optional but good practice with transformers)
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=Config.WARMUP_STEPS, num_training_steps=total_steps
    )

    # 6. Training Loop
    print(f"Training for {Config.EPOCHS} epoch(s)...")
    for epoch in range(Config.EPOCHS):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, Config.DEVICE, loss_fn
        )
        print(f"Epoch {epoch+1}/{Config.EPOCHS} - Training Loss: {avg_loss:.4f}")

        # Validate logic: Loss should be a valid number
        assert not np.isnan(avg_loss), "Training loss is NaN"

    # 7. Evaluation
    print("Evaluating on validation set...")
    val_loss, val_auc = evaluate(model, val_loader, Config.DEVICE, loss_fn)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")

    # Validate logic: AUC should be within [0, 1]
    assert 0.0 <= val_auc <= 1.0, f"Invalid AUC score: {val_auc}"

    # 8. Prediction on Test Set
    print("Generating predictions on test set...")
    test_probs = predict(model, test_loader, Config.DEVICE)

    # Verification: Check prediction count matches test set size
    # Load test metadata to compare lengths
    df_test = pd.read_csv(Config.TEST_FILE)
    assert len(test_probs) == len(
        df_test
    ), f"Mismatch in predictions: Got {len(test_probs)}, expected {len(df_test)}"

    print(f"Generated {len(test_probs)} predictions.")

    # 9. Submission Generation
    # The submission format requires the original columns with 'Insult' filled with probabilities
    print("Saving submission...")
    df_submission = df_test.copy()
    df_submission["Insult"] = test_probs

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_submission.to_csv(Config.SUBMISSION_FILE, index=False)

    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print("Demonstration complete.")


if __name__ == "__main__":
    main()
