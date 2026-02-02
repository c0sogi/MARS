import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import from the provided library files
from library.config import Config
from library.data import prepare_loaders
from library.model import CustomModel
from library.utils import set_seed, get_optimizer_grouped_parameters
from library.engine import train_one_epoch, valid_one_epoch
from library.inference import predict


def run_demo():
    # 1. Setup and Configuration Overrides for Speed
    print(">>> 1. Configuring for Demo Run")
    # Enable debug mode to use a small subset of data (100 train, 50 val)
    Config.debug = True
    # Reduce epochs to 1 for speed
    Config.epochs = 1
    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    set_seed(Config.seed)
    device = Config.device
    print(f"Device: {device}")
    print(f"Debug Mode: {Config.debug}")

    # 2. Data Loading
    print("\n>>> 2. Preparing Data Loaders")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # prepare_loaders will handle caching and subsampling due to Config.debug=True
    train_loader, val_loader, test_loader = prepare_loaders(
        tokenizer=tokenizer, load_cached_data=True, debug=Config.debug
    )

    # Verify Data Loader
    print("Verifying Train Loader batch structure...")
    batch = next(iter(train_loader))

    # Assertions for batch structure
    assert "input_ids" in batch, "Batch missing input_ids"
    assert "attention_mask" in batch, "Batch missing attention_mask"
    assert "labels" in batch, "Batch missing labels"

    # Check shapes
    batch_size = batch["input_ids"].size(0)
    seq_len = batch["input_ids"].size(1)
    print(f"Batch Size: {batch_size}, Sequence Length: {seq_len}")

    assert batch["input_ids"].shape == (batch_size, seq_len)
    assert batch["attention_mask"].shape == (batch_size, seq_len)
    assert batch["labels"].shape == (batch_size,)

    # 3. Model Initialization
    print("\n>>> 3. Initializing Model")
    model = CustomModel()
    model.to(device)

    # Verify Forward Pass
    print("Verifying forward pass...")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    print(f"Output Shape: {outputs.shape}")
    # Output should be (batch_size, 1) for regression
    assert outputs.shape == (
        batch_size,
        1,
    ), f"Expected output shape {(batch_size, 1)}, got {outputs.shape}"

    # 4. Training Loop Demonstration
    print("\n>>> 4. Running Training Loop (1 Epoch)")

    # Setup Optimizer and Scheduler
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model,
        learning_rate=Config.learning_rate,
        weight_decay=Config.weight_decay,
        llrd_decay=Config.llrd_decay,
    )
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=Config.learning_rate, eps=Config.eps
    )

    num_train_steps = len(train_loader) * Config.epochs
    num_warmup_steps = int(num_train_steps * Config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # Run Train Step
    train_loss = train_one_epoch(
        model, optimizer, scheduler, train_loader, device, epoch=0
    )
    print(f"Train Loss: {train_loss:.4f}")
    assert isinstance(train_loss, float), "Train loss should be a float"
    assert train_loss > 0, "Train loss should be positive"

    # Run Validation Step
    val_loss, val_pearson = valid_one_epoch(model, val_loader, device)
    print(f"Val Loss: {val_loss:.4f}, Val Pearson: {val_pearson:.4f}")
    assert isinstance(val_pearson, float), "Pearson score should be a float"

    # 5. Save Model for Inference
    print("\n>>> 5. Saving Model for Inference")
    # The predict function loads the model from disk, so we must save it first.
    torch.save(model.state_dict(), Config.model_path)
    assert os.path.exists(Config.model_path), "Model file was not saved correctly"
    print(f"Model saved to {Config.model_path}")

    # 6. Inference Pipeline
    print("\n>>> 6. Running Inference Pipeline")
    # predict() inside library/inference.py forces debug=False for the test set loader,
    # ensuring we predict on the full test set. It loads the model we just saved.
    predict(load_cached_data=True)

    # 7. Verification of Submission
    print("\n>>> 7. Verifying Submission File")
    submission_path = Config.submission_path

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")
    print("First 3 rows:")
    print(df_sub.head(3))

    # Check columns
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert "score" in df_sub.columns, "Submission missing 'score' column"

    # Check value range
    min_score = df_sub["score"].min()
    max_score = df_sub["score"].max()
    print(f"Score Range: [{min_score}, {max_score}]")
    assert min_score >= 0, "Scores contain negative values"
    assert max_score <= 1, "Scores exceed 1.0"

    # Check count (Test set has 3648 rows)
    expected_count = 3648
    assert (
        len(df_sub) == expected_count
    ), f"Expected {expected_count} predictions, got {len(df_sub)}"

    print("\n>>> Demo Completed Successfully!")


if __name__ == "__main__":
    run_demo()
