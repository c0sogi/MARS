import os
import sys
import pandas as pd
import torch
import torch.optim as optim
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import load_dataset_df, ChatbotDataset, CollateFn
from library.model import SiameseDeberta
from library.engine import train_fn, eval_fn, inference_fn

# Initialize Logger
logger = get_logger("demo_script")


def main():
    # 1. Setup and Configuration
    print("Setting up environment...")
    seed_everything(Config.SEED)

    # Override Config for speed in this demo
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8
    # We will use a small subset of data
    SUBSET_SIZE_TRAIN = 64
    SUBSET_SIZE_VAL = 32
    SUBSET_SIZE_TEST = 32

    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading (Manual subsetting for speed)
    print("Loading and subsetting data...")

    # Load raw dataframes using library function
    # Note: load_dataset_df handles caching and augmentation logic
    train_df = load_dataset_df("train", load_cached_data=False)
    val_df = load_dataset_df("val", load_cached_data=False)
    test_df = load_dataset_df("test", load_cached_data=False)

    # Slice dataframes
    train_df = train_df.iloc[:SUBSET_SIZE_TRAIN].reset_index(drop=True)
    val_df = val_df.iloc[:SUBSET_SIZE_VAL].reset_index(drop=True)
    test_df = test_df.iloc[:SUBSET_SIZE_TEST].reset_index(drop=True)

    print(f"Train subset size: {len(train_df)}")
    print(f"Val subset size: {len(val_df)}")
    print(f"Test subset size: {len(test_df)}")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Create Datasets
    train_dataset = ChatbotDataset(train_df, tokenizer, Config.MAX_LEN, is_test=False)
    val_dataset = ChatbotDataset(val_df, tokenizer, Config.MAX_LEN, is_test=False)
    test_dataset = ChatbotDataset(test_df, tokenizer, Config.MAX_LEN, is_test=True)

    # Create Collator
    collator = CollateFn(tokenizer)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for simple debugging/demo
        collate_fn=collator,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    # 3. Verification: Inspect a Batch
    print("Verifying data batch structure...")
    batch = next(iter(train_loader))

    # Check keys
    expected_keys = [
        "input_ids_a",
        "attention_mask_a",
        "input_ids_b",
        "attention_mask_b",
        "scalars",
        "target",
    ]
    for key in expected_keys:
        assert key in batch, f"Missing key in batch: {key}"

    # Check shapes
    assert batch["input_ids_a"].shape == batch["input_ids_b"].shape
    assert batch["scalars"].shape == (
        Config.TRAIN_BATCH_SIZE,
        3,
    ), f"Scalar shape mismatch: {batch['scalars'].shape}"
    assert batch["target"].shape == (
        Config.TRAIN_BATCH_SIZE,
        3,
    ), f"Target shape mismatch: {batch['target'].shape}"

    print("Batch verification passed.")

    # 4. Model Initialization
    print("Initializing model...")
    model = SiameseDeberta()
    model.to(device)

    # 5. Verification: Forward Pass
    print("Verifying model forward pass...")
    model.eval()
    with torch.no_grad():
        # Move batch to device
        input_ids_a = batch["input_ids_a"].to(device)
        attention_mask_a = batch["attention_mask_a"].to(device)
        input_ids_b = batch["input_ids_b"].to(device)
        attention_mask_b = batch["attention_mask_b"].to(device)
        scalars = batch["scalars"].to(device)

        token_type_ids_a = batch.get("token_type_ids_a")
        if token_type_ids_a is not None:
            token_type_ids_a = token_type_ids_a.to(device)

        token_type_ids_b = batch.get("token_type_ids_b")
        if token_type_ids_b is not None:
            token_type_ids_b = token_type_ids_b.to(device)

        logits = model(
            input_ids_a=input_ids_a,
            attention_mask_a=attention_mask_a,
            input_ids_b=input_ids_b,
            attention_mask_b=attention_mask_b,
            scalars=scalars,
            token_type_ids_a=token_type_ids_a,
            token_type_ids_b=token_type_ids_b,
        )

    assert logits.shape == (
        Config.TRAIN_BATCH_SIZE,
        3,
    ), f"Logits shape mismatch: {logits.shape}"
    print("Forward pass verification passed.")

    # 6. Training Loop
    print("Starting training loop (1 epoch)...")

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Simple scheduler
    num_training_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * Config.WARMUP_RATIO),
        num_training_steps=num_training_steps,
    )

    train_loss = train_fn(model, train_loader, optimizer, scheduler, device, epoch=1)
    print(f"Train Loss: {train_loss:.4f}")

    # 7. Evaluation
    print("Starting evaluation...")
    val_loss = eval_fn(model, val_loader, device)
    print(f"Validation Loss: {val_loss:.4f}")

    # 8. Inference
    print("Starting inference on test set...")
    predictions = inference_fn(model, test_loader, device)

    assert predictions.shape == (
        len(test_df),
        3,
    ), f"Prediction shape mismatch: {predictions.shape}"
    # Verify probabilities sum to 1 (approx)
    sums = predictions.sum(axis=1)
    assert pd.Series(sums).between(0.99, 1.01).all(), "Predictions do not sum to 1."

    print("Inference complete.")

    # 9. Submission Generation
    print("Generating submission file...")
    submission_df = pd.DataFrame(
        predictions, columns=["winner_model_a", "winner_model_b", "winner_tie"]
    )
    submission_df["id"] = test_df["id"]

    # Reorder columns to match format: id, winner_model_a, winner_model_b, winner_tie
    submission_df = submission_df[
        ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    ]

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_path = Config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")

    # Final check
    saved_df = pd.read_csv(submission_path)
    print("Submission Head:")
    print(saved_df.head())
    assert len(saved_df) == len(test_df), "Submission length mismatch."
    assert list(saved_df.columns) == [
        "id",
        "winner_model_a",
        "winner_model_b",
        "winner_tie",
    ], "Column mismatch."

    print("\nSUCCESS: All steps completed and verified.")


if __name__ == "__main__":
    main()
