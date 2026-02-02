import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
from transformers import get_linear_schedule_with_warmup

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_llrd_optimizer_params
from library.data import (
    get_tokenizer_and_resize,
    make_dataloaders,
    make_test_dataloader,
)
from library.model import CustomDeberta
from library.engine import train_one_epoch, validate

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    print("=== Phrase Similarity Model: Demo & Verification Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # We use debug=True to limit data size to 200 samples and n_folds to 2.
    # We explicitly set epochs=1 to ensure the run finishes very quickly.
    config = Config(debug=True, epochs=1)

    # Set a unique output directory for this run
    config.output_dir = "./working/demo_run"
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"[Setup] Device: {config.device}")
    print(f"[Setup] Debug Mode: {config.debug}")
    print(f"[Setup] Output Directory: {config.output_dir}")

    # Ensure reproducibility
    seed_everything(config.seed)

    # -------------------------------------------------------------------------
    # 2. Tokenizer & Data Preprocessing
    # -------------------------------------------------------------------------
    print("\n[Step 1] Initializing Tokenizer and Resizing for Contexts...")
    # This function loads the base tokenizer and adds unique context codes (e.g., "A47") as special tokens
    tokenizer = get_tokenizer_and_resize(config)

    # Validation: Ensure tokenizer vocab is large enough (DeBERTa-v3-large base is ~128k)
    vocab_size = len(tokenizer)
    print(f"   -> Tokenizer Vocab Size: {vocab_size}")
    if vocab_size < 128000:
        raise AssertionError(
            f"Tokenizer vocab size {vocab_size} is unexpectedly small."
        )

    print("\n[Step 2] Creating DataLoaders (Fold 0)...")
    # Generate DataLoaders. load_cached_data=False forces re-computation of folds for demonstration.
    train_loader, val_loader = make_dataloaders(
        config, tokenizer, fold=0, load_cached_data=False
    )

    print(f"   -> Train Batches: {len(train_loader)}")
    print(f"   -> Val Batches:   {len(val_loader)}")

    # Validation: Inspect the structure of a single training batch
    sample_batch = next(iter(train_loader))
    required_keys = ["input_ids", "attention_mask", "labels"]
    for key in required_keys:
        if key not in sample_batch:
            raise AssertionError(f"DataLoader batch is missing key: '{key}'")

    # Validation: Check input tensor shapes [batch_size, max_length]
    expected_shape = (config.train_batch_size, config.max_length)
    if sample_batch["input_ids"].shape != expected_shape:
        raise AssertionError(
            f"Input shape mismatch. Expected {expected_shape}, got {sample_batch['input_ids'].shape}"
        )

    print("   -> DataLoader batch structure and shapes verified.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n[Step 3] Initializing CustomDeberta Model...")
    model = CustomDeberta(config, tokenizer)
    model.to(config.device)

    # Validation: Verify Embedding Resizing
    # The model's embedding layer must match the tokenizer's new vocabulary size
    model_emb_size = model.model.embeddings.word_embeddings.weight.shape[0]
    if model_emb_size != vocab_size:
        raise AssertionError(
            f"Model embedding size ({model_emb_size}) != Tokenizer size ({vocab_size})"
        )
    print("   -> Model embedding layer successfully resized.")

    # -------------------------------------------------------------------------
    # 4. Optimizer & Scheduler Setup
    # -------------------------------------------------------------------------
    print("\n[Step 4] Configuring Optimizer (LLRD) and Scheduler...")
    # Use Layer-wise Learning Rate Decay
    optimizer_params = get_llrd_optimizer_params(
        model.model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        llrd_decay=config.llrd_decay,
    )

    optimizer = torch.optim.AdamW(
        optimizer_params, lr=config.learning_rate, eps=config.eps
    )

    num_training_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps,
    )
    print("   -> Optimizer and Scheduler initialized.")

    # -------------------------------------------------------------------------
    # 5. Training Loop (1 Epoch)
    # -------------------------------------------------------------------------
    print("\n[Step 5] Starting Training (1 Epoch)...")
    train_loss = train_one_epoch(
        model, optimizer, scheduler, train_loader, config.device, epoch=0, config=config
    )
    print(f"   -> Epoch 0 Train Loss: {train_loss:.4f}")

    # Validation: Loss should be a valid number
    if np.isnan(train_loss) or train_loss < 0:
        raise AssertionError("Training loss is NaN or negative.")

    # -------------------------------------------------------------------------
    # 6. Validation Loop
    # -------------------------------------------------------------------------
    print("\n[Step 6] Starting Validation...")
    val_loss, val_score = validate(model, val_loader, config.device, config)
    print(f"   -> Validation Loss: {val_loss:.4f}")
    print(f"   -> Pearson Score:   {val_score:.4f}")

    # Validation: Score range
    if not (-1.0 <= val_score <= 1.0):
        raise AssertionError(
            f"Pearson score {val_score} is out of valid range [-1.0, 1.0]"
        )

    # -------------------------------------------------------------------------
    # 7. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 7] Running Inference on Test Set...")
    test_loader = make_test_dataloader(config, tokenizer)

    model.eval()
    ids = []
    preds = []

    # Run inference on the first batch only for demonstration speed
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(config.device)

            # Forward pass
            outputs = model(input_ids, attention_mask, token_type_ids=token_type_ids)
            logits = outputs.logits.squeeze(-1).cpu().numpy()

            ids.extend(batch["id"])
            preds.extend(logits)

            # Stop after first batch
            break

    print(f"   -> Inference run on {len(preds)} samples.")

    # Display sample output
    submission_example = pd.DataFrame({"id": ids, "score": preds})
    print("   -> Sample Predictions:")
    print(submission_example.head())

    print("\n=== Script Completed Successfully ===")


if __name__ == "__main__":
    main()
