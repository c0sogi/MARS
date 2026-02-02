import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import from the provided library files
from library.utils import seed_everything, compute_spearmanr
from library.data import (
    QADataset,
    DynamicPaddingCollator,
    preprocess_df,
    prepare_loaders,
    MODEL_NAME,
)
from library.model import LoRADebertaDualEncoder
from library.train import train_fn, eval_fn, predict_fn


def main():
    print("Starting demonstration script...")

    # 1. Setup
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ==========================================
    # 2. Verify Metric
    # ==========================================
    print("\n--- Verifying Metric (Spearman Correlation) ---")
    # Generate random predictions and targets [0, 1]
    dummy_preds = np.random.rand(100, 30)
    dummy_targets = np.random.rand(100, 30)

    score = compute_spearmanr(dummy_preds, dummy_targets)
    print(f"Computed Spearman Score: {score:.4f}")

    assert isinstance(score, float), "Score must be a float"
    assert -1.0 <= score <= 1.0, "Score must be between -1 and 1"
    print("Metric verification passed.")

    # ==========================================
    # 3. Verify Data Components (Dataset & Collator)
    # ==========================================
    print("\n--- Verifying Data Components ---")

    # Create dummy data
    target_cols = [f"target_{i}" for i in range(30)]
    dummy_data = {
        "qa_id": [1, 2, 3],
        "question_title": ["How to code?", "What is ML?", "Error in python"],
        "question_body": [
            "I need help with python.",
            "Explain machine learning.",
            "Import error.",
        ],
        "answer": ["Use print()", "It is stats.", "Check path."],
    }
    # Add target columns
    for col in target_cols:
        dummy_data[col] = np.random.rand(3)

    df_dummy = pd.DataFrame(dummy_data)

    # Preprocess (creates 'question_text')
    df_dummy = preprocess_df(df_dummy)

    # Initialize Tokenizer
    # Note: This requires internet access to download the tokenizer config.
    # If offline, this assumes the model is cached.
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Instantiate Dataset
    dataset = QADataset(df_dummy, tokenizer, target_cols=target_cols, max_len=128)

    # Test __getitem__
    sample = dataset[0]
    required_keys = [
        "qa_id",
        "q_input_ids",
        "q_attention_mask",
        "a_input_ids",
        "a_attention_mask",
        "labels",
    ]
    for key in required_keys:
        assert key in sample, f"Missing key in dataset item: {key}"

    print("Dataset __getitem__ structure verified.")

    # Test Collator
    collator = DynamicPaddingCollator(tokenizer)
    batch_list = [dataset[i] for i in range(3)]
    batch = collator(batch_list)

    # Check shapes
    # Batch size is 3
    assert batch["q_input_ids"].shape[0] == 3
    assert batch["a_input_ids"].shape[0] == 3
    assert batch["labels"].shape == (3, 30)
    print("Collator batching verified.")

    # ==========================================
    # 4. Verify Model
    # ==========================================
    print("\n--- Verifying Model Architecture ---")
    model = LoRADebertaDualEncoder(model_name=MODEL_NAME, num_labels=30)
    model.to(device)

    # Move batch to device
    batch_device = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }

    # Forward pass
    logits = model(
        q_input_ids=batch_device["q_input_ids"],
        q_attention_mask=batch_device["q_attention_mask"],
        a_input_ids=batch_device["a_input_ids"],
        a_attention_mask=batch_device["a_attention_mask"],
    )

    assert logits.shape == (3, 30), f"Expected logits shape (3, 30), got {logits.shape}"
    print("Model forward pass verified.")

    # ==========================================
    # 5. Integration Test (Training Loop)
    # ==========================================
    print("\n--- Running Integration Test (Mini-Training Loop) ---")

    # Load real data loaders (but we will only use a subset to save time)
    # prepare_loaders handles caching and preprocessing
    train_loader, val_loader, test_loader, real_target_cols = prepare_loaders(
        load_cached_data=True,
        batch_size=4,
        max_len=128,  # Reduced max_len for speed
        seed=42,
    )

    # Create mini-loaders (list of batches) to simulate a very short epoch
    # We take 2 batches from each
    mini_train = [next(iter(train_loader)) for _ in range(2)]
    mini_val = [next(iter(val_loader)) for _ in range(2)]
    mini_test = [next(iter(test_loader)) for _ in range(2)]

    print(f"Created mini-loaders with {len(mini_train)} batches each.")

    # Re-initialize model for the correct number of targets in the real dataset
    model = LoRADebertaDualEncoder(
        model_name=MODEL_NAME, num_labels=len(real_target_cols)
    )
    model.to(device)

    # Optimizer setup
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=10
    )

    # 5a. Train Step
    print("Running training step...")
    train_loss = train_fn(model, mini_train, optimizer, scheduler, device, loss_fn)
    print(f"Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # 5b. Eval Step
    print("Running evaluation step...")
    val_loss, val_score = eval_fn(model, mini_val, device, loss_fn)
    print(f"Val Loss: {val_loss:.4f} | Val Spearman: {val_score:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"

    # 5c. Predict Step
    print("Running prediction step...")
    preds, qa_ids = predict_fn(model, mini_test, device)

    assert (
        preds.shape[0] == len(mini_test) * 4
    ), "Prediction count mismatch (2 batches * 4 size)"
    assert preds.shape[1] == len(real_target_cols), "Prediction target columns mismatch"

    # 5d. Generate Submission
    print("Generating sample submission...")
    submission_df = pd.DataFrame(preds, columns=real_target_cols)
    submission_df.insert(0, "qa_id", qa_ids)

    os.makedirs("./working", exist_ok=True)
    sub_path = "./working/demo_submission.csv"
    submission_df.to_csv(sub_path, index=False)

    assert os.path.exists(sub_path), "Submission file was not created"
    print(f"Demo submission saved to {sub_path}")
    print(f"Submission shape: {submission_df.shape}")

    print("\nAll demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    main()
