import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, compute_spearmanr_score
from library.dataset import load_data, QuestDataset, Collate
from library.model import QuestModel
from library.train import train_fn, valid_fn, inference_fn


def main():
    # 1. Setup and Configuration Override for Speed
    print("--- Setting up environment and configuration ---")
    seed_everything(42)

    # Suppress verbose logs
    transformers.logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Override Config for a fast demo run
    Config.debug = True
    Config.debug_sample_size = 50  # Use only 50 samples
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.max_len = 128  # Reduce sequence length for speed
    Config.working_dir = "./working/demo_run/"
    Config.output_model_path = os.path.join(Config.working_dir, "best_model.pth")

    os.makedirs(Config.working_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Loading
    print("\n--- Testing Data Loading ---")
    train_df, val_df, test_df = load_data(load_cached_data=False)

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # Assertions to ensure data loading logic is correct
    assert (
        len(train_df) == Config.debug_sample_size
    ), "Train DF size mismatch in debug mode"
    assert "question_title" in train_df.columns
    assert "answer" in train_df.columns
    # Check if targets exist in train
    assert all(col in train_df.columns for col in Config.target_cols)

    # 3. Tokenizer, Dataset, and Collate
    print("\n--- Testing Dataset and Collation ---")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Create Datasets
    train_dataset = QuestDataset(train_df, tokenizer, is_test=False)
    test_dataset = QuestDataset(test_df, tokenizer, is_test=True)

    # Test __getitem__
    sample_item = train_dataset[0]
    assert "q_input_ids" in sample_item
    assert "a_input_ids" in sample_item
    assert "labels" in sample_item
    assert isinstance(
        sample_item["q_input_ids"], list
    ), "Dataset should return lists for dynamic padding"

    # Test Collate
    collate_fn = Collate(tokenizer)
    batch_size = 4
    raw_batch = [train_dataset[i] for i in range(batch_size)]
    collated_batch = collate_fn(raw_batch)

    # Verify Collated Batch
    assert "q_input_ids" in collated_batch
    assert isinstance(collated_batch["q_input_ids"], torch.Tensor)
    assert collated_batch["q_input_ids"].shape[0] == batch_size
    # Check if padding worked (sequence length should be consistent within batch)
    assert (
        collated_batch["q_input_ids"].shape[1]
        == collated_batch["q_attention_mask"].shape[1]
    )
    assert collated_batch["labels"].shape == (batch_size, 30)

    print("Dataset and Collate checks passed.")

    # 4. Model Initialization and Forward Pass
    print("\n--- Testing Model Architecture ---")
    model = QuestModel()
    model.to(device)

    # Move batch to device
    q_input_ids = collated_batch["q_input_ids"].to(device)
    q_mask = collated_batch["q_attention_mask"].to(device)
    a_input_ids = collated_batch["a_input_ids"].to(device)
    a_mask = collated_batch["a_attention_mask"].to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(q_input_ids, q_mask, a_input_ids, a_mask)

    assert logits.shape == (
        batch_size,
        30,
    ), f"Output shape mismatch. Expected ({batch_size}, 30), got {logits.shape}"
    print("Model forward pass successful.")

    # 5. Training Loop Simulation
    print("\n--- Testing Training Loop ---")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        train_dataset,  # Use train as val for demo to ensure size consistency
        batch_size=Config.valid_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Setup Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Run one training epoch
    avg_loss = train_fn(
        model,
        train_loader,
        optimizer,
        scheduler=None,
        criterion=criterion,
        device=device,
        epoch=0,
    )
    print(f"Training Epoch 1 Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Training loss is NaN"

    # Run validation
    val_loss, val_preds, val_targets = valid_fn(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")
    assert val_preds.shape == (len(train_dataset), 30)

    # Save dummy model for inference test
    torch.save(model.state_dict(), Config.output_model_path)
    print("Training loop simulation successful.")

    # 6. Inference
    print("\n--- Testing Inference ---")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Reload model to ensure state dict loading works
    model.load_state_dict(torch.load(Config.output_model_path, map_location=device))
    model.to(device)

    test_preds = inference_fn(model, test_loader, device)

    assert test_preds.shape == (len(test_dataset), 30)
    assert (test_preds >= 0).all() and (
        test_preds <= 1
    ).all(), "Predictions must be probabilities [0, 1]"

    # Create submission file
    submission = pd.DataFrame(test_preds, columns=Config.target_cols)
    submission.insert(0, "qa_id", test_df["qa_id"])

    # Verify submission format
    assert len(submission) == len(test_df)
    assert submission.shape[1] == 31  # qa_id + 30 targets
    print("Inference and submission generation successful.")

    # 7. Metric Calculation
    print("\n--- Testing Metric Calculation ---")
    # Create synthetic targets and predictions
    # Case 1: Perfect correlation
    synth_targets = np.random.rand(100, 30)
    synth_preds = synth_targets.copy()
    score = compute_spearmanr_score(synth_preds, synth_targets)
    print(f"Perfect Score: {score:.4f}")
    assert np.isclose(score, 1.0), "Perfect correlation should be 1.0"

    # Case 2: Random noise
    synth_preds_random = np.random.rand(100, 30)
    score_random = compute_spearmanr_score(synth_preds_random, synth_targets)
    print(f"Random Score: {score_random:.4f}")
    assert -1.0 <= score_random <= 1.0, "Score must be between -1 and 1"

    print("Metric calculation verified.")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    main()
