import os
import torch
import numpy as np
import pandas as pd
import shutil
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import from provided library files
from library.utils import set_seed, compute_spearman_metric
from library.loss import RDropLoss
from library.dataset import QADataset, CollateQA
from library.model import DualDistilRoBERTa
from library.trainer import Trainer


def test_utils():
    print("\n=== Testing Utils ===")
    set_seed(42)

    # Test Spearman Metric
    # Case 1: Perfect correlation
    y_true = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    y_pred = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    score = compute_spearman_metric(y_true, y_pred)
    print(f"Perfect correlation score: {score}")
    assert np.isclose(score, 1.0), "Spearman metric should be 1.0 for identical arrays"

    # Case 2: Inverse correlation (rank-wise)
    y_true_inv = np.array([[1.0], [2.0], [3.0]])
    y_pred_inv = np.array([[3.0], [2.0], [1.0]])
    score_inv = compute_spearman_metric(y_true_inv, y_pred_inv)
    print(f"Inverse correlation score: {score_inv}")
    assert np.isclose(
        score_inv, -1.0
    ), "Spearman metric should be -1.0 for inverse ranks"


def test_loss():
    print("\n=== Testing Loss Function ===")
    loss_fn = RDropLoss(alpha=1.0)

    batch_size = 4
    num_labels = 30

    # Dummy logits and targets
    logits1 = torch.randn(batch_size, num_labels)
    logits2 = torch.randn(batch_size, num_labels)
    targets = torch.rand(batch_size, num_labels)  # Targets in [0, 1]

    loss = loss_fn(logits1, logits2, targets)
    print(f"Computed R-Drop Loss: {loss.item()}")

    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"


def test_dataset_and_dataloader():
    print("\n=== Testing Dataset and DataLoader ===")

    # Create a small synthetic dataframe
    target_cols = [f"target_{i}" for i in range(30)]
    data = {
        "qa_id": [1, 2, 3, 4],
        "question_title": [
            "How to code?",
            "What is Python?",
            "Error in loop",
            "Help me",
        ],
        "question_body": [
            "I need help with coding.",
            "Python is a language.",
            "While loop error.",
            "Just help.",
        ],
        "answer": ["Use an IDE.", "It is a snake.", "Check condition.", "Read docs."],
    }
    for col in target_cols:
        data[col] = np.random.rand(4).astype(np.float32)

    df = pd.DataFrame(data)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    # Instantiate Dataset
    dataset = QADataset(
        df, tokenizer, target_cols=target_cols, max_length=64, is_test=False
    )

    # Check __getitem__
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
        assert key in sample, f"Missing key {key} in dataset sample"

    print("Dataset sample keys verified.")

    # Test Collator and DataLoader
    collator = CollateQA(tokenizer)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collator)

    batch = next(iter(loader))
    assert batch["q_input_ids"].shape[0] == 2, "Batch size mismatch"
    assert "labels" in batch, "Labels missing in batch"
    assert batch["labels"].shape == (2, 30), "Labels shape mismatch"

    print("DataLoader batch shape verified.")
    return loader, target_cols


def test_model():
    print("\n=== Testing Model ===")
    model = DualDistilRoBERTa(num_labels=30)

    # Create dummy inputs
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))

    # Forward pass
    logits = model(input_ids, attention_mask, input_ids, attention_mask)

    print(f"Model output shape: {logits.shape}")
    assert logits.shape == (
        batch_size,
        30,
    ), f"Expected shape ({batch_size}, 30), got {logits.shape}"


def test_trainer_integration(mock_loader, target_cols):
    print("\n=== Testing Trainer Integration ===")

    # Define temporary directories for the demo
    demo_working_dir = "./working/demo_run"
    demo_sub_dir = "./working/demo_run"

    # Clean up if exists
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)

    # Initialize Trainer
    # Note: Trainer initialization loads the full dataset by default.
    # Since the full dataset is relatively small (~4k rows), this is acceptable.
    # We will swap the loaders immediately after initialization to speed up the 'train' loop.
    trainer = Trainer(
        batch_size=4,
        max_length=64,
        working_dir=demo_working_dir,
        submission_dir=demo_sub_dir,
    )

    print("Trainer initialized.")

    # Inject the mock loader (small subset) to ensure the training loop finishes quickly
    trainer.train_loader = mock_loader
    trainer.val_loader = mock_loader
    trainer.test_loader = mock_loader

    # Update target columns in trainer to match our mock data (though names don't matter for shape)
    # The Trainer initialized with real columns, which is fine as long as count matches.
    # Our mock loader has 30 targets, real data has 30 targets.

    # Run training for 1 epoch (Phase 1 warmup only runs 1 epoch, Phase 2 runs epochs 2-8)
    # We limit total epochs to 2 to verify transition between phases without waiting too long.
    print("Starting short training run...")
    trainer.train(epochs=2)

    # Check artifacts
    best_model_path = os.path.join(demo_working_dir, "best_model.pth")
    submission_path = os.path.join(demo_sub_dir, "submission.csv")

    assert os.path.exists(best_model_path), "Best model file was not saved."
    assert os.path.exists(submission_path), "Submission file was not generated."

    # Verify submission format
    sub_df = pd.read_csv(submission_path)
    assert "qa_id" in sub_df.columns, "Submission missing qa_id column"
    assert (
        len(sub_df.columns) == 31
    ), f"Expected 31 columns in submission, got {len(sub_df.columns)}"
    print("Training and submission generation successful.")


if __name__ == "__main__":
    # 1. Verify Utilities
    test_utils()

    # 2. Verify Loss
    test_loss()

    # 3. Verify Dataset & Get Mock Loader
    mock_loader, target_cols = test_dataset_and_dataloader()

    # 4. Verify Model Architecture
    test_model()

    # 5. Verify Trainer Loop (using mock loader for speed)
    test_trainer_integration(mock_loader, target_cols)

    print("\nAll demonstrations passed successfully.")
