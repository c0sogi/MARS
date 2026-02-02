import os
import torch
import pandas as pd
import numpy as np
from library.utils import seed_everything, compute_spearman_metric
from library.dataset import get_loaders, ALL_TARGETS
from library.model import DebertaDualHead
from library.engine import Engine, run_task


def test_utils():
    """Verifies utility functions."""
    print("\n=== Testing Utils ===")

    # Test seed_everything
    seed_everything(42)
    t1 = torch.rand(1)
    seed_everything(42)
    t2 = torch.rand(1)
    assert torch.equal(
        t1, t2
    ), "seed_everything failed to produce deterministic results"

    # Test compute_spearman_metric
    # Case: Perfect correlation
    preds = np.random.rand(10, 30)
    targets = preds.copy()
    score = compute_spearman_metric(preds, targets)
    assert np.isclose(score, 1.0), f"Expected 1.0 for perfect correlation, got {score}"

    # Case: Split input (Question + Answer heads)
    q_preds = np.random.rand(10, 21)
    a_preds = np.random.rand(10, 9)
    targets = np.concatenate([q_preds, a_preds], axis=1)
    score = compute_spearman_metric([q_preds, a_preds], targets)
    assert np.isclose(score, 1.0), f"Expected 1.0 for split input, got {score}"

    print("Utils verification passed.")


def create_mock_data():
    """
    Creates tiny cached parquet files in ./working/idea_3/ to force the
    library to use a small dataset for speed.
    """
    print("\n=== Creating Mock Data ===")
    cache_dir = "./working/idea_3/"
    os.makedirs(cache_dir, exist_ok=True)

    # Create dummy data (16 samples)
    num_samples = 16
    data = {
        "qa_id": np.arange(num_samples),
        "question_title": [f"Title {i}" for i in range(num_samples)],
        "question_body": [f"Body content {i}" for i in range(num_samples)],
        "answer": [f"Answer content {i}" for i in range(num_samples)],
    }

    # Add random targets
    for col in ALL_TARGETS:
        data[col] = np.random.rand(num_samples).astype(np.float32)

    df = pd.DataFrame(data)

    # Apply preprocessing logic expected by QuestDataset
    df["question_input"] = df["question_title"] + " " + df["question_body"]
    df["answer_input"] = df["answer"]

    # Save as parquet files where preprocess_and_cache expects them
    # This mocks the result of the preprocessing step
    df.to_parquet(os.path.join(cache_dir, "train_processed.parquet"), index=False)
    df.to_parquet(os.path.join(cache_dir, "val_processed.parquet"), index=False)

    # For test, drop targets to simulate real inference scenario
    df_test = df.drop(columns=ALL_TARGETS)
    df_test.to_parquet(os.path.join(cache_dir, "test_processed.parquet"), index=False)

    print(f"Mock data created in {cache_dir}")


def test_dataset_and_loader():
    """Verifies Data Loading pipeline."""
    print("\n=== Testing Dataset and DataLoader ===")

    # Loaders will pick up the mock parquet files created above
    train_loader, val_loader, test_loader = get_loaders(
        tokenizer_name="microsoft/deberta-v3-base",
        batch_size=4,
        max_length=32,  # Short length for speed
        load_cached_data=True,
        num_workers=0,
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify keys
    expected_keys = [
        "q_input_ids",
        "q_attention_mask",
        "a_input_ids",
        "a_attention_mask",
        "qa_id",
        "q_labels",
        "a_labels",
    ]
    for k in expected_keys:
        assert k in batch, f"Batch missing key: {k}"

    # Verify shapes
    # Batch size is 4, Q targets=21, A targets=9
    assert batch["q_input_ids"].shape[0] == 4
    assert batch["q_labels"].shape == (4, 21)
    assert batch["a_labels"].shape == (4, 9)

    print("Dataset and DataLoader verification passed.")
    return batch


def test_model(batch):
    """Verifies Model architecture and forward pass."""
    print("\n=== Testing Model ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DebertaDualHead(model_name="microsoft/deberta-v3-base")
    model.to(device)
    model.eval()

    # Move batch to device
    q_ids = batch["q_input_ids"].to(device)
    q_mask = batch["q_attention_mask"].to(device)
    a_ids = batch["a_input_ids"].to(device)
    a_mask = batch["a_attention_mask"].to(device)

    with torch.no_grad():
        q_logits, a_logits = model(q_ids, q_mask, a_ids, a_mask)

    # Verify output shapes
    # q_logits: (Batch, 21)
    # a_logits: (Batch, 9)
    assert q_logits.shape == (4, 21), f"Expected q_logits (4, 21), got {q_logits.shape}"
    assert a_logits.shape == (4, 9), f"Expected a_logits (4, 9), got {a_logits.shape}"

    print("Model forward pass passed.")
    return model


def test_engine_logic(model, batch):
    """Verifies Engine methods (validate/predict)."""
    print("\n=== Testing Engine Logic ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    engine = Engine(model, device)

    # Create a dummy loader yielding the single batch
    class MockLoader:
        def __iter__(self):
            yield batch

    loader = MockLoader()

    # Test Validate
    score = engine.validate(loader)
    assert isinstance(score, float), "Validate should return a float score"
    print(f"Validation score: {score:.4f}")

    # Test Predict
    qa_ids, preds = engine.predict(loader)
    assert len(qa_ids) == 4
    assert preds.shape == (4, 30), "Prediction shape mismatch"

    print("Engine logic verification passed.")


def test_full_run():
    """Executes the full task pipeline using the mock data."""
    print("\n=== Testing Full Run Task ===")

    # run_task initializes its own model and loaders.
    # Because we cached the mock data, this will run on the 16-sample dataset.
    run_task(epochs=1, batch_size=4, lr=1e-5, load_cached_data=True)

    # Verify submission file
    sub_path = "./submission/submission.csv"
    assert os.path.exists(sub_path), "Submission file not found"

    df_sub = pd.read_csv(sub_path)
    # We expect 16 rows (size of mock test set)
    assert len(df_sub) == 16, f"Expected 16 rows in submission, got {len(df_sub)}"
    assert df_sub.shape[1] == 31, "Expected 31 columns (qa_id + 30 targets)"

    print("Full pipeline execution passed.")


if __name__ == "__main__":
    # 1. Verify utility functions
    test_utils()

    # 2. Setup Mock Data (Essential for speed)
    create_mock_data()

    # 3. Verify Dataset Loading
    batch = test_dataset_and_loader()

    # 4. Verify Model Architecture
    model = test_model(batch)

    # 5. Verify Engine Components
    test_engine_logic(model, batch)

    # 6. Verify End-to-End Pipeline
    test_full_run()

    print("\nAll tests passed successfully.")
