import os
import torch
import numpy as np
import pandas as pd
from library.utils import set_seed, load_metadata
from library.data_loader import build_tokenizer, get_dataloader
from library.modeling import DanTqpModel
from library.trainer import ModelTrainer
from library.inference import SubmissionGenerator

# --- Configuration ---
SEED = 42
BATCH_SIZE = 128
MAX_LEN = 128
EMBEDDING_DIM = 50
HIDDEN_DIM = 64
LEARNING_RATE = 1e-3
EPOCHS = 3
NEG_RATIO = 0.5  # Increased to provide more negatives for robust training
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = "./working/idea_3"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
SUBMISSION_PATH = "submission.csv"
TEST_FILE_RAW = "./input/simplified-nq-test.jsonl"


def perform_failure_analysis(model, dataloader, device):
    """
    Analyzes correlations between model error and input features (sequence lengths).
    """
    print("\n--- Performing Failure Analysis ---")
    model.eval()

    errors = []
    q_lengths = []
    c_lengths = []

    with torch.no_grad():
        for batch in dataloader:
            q_input_ids = batch["q_input_ids"].to(device)
            c_input_ids = batch["c_input_ids"].to(device)
            label_long = batch["label_long"].to(device)

            # Forward pass
            ranker_logits, _ = model(q_input_ids, c_input_ids)
            probs = torch.sigmoid(ranker_logits).squeeze(-1)

            # Calculate absolute error
            batch_errors = torch.abs(probs - label_long).cpu().numpy()
            errors.extend(batch_errors)

            # Calculate lengths (non-zero tokens)
            # Assuming padding index is 0
            q_len = (q_input_ids != 0).sum(dim=1).cpu().numpy()
            c_len = (c_input_ids != 0).sum(dim=1).cpu().numpy()

            q_lengths.extend(q_len)
            c_lengths.extend(c_len)

    # Compute correlations
    df_analysis = pd.DataFrame(
        {"error": errors, "q_length": q_lengths, "c_length": c_lengths}
    )

    corr_q = df_analysis["error"].corr(df_analysis["q_length"])
    corr_c = df_analysis["error"].corr(df_analysis["c_length"])

    print(f"Correlation between Error and Question Length: {corr_q:.4f}")
    print(f"Correlation between Error and Candidate Length: {corr_c:.4f}")


def main():
    # 1. Setup
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Running on device: {DEVICE}")

    # Clear cached train data to apply new NEG_RATIO
    train_cache = "./working/idea_3/train_flattened.parquet"
    if os.path.exists(train_cache):
        print(
            f"Removing cached training data {train_cache} to apply new configuration."
        )
        os.remove(train_cache)

    # 2. Data Preparation
    print("Loading metadata and building tokenizer...")
    train_meta = load_metadata("train")
    # Build tokenizer on a sample to be fast
    tokenizer = build_tokenizer(train_meta, sample_size=20000, load_cached_data=True)
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")

    print("Preparing DataLoaders...")
    train_loader = get_dataloader(
        split="train",
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        neg_ratio=NEG_RATIO,
        num_workers=NUM_WORKERS,
        load_cached_data=True,
    )

    val_loader = get_dataloader(
        split="val",
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        neg_ratio=NEG_RATIO,  # Not used for val logic in data_loader but required by sig
        num_workers=NUM_WORKERS,
        load_cached_data=True,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = DanTqpModel(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        padding_idx=0,
    )

    # 4. Training
    trainer = ModelTrainer(model, DEVICE, learning_rate=LEARNING_RATE)
    trainer.train(
        train_loader, val_loader, epochs=EPOCHS, patience=1, save_path=CHECKPOINT_PATH
    )

    # 5. Final Validation Assessment
    print("\n--- Final Validation Assessment ---")
    # Reload best model
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)

    # Re-run validation to get the exact metric for reporting
    val_loss, val_f1 = trainer.validate(val_loader)
    print(f"Final Validation Metric: {val_f1}")

    # 6. Failure Analysis
    perform_failure_analysis(model, val_loader, DEVICE)

    # 7. Inference and Submission
    print("\n--- Generating Submission ---")

    # Load Test Data
    test_loader = get_dataloader(
        split="test",
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        num_workers=NUM_WORKERS,
        load_cached_data=True,
    )

    generator = SubmissionGenerator(model, DEVICE, tokenizer)
    predictions = generator.predict(test_loader)

    generator.generate_submission_file(predictions, TEST_FILE_RAW, SUBMISSION_PATH)
    print("Runfile execution completed.")


if __name__ == "__main__":
    main()
