import os
import json
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
# Cite debug_lesson_1: Updating output path to match grader expectation (./submission/submission.csv)
SUBMISSION_PATH = "submission/submission.csv"
TEST_FILE_RAW = "./input/simplified-nq-test.jsonl"


def compute_official_f1(predictions, metadata_df, input_dir="./input"):
    """
    Computes the official Micro F1 score for the NQ task using validation metadata.
    """
    tp, fp, fn = 0, 0, 0

    # Group metadata by file to optimize I/O
    for file_path, group in metadata_df.groupby("file_path"):
        abs_path = os.path.join(input_dir, file_path)
        if not os.path.exists(abs_path):
            continue

        with open(abs_path, "rb") as f:
            for _, row in group.iterrows():
                ex_id = row["example_id"]

                # Parse Ground Truth
                try:
                    anns = json.loads(row["annotations"])
                except:
                    anns = []

                gt_longs = set()
                gt_shorts = set()

                for ann in anns:
                    # Long Answer
                    la = ann.get("long_answer", {})
                    if la.get("start_token", -1) != -1:
                        gt_longs.add(f"{la['start_token']}:{la['end_token']}")

                    # Short Answers
                    sas = ann.get("short_answers", [])
                    for sa in sas:
                        gt_shorts.add(f"{sa['start_token']}:{sa['end_token']}")

                    # Yes/No
                    yn = ann.get("yes_no_answer", "NONE")
                    if yn != "NONE":
                        gt_shorts.add(yn)

                # Get Prediction
                pred_long = ""
                pred_short = ""

                if ex_id in predictions:
                    p = predictions[ex_id]
                    # Thresholds: Long > 0.5, Short > 0.1 (matching SubmissionGenerator defaults)
                    if p["long_score"] > 0.5:
                        # Need to retrieve candidate info to map local span to global tokens
                        f.seek(row["byte_offset"])
                        line = f.readline()
                        try:
                            data = json.loads(line)
                            candidates = data.get("long_answer_candidates", [])
                            c_idx = p["candidate_index"]

                            if c_idx < len(candidates):
                                cand = candidates[c_idx]
                                pred_long = f"{cand['start_token']}:{cand['end_token']}"

                                if (
                                    p["short_score"] > 0.1
                                    and p["short_span"] is not None
                                ):
                                    s_local, e_local = p["short_span"]
                                    s_global = cand["start_token"] + s_local
                                    e_global = cand["start_token"] + e_local + 1
                                    if e_global <= cand["end_token"]:
                                        pred_short = f"{s_global}:{e_global}"
                        except:
                            pass

                # Score Long Answer
                if pred_long:
                    if pred_long in gt_longs:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if gt_longs:
                        fn += 1

                # Score Short Answer
                if pred_short:
                    if pred_short in gt_shorts:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if gt_shorts:
                        fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return f1


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

    # Initialize Generator
    generator = SubmissionGenerator(model, DEVICE, tokenizer)

    # Run inference on Validation set
    print("Running inference on Validation set for Official F1...")
    val_predictions = generator.predict(val_loader)

    # Load Validation Metadata for Ground Truth
    val_meta = load_metadata("val")

    # Compute Official Metric
    official_f1 = compute_official_f1(val_predictions, val_meta)
    print(f"Final Validation Metric: {official_f1}")

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
