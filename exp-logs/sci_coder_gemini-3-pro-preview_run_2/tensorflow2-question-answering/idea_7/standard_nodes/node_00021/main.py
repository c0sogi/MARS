import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Import library modules
from library.config import Config
from library.data_utils import Tokenizer, ensure_dir
from library.dataset import NQDataset
from library.trainer import Trainer
from library.inference import Evaluator
from library.model import GlobalContextPointwiseNet


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)


def ensure_vocab_exists():
    """
    Ensures the vocabulary file exists. If not, fits tokenizer on a subset of train data.
    """
    # Explicitly invalidate cache to prevent loading stale/tiny vocab (Cite debug_lesson_2)
    if os.path.exists(Config.VOCAB_CACHE_FILE):
        os.remove(Config.VOCAB_CACHE_FILE)

    if not os.path.exists(Config.VOCAB_CACHE_FILE):
        print("Vocab file not found. Building vocabulary from training data...")
        tokenizer = Tokenizer()

        # Read a subset of training data to build vocab quickly
        texts = []
        sample_size = 20000
        with open(Config.TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                entry = json.loads(line)
                texts.append(entry["question_text"])
                texts.append(entry["document_text"])

        tokenizer.fit_on_texts(
            texts, min_freq=Config.MIN_FREQ, vocab_size=Config.VOCAB_SIZE
        )
        tokenizer.save(Config.VOCAB_CACHE_FILE)
        print("Vocabulary built and saved.")


def calculate_f1(preds, truths):
    """
    Calculates Micro F1 score.
    preds: list of predicted strings
    truths: list of ground truth strings
    """
    tp = 0
    fp = 0
    fn = 0

    for p, t in zip(preds, truths):
        # Normalize
        p = str(p).strip()
        t = str(t).strip()

        if p == t:
            if p != "":
                tp += 1
            # If both are empty, it's a True Negative (doesn't affect F1)
        else:
            if p != "" and t != "":
                fp += 1
                fn += 1
            elif p != "" and t == "":
                fp += 1
            elif p == "" and t != "":
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    return f1


def validate_and_analyze(model, tokenizer):
    print("\n--- Starting Validation and Failure Analysis ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Load Validation Dataset
    # We use the validation metadata but need to process it to get candidates
    val_dataset = NQDataset(
        metadata_path=Config.VAL_META_PATH,
        raw_data_path=Config.TRAIN_DATA_PATH,  # Validation is a split of train file
        tokenizer=tokenizer,
        is_train=False,  # Use all candidates for evaluation
        load_cached_data=True,
        debug=Config.DEBUG,
        sample_size=Config.DEBUG_SAMPLE_SIZE,
    )

    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Storage for aggregation
    # We need to aggregate candidates by example_id to find the best answer
    examples = {}

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            q_seq = batch["q_seq"].to(device)
            c_seq = batch["c_seq"].to(device)

            # Forward pass
            l_logits, s_logits, e_logits, yn_logits = model(q_seq, c_seq)

            l_probs = torch.sigmoid(l_logits).cpu().numpy()
            s_probs = F.softmax(s_logits, dim=1).cpu().numpy()
            e_probs = F.softmax(e_logits, dim=1).cpu().numpy()
            yn_probs = F.softmax(yn_logits, dim=1).cpu().numpy()

            # Metadata
            example_ids = batch["example_id"]
            cand_indices = batch["candidate_index"].numpy()
            long_labels = batch["long_label"].numpy()  # 1 if gold long answer

            # We need ground truth strings.
            # In NQDataset for validation (is_train=False), we don't get exact span indices in global context easily
            # without re-parsing. However, we can infer correctness based on labels provided in batch.
            # But wait, NQDataset with is_train=False returns 'long_label' which is 1 if it is the gold candidate.
            # It also returns 'short_start' and 'short_end' relative to candidate if it is gold.

            s_starts = batch["short_start"].numpy()
            s_ends = batch["short_end"].numpy()
            yn_labels = batch["yes_no_label"].numpy()

            # Input features for failure analysis
            # Calculate lengths from sequences (non-zero elements)
            q_lens = (batch["q_seq"] != 0).sum(dim=1).numpy()
            c_lens = (batch["c_seq"] != 0).sum(dim=1).numpy()

            for i in range(len(example_ids)):
                eid = str(example_ids[i])

                if eid not in examples:
                    examples[eid] = {
                        "candidates": [],
                        "q_len": q_lens[i],
                        "gold_long_found": False,
                        "gold_short_str": "",
                        "gold_long_str": "",
                    }

                # Construct prediction info for this candidate
                # Predict Short Span
                s_idx = np.argmax(s_probs[i])
                e_idx = np.argmax(e_probs[i])
                span_score = 0.0
                pred_short_str = ""

                if s_idx <= e_idx:
                    span_score = s_probs[i][s_idx] * e_probs[i][e_idx]
                    # We use relative indices as a proxy for string matching in this simplified validation
                    pred_short_str = f"{s_idx}:{e_idx+1}"

                # Predict Yes/No
                yn_idx = np.argmax(yn_probs[i])
                if yn_idx == 1:
                    pred_short_str = "YES"
                elif yn_idx == 2:
                    pred_short_str = "NO"

                # Construct Ground Truth info
                # If this is the gold candidate (long_label == 1)
                if long_labels[i] == 1:
                    examples[eid]["gold_long_found"] = True
                    examples[eid][
                        "gold_long_str"
                    ] = f"CAND_{cand_indices[i]}"  # Abstract representation

                    # Short truth
                    if yn_labels[i] == 1:
                        examples[eid]["gold_short_str"] = "YES"
                    elif yn_labels[i] == 2:
                        examples[eid]["gold_short_str"] = "NO"
                    elif s_starts[i] != -1:
                        examples[eid]["gold_short_str"] = f"{s_starts[i]}:{s_ends[i]+1}"
                    else:
                        examples[eid]["gold_short_str"] = ""

                examples[eid]["candidates"].append(
                    {
                        "l_score": l_probs[i],
                        "span_score": span_score,
                        "pred_short_str": pred_short_str,
                        "c_idx": cand_indices[i],
                        "c_len": c_lens[i],
                    }
                )

    # Aggregation and Metric Calculation
    long_preds = []
    long_truths = []
    short_preds = []
    short_truths = []

    # For failure analysis
    fa_data = []

    for eid, data in examples.items():
        # Find best candidate by long score
        best_cand = max(data["candidates"], key=lambda x: x["l_score"])

        # Long Answer Prediction
        pred_l_str = ""
        if best_cand["l_score"] > Config.LONG_CONFIDENCE_THRESHOLD:
            pred_l_str = f"CAND_{best_cand['c_idx']}"

        # Short Answer Prediction
        pred_s_str = ""
        if best_cand["l_score"] > Config.LONG_CONFIDENCE_THRESHOLD:
            if best_cand["span_score"] > Config.SHORT_CONFIDENCE_THRESHOLD or best_cand[
                "pred_short_str"
            ] in ["YES", "NO"]:
                pred_s_str = best_cand["pred_short_str"]

        # Ground Truths
        # If gold_long_found is False, it means the answer is not in the candidates provided
        # (or dataset sampling missed it, but validation set should have it).
        # If not found, truth is empty.
        true_l_str = data["gold_long_str"] if data["gold_long_found"] else ""
        true_s_str = data["gold_short_str"] if data["gold_long_found"] else ""

        long_preds.append(pred_l_str)
        long_truths.append(true_l_str)
        short_preds.append(pred_s_str)
        short_truths.append(true_s_str)

        # Failure Analysis Data
        # Error = 1 if either long or short is wrong (strict)
        is_correct = (pred_l_str == true_l_str) and (pred_s_str == true_s_str)
        error = 0 if is_correct else 1

        fa_data.append(
            {
                "q_len": data["q_len"],
                "doc_len": best_cand["c_len"],  # Length of the selected candidate
                "error": error,
            }
        )

    # Compute F1
    f1_long = calculate_f1(long_preds, long_truths)
    f1_short = calculate_f1(short_preds, short_truths)

    # Combined Micro F1 (Treating Long and Short as separate instances per question)
    # Total TP, FP, FN
    # We can just average them or compute globally. Task says "Micro F1".
    # Let's compute globally.
    all_preds = long_preds + short_preds
    all_truths = long_truths + short_truths
    final_metric = calculate_f1(all_preds, all_truths)

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    df_fa = pd.DataFrame(fa_data)
    if (
        not df_fa.empty
        and df_fa["error"].sum() > 0
        and df_fa["error"].sum() < len(df_fa)
    ):
        corr_q, _ = pearsonr(df_fa["q_len"], df_fa["error"])
        corr_d, _ = pearsonr(df_fa["doc_len"], df_fa["error"])

        print("\n--- Failure Analysis ---")
        print(f"Correlation between Error and Question Length: {corr_q:.4f}")
        print(f"Correlation between Error and Candidate Length: {corr_d:.4f}")
    else:
        print("\n--- Failure Analysis ---")
        print("Insufficient variance in errors to calculate correlation.")


def main():
    # 1. Configuration Patching for Fast Baseline
    print("Patching Config for fast execution...")
    Config.NUM_EPOCHS = 1
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50000  # Enough to learn something, small enough for 2h
    Config.BATCH_SIZE = 64  # Safe size

    set_seed(Config.SEED)

    # 2. Ensure Vocab Exists
    ensure_vocab_exists()

    # 3. Train Model
    print("\n--- Initializing Trainer ---")
    trainer = Trainer(load_cached_data=True)
    # Force regeneration of data to fix stale cache issues (Cite debug_lesson_2)
    trainer.fit(load_cached_data=False)

    # 4. Validation and Failure Analysis
    # We need the tokenizer to load the validation set correctly
    validate_and_analyze(trainer.model, trainer.tokenizer)

    # 5. Generate Submission
    print("\n--- Generating Submission ---")
    # We pass the trained model to avoid reloading from disk if possible,
    # though Evaluator loads from disk by default. We'll let Evaluator load the best saved model.
    evaluator = Evaluator(load_cached_data=True)
    evaluator.generate_submission()

    print("\nRunfile execution completed.")


if __name__ == "__main__":
    main()
