import os
import sys
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Import from library
from library.config import Config
from library.data import get_vocabulary, Vocabulary
from library.utils import (
    load_glove_embeddings,
    tokenize_text,
    parse_html_candidates,
    f1_score,
    normalize_answer,
)
from library.models import ANBoWRanker, ConvBiDAFReader
from library.trainer import Trainer
from library.inference import InferencePipeline

# Set silent mode for pandas
pd.options.mode.chained_assignment = None


def set_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ValidationDataset(Dataset):
    def __init__(self, items, vocab):
        self.items = items
        self.vocab = vocab

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        q_ids = self.vocab.convert_tokens_to_ids(item["q_tokens"], Config.MAX_Q_LEN)
        c_ids = self.vocab.convert_tokens_to_ids(item["c_tokens"], Config.MAX_DOC_LEN)
        return {
            "q_ids": torch.tensor(q_ids, dtype=torch.long),
            "c_ids": torch.tensor(c_ids, dtype=torch.long),
            "doc_idx": item["doc_idx"],
            "cand_idx": item["cand_idx"],
            "token_start": item["token_start"],
            "token_end": item["token_end"],
        }


def get_best_span(start_probs, end_probs, max_span_len=30):
    best_score = -1.0
    best_start = 0
    best_end = 0
    seq_len = len(start_probs)
    for s in range(seq_len):
        if start_probs[s] < 0.01:
            continue
        for e in range(s, min(seq_len, s + max_span_len)):
            score = start_probs[s] * end_probs[e]
            if score > best_score:
                best_score = score
                best_start = s
                best_end = e
    return best_start, best_end, best_score


def validate_and_analyze():
    print("\n--- Starting Validation and Failure Analysis ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = get_vocabulary(load_cached_data=True)
    embedding_matrix = load_glove_embeddings(
        vocab.token_to_idx, Config.EMBEDDING_DIM, load_cached_data=True
    )

    # Load Models
    ranker = ANBoWRanker(embedding_matrix=embedding_matrix).to(device)
    reader = ConvBiDAFReader(embedding_matrix=embedding_matrix).to(device)

    if os.path.exists(Config.RANKER_MODEL_PATH):
        ranker.load_state_dict(
            torch.load(Config.RANKER_MODEL_PATH, map_location=device)
        )
    if os.path.exists(Config.READER_MODEL_PATH):
        reader.load_state_dict(
            torch.load(Config.READER_MODEL_PATH, map_location=device)
        )

    ranker.eval()
    reader.eval()

    # Load Validation Metadata
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    # Use config limit
    if Config.VAL_SAMPLE_SIZE:
        val_meta = val_meta.head(Config.VAL_SAMPLE_SIZE)

    print(f"Validating on {len(val_meta)} samples...")

    # Metrics
    total_f1 = 0.0
    count = 0

    # Analysis Data
    analysis_records = []

    # We process document by document to simulate the pipeline
    # To speed up, we can batch candidates for the ranker

    # Pre-read data to avoid IO bottleneck during inference loop
    # We will process in chunks of documents

    doc_chunk_size = 100

    for i in range(0, len(val_meta), doc_chunk_size):
        chunk = val_meta.iloc[i : i + doc_chunk_size]

        # Prepare candidates for this chunk
        chunk_candidates = []
        ground_truths = {}  # doc_idx -> annotation

        with open(Config.TRAIN_FILE, "rb") as f:
            for idx, row in chunk.iterrows():
                f.seek(row["byte_offset"])
                line = f.readline()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    q_text = entry["question_text"]
                    doc_text = entry["document_text"]
                    doc_tokens = tokenize_text(doc_text)

                    # Store GT
                    ground_truths[idx] = {
                        "annotations": entry["annotations"],
                        "doc_tokens": doc_tokens,
                        "q_tokens": tokenize_text(q_text),
                    }

                    candidates = parse_html_candidates(doc_tokens)
                    for c_idx, (start, end) in enumerate(candidates):
                        c_tokens = doc_tokens[start:end]
                        if len(c_tokens) < 5:
                            continue

                        chunk_candidates.append(
                            {
                                "doc_idx": idx,
                                "cand_idx": c_idx,
                                "q_tokens": ground_truths[idx]["q_tokens"],
                                "c_tokens": c_tokens,
                                "token_start": start,
                                "token_end": end,
                            }
                        )

                except Exception:
                    continue

        if not chunk_candidates:
            continue

        # Ranker Inference
        ranker_dataset = ValidationDataset(chunk_candidates, vocab)
        ranker_loader = DataLoader(
            ranker_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        ranker_scores = []
        with torch.no_grad():
            for batch in ranker_loader:
                q_ids = batch["q_ids"].to(device)
                c_ids = batch["c_ids"].to(device)
                logits = ranker(q_ids, c_ids)
                probs = torch.sigmoid(logits).cpu().numpy()
                ranker_scores.extend(probs)

        # Assign scores back
        for j, item in enumerate(chunk_candidates):
            item["ranker_score"] = ranker_scores[j]

        # Group by document and pick best
        df_candidates = pd.DataFrame(chunk_candidates)
        best_candidates_indices = df_candidates.groupby("doc_idx")[
            "ranker_score"
        ].idxmax()

        # Reader Inference on best candidates
        top_candidates = [chunk_candidates[idx] for idx in best_candidates_indices]

        reader_dataset = ValidationDataset(top_candidates, vocab)
        reader_loader = DataLoader(
            reader_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        reader_results = []  # (start_logits, end_logits)

        with torch.no_grad():
            for batch in reader_loader:
                q_ids = batch["q_ids"].to(device)
                c_ids = batch["c_ids"].to(device)
                start_logits, end_logits = reader(q_ids, c_ids)

                s_probs = F.softmax(start_logits, dim=-1).cpu().numpy()
                e_probs = F.softmax(end_logits, dim=-1).cpu().numpy()

                for k in range(len(s_probs)):
                    reader_results.append((s_probs[k], e_probs[k]))

        # Final Evaluation
        for k, item in enumerate(top_candidates):
            doc_idx = item["doc_idx"]
            gt = ground_truths[doc_idx]

            s_probs, e_probs = reader_results[k]

            # Get best span
            s_idx, e_idx, span_score = get_best_span(s_probs, e_probs)

            ranker_score = item["ranker_score"]
            final_confidence = ranker_score * span_score

            # Prediction
            pred_long_span = (-1, -1)
            pred_short_text = ""

            if final_confidence >= Config.INFERENCE_THRESHOLD:
                # Long
                la_start = item["token_start"]
                la_end = item["token_end"]
                pred_long_span = (la_start, la_end)

                # Short
                # e_idx is inclusive in model output, exclusive in slicing logic usually
                # The util get_best_span returns inclusive end index relative to candidate
                sa_rel_start = s_idx
                sa_rel_end = e_idx + 1  # exclusive

                sa_global_start = la_start + sa_rel_start
                sa_global_end = la_start + sa_rel_end

                # Clip
                sa_global_start = max(la_start, min(sa_global_start, la_end))
                sa_global_end = max(sa_global_start, min(sa_global_end, la_end))

                if sa_global_end > sa_global_start:
                    pred_short_text = " ".join(
                        gt["doc_tokens"][sa_global_start:sa_global_end]
                    )

            # Evaluate against GT
            # NQ Metric: Micro F1 over Long and Short
            # We treat Long and Short as separate instances for Micro F1
            # But typically NQ evaluation is complex. Here we simplify to:
            # Check if prediction matches ANY valid annotation.

            # Check Long Answer Match
            long_match = 0.0
            short_match = 0.0

            valid_long_exists = False
            valid_short_exists = False

            for ann in gt["annotations"]:
                # Long
                gt_la_start = ann["long_answer"]["start_token"]
                gt_la_end = ann["long_answer"]["end_token"]

                if gt_la_start != -1:
                    valid_long_exists = True
                    # Exact match for long answer indices
                    if pred_long_span == (gt_la_start, gt_la_end):
                        long_match = 1.0

                # Short
                if ann["short_answers"]:
                    valid_short_exists = True
                    for sa in ann["short_answers"]:
                        gt_sa_text = " ".join(
                            gt["doc_tokens"][sa["start_token"] : sa["end_token"]]
                        )
                        if f1_score(pred_short_text, gt_sa_text) > short_match:
                            short_match = f1_score(pred_short_text, gt_sa_text)
                elif ann["yes_no_answer"] != "NONE":
                    valid_short_exists = True
                    if normalize_answer(pred_short_text) == normalize_answer(
                        ann["yes_no_answer"]
                    ):
                        short_match = 1.0

            # Handle Null predictions
            if not valid_long_exists:
                # If no answer exists and we predicted nothing, that's a match (F1=1)
                # If we predicted something, F1=0
                if pred_long_span == (-1, -1):
                    long_match = 1.0
                else:
                    long_match = 0.0
            else:
                # Valid answer exists, if we predicted nothing, F1=0
                if pred_long_span == (-1, -1):
                    long_match = 0.0

            if not valid_short_exists:
                if pred_short_text == "":
                    short_match = 1.0
                else:
                    short_match = 0.0
            else:
                if pred_short_text == "":
                    short_match = 0.0

            # Average F1 for this example
            example_f1 = (long_match + short_match) / 2.0
            total_f1 += example_f1
            count += 1

            # Log for failure analysis
            analysis_records.append(
                {
                    "q_len": len(gt["q_tokens"]),
                    "doc_len": len(gt["doc_tokens"]),
                    "cand_len": item["token_end"] - item["token_start"],
                    "ranker_score": ranker_score,
                    "reader_score": span_score,
                    "f1": example_f1,
                    "error": 1.0 - example_f1,
                }
            )

    final_metric = total_f1 / count if count > 0 else 0.0
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    df_analysis = pd.DataFrame(analysis_records)
    if not df_analysis.empty:
        correlations = (
            df_analysis[
                [
                    "error",
                    "q_len",
                    "doc_len",
                    "cand_len",
                    "ranker_score",
                    "reader_score",
                ]
            ]
            .corr()["error"]
            .sort_values(ascending=False)
        )
        print("Correlation with Error Magnitude:")
        print(correlations)
    else:
        print("No analysis data available.")


def main():
    set_seed(Config.SEED)

    # 1. Train
    trainer = Trainer()
    trainer.train_ranker()
    trainer.train_reader()

    # 2. Validate & Analyze
    validate_and_analyze()

    # 3. Inference & Submission
    pipeline = InferencePipeline()
    pipeline.run_inference()


if __name__ == "__main__":
    main()
