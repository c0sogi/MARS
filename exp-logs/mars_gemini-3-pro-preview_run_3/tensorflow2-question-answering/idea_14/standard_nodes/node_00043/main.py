import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# Import library modules
from library import config
from library import text_utils
from library import data_factory
from library import training_utils
from library import inference_utils
from library.ranker_net import KMaxInteractionRanker
from library.reader_net import HighwayCoAttentionReader

# -----------------------------------------------------------------------------
# Configuration Overrides for Fast Baseline
# -----------------------------------------------------------------------------
config.DEBUG_SAMPLE_SIZE = 2000  # Limit data size for speed
config.NUM_EPOCHS = 1  # Single epoch for baseline speed
config.BATCH_SIZE = 32  # Adjust batch size
config.EARLY_STOPPING_PATIENCE = 1

# Ensure reproducibility
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Validation Helper Functions
# -----------------------------------------------------------------------------


def get_ground_truth(record):
    """Extracts ground truth spans from a raw record."""
    long_answers = []
    short_answers = []

    for ann in record["annotations"]:
        # Long answer
        la = ann["long_answer"]
        if la["start_token"] != -1:
            long_answers.append(f"{la['start_token']}:{la['end_token']}")

        # Short answer
        if ann["short_answers"]:
            # Take first span for simplicity in baseline
            sa = ann["short_answers"][0]
            short_answers.append(f"{sa['start_token']}:{sa['end_token']}")
        elif ann["yes_no_answer"] != "NONE":
            short_answers.append(ann["yes_no_answer"])

    return long_answers, short_answers


def compute_f1(preds, ground_truths):
    """Computes F1 for a single prediction against a list of ground truths."""
    # If no ground truth, expected prediction is empty string (NULL)
    if not ground_truths:
        return 1.0 if preds == config.NULL_PREDICTION_STRING else 0.0

    # If ground truth exists but prediction is NULL
    if preds == config.NULL_PREDICTION_STRING:
        return 0.0

    # Exact match check
    if preds in ground_truths:
        return 1.0
    return 0.0


def run_validation_inference(ranker, reader, vocab, embeddings):
    """
    Runs inference on the validation set to compute metrics and perform failure analysis.
    """
    print("\n--- Running Validation Inference ---")
    ranker.eval()
    reader.eval()

    val_meta_df = pd.read_csv(config.VAL_METADATA_PATH)
    # Cite debug_lesson_2: Validate on the Entire Dataset
    # if config.DEBUG_SAMPLE_SIZE:
    #     val_meta_df = val_meta_df.iloc[: config.DEBUG_SAMPLE_SIZE]

    f1_scores = []
    meta_features = {"q_len": [], "doc_len": [], "error": []}

    # We process validation samples one by one to simulate the full pipeline
    # This is slower but necessary to accurately measure the pipeline's F1

    with open(config.TRAIN_DATA_FILE, "rb") as f:
        for idx, row in val_meta_df.iterrows():
            offset = row["byte_offset"]
            f.seek(offset)
            line = f.readline()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            q_text = record["question_text"]
            doc_tokens = text_utils.tokenize(record["document_text"])

            # Ground Truth
            gt_long, gt_short = get_ground_truth(record)

            # 1. Candidate Generation
            candidates = text_utils.segment_document(doc_tokens)
            if not candidates:
                f1_scores.append(1.0 if not gt_long and not gt_short else 0.0)
                continue

            # 2. Ranking
            q_indices = text_utils.text_to_indices(q_text, vocab, config.MAX_Q_LEN)
            q_tensor = torch.tensor([q_indices], dtype=torch.long).to(DEVICE)

            cand_tensors = []
            valid_candidates = []

            for cand in candidates:
                if len(cand["text"].split()) < config.MIN_DOC_LEN:
                    continue
                c_idx = text_utils.text_to_indices(
                    cand["text"], vocab, config.MAX_DOC_LEN
                )
                cand_tensors.append(c_idx)
                valid_candidates.append(cand)

            if not valid_candidates:
                f1_scores.append(1.0 if not gt_long and not gt_short else 0.0)
                continue

            c_tensor = torch.tensor(cand_tensors, dtype=torch.long).to(DEVICE)

            # Repeat Q tensor to match candidates
            q_batch = q_tensor.repeat(len(valid_candidates), 1)

            with torch.no_grad():
                rank_scores = ranker(q_batch, c_tensor)
                best_idx = torch.argmax(rank_scores).item()

            best_cand = valid_candidates[best_idx]

            # 3. Reading
            # Reuse tensors for the best candidate
            best_c_tensor = c_tensor[best_idx].unsqueeze(0)
            best_q_tensor = q_tensor

            with torch.no_grad():
                start_logits, end_logits = reader(best_q_tensor, best_c_tensor)

            # Decode span
            # Simple decoding logic mirroring inference_utils
            s_logits = start_logits[0]
            e_logits = end_logits[0]

            # Find best valid span
            # Simplified version of inference_utils._get_best_span logic
            start_probs = torch.softmax(s_logits, dim=-1)
            end_probs = torch.softmax(e_logits, dim=-1)

            score_mat = torch.ger(start_probs, end_probs)
            mask = torch.triu(torch.ones_like(score_mat))
            score_mat = score_mat * mask

            max_score = score_mat.max()
            flat_idx = score_mat.argmax()
            best_s = (flat_idx // len(s_logits)).item()
            best_e = (flat_idx % len(s_logits)).item()
            confidence = max_score.item()

            # 4. Formulate Prediction
            pred_long = config.NULL_PREDICTION_STRING
            pred_short = config.NULL_PREDICTION_STRING

            if confidence >= config.CONFIDENCE_THRESHOLD:
                # Long
                pred_long = f"{best_cand['start_token']}:{best_cand['end_token']}"
                # Short
                sa_start = best_cand["start_token"] + best_s
                sa_end = best_cand["start_token"] + best_e + 1
                pred_short = f"{sa_start}:{sa_end}"

            # 5. Evaluate
            f1_long = compute_f1(pred_long, gt_long)
            f1_short = compute_f1(pred_short, gt_short)

            # Task metric is Micro F1 over all predictions (long and short treated as separate instances)
            # Here we average them for the sample, but for global micro F1 we should sum TP/FP/FN.
            # However, simplified metric calculation: average sample F1 is often used as approximation or macro.
            # The prompt asks for Micro F1.
            # Let's accumulate counts for Micro F1.

            # For this specific sample:
            # Cite debug_lesson_5: Decouple False Positive and False Negative Logic
            # Long:
            l_tp = 1 if pred_long in gt_long and pred_long != "" else 0
            l_fp = 1 if pred_long not in gt_long and pred_long != "" else 0
            l_fn = 1 if gt_long and l_tp == 0 else 0

            # Short:
            s_tp = 1 if pred_short in gt_short and pred_short != "" else 0
            s_fp = 1 if pred_short not in gt_short and pred_short != "" else 0
            s_fn = 1 if gt_short and s_tp == 0 else 0

            # We store tuple for global aggregation later
            f1_scores.append(
                {
                    "l_tp": l_tp,
                    "l_fp": l_fp,
                    "l_fn": l_fn,
                    "s_tp": s_tp,
                    "s_fp": s_fp,
                    "s_fn": s_fn,
                }
            )

            # Collect meta features for failure analysis
            # Error = 1 - Average F1 for this sample
            sample_f1_long = (
                1.0
                if (pred_long in gt_long) or (pred_long == "" and not gt_long)
                else 0.0
            )
            sample_f1_short = (
                1.0
                if (pred_short in gt_short) or (pred_short == "" and not gt_short)
                else 0.0
            )
            avg_f1 = (sample_f1_long + sample_f1_short) / 2.0

            meta_features["q_len"].append(len(q_text.split()))
            meta_features["doc_len"].append(len(doc_tokens))
            meta_features["error"].append(1.0 - avg_f1)

    # Compute Global Micro F1
    total_tp = sum(d["l_tp"] + d["s_tp"] for d in f1_scores)
    total_fp = sum(d["l_fp"] + d["s_fp"] for d in f1_scores)
    total_fn = sum(d["l_fn"] + d["s_fn"] for d in f1_scores)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print(f"Final Validation Metric: {micro_f1}")

    return meta_features


def perform_failure_analysis(meta_features):
    """
    Calculates correlation between error magnitude and input features.
    """
    print("\n--- Failure Analysis ---")
    df = pd.DataFrame(meta_features)
    if len(df) == 0:
        print("No data for failure analysis.")
        return

    # Correlation
    corr_q = df["error"].corr(df["q_len"])
    corr_doc = df["error"].corr(df["doc_len"])

    print("Correlation between Error and Input Features:")
    print(f"Question Length: {corr_q:.4f}")
    print(f"Document Length: {corr_doc:.4f}")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------


def main():
    print("Initializing Fast Baseline Run...")

    # 1. Data Preparation
    # Build Vocab
    vocab = text_utils.build_vocab(load_cached_data=True)
    vocab_size = len(vocab)

    # Load Embeddings
    embeddings = text_utils.load_embeddings(vocab, load_cached_data=True)

    # Process Data
    print("Processing Ranker Data...")
    ranker_train_df = data_factory.process_ranker_data(
        config.TRAIN_METADATA_PATH, vocab, load_cached_data=True, is_train=True
    )
    ranker_val_df = data_factory.process_ranker_data(
        config.VAL_METADATA_PATH, vocab, load_cached_data=True, is_train=False
    )

    print("Processing Reader Data...")
    reader_train_df = data_factory.process_reader_data(
        config.TRAIN_METADATA_PATH, vocab, load_cached_data=True, is_train=True
    )
    reader_val_df = data_factory.process_reader_data(
        config.VAL_METADATA_PATH, vocab, load_cached_data=True, is_train=False
    )

    # DataLoaders
    ranker_train_loader = DataLoader(
        data_factory.RankerDataset(ranker_train_df),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    ranker_val_loader = DataLoader(
        data_factory.RankerDataset(ranker_val_df),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    reader_train_loader = DataLoader(
        data_factory.ReaderDataset(reader_train_df),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    reader_val_loader = DataLoader(
        data_factory.ReaderDataset(reader_val_df),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    # 2. Model Initialization
    ranker = KMaxInteractionRanker(
        vocab_size=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        pretrained_embeddings=embeddings,
    ).to(DEVICE)

    reader = HighwayCoAttentionReader(
        vocab_size=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        pretrained_embeddings=embeddings,
    ).to(DEVICE)

    # 3. Training
    print("\n--- Training Ranker ---")
    training_utils.train_ranker(ranker, ranker_train_loader, ranker_val_loader, DEVICE)

    print("\n--- Training Reader ---")
    training_utils.train_reader(reader, reader_train_loader, reader_val_loader, DEVICE)

    # 4. Validation & Failure Analysis
    # We pass the trained models directly to avoid reloading overhead
    meta_features = run_validation_inference(ranker, reader, vocab, embeddings)
    perform_failure_analysis(meta_features)

    # 5. Submission
    print("\n--- Generating Submission ---")
    # We need to ensure the InferencePipeline uses the models we just trained.
    # The pipeline class loads from disk by default. We save the current state to ensure consistency.
    torch.save(ranker.state_dict(), config.RANKER_MODEL_PATH)
    torch.save(reader.state_dict(), config.READER_MODEL_PATH)

    pipeline = inference_utils.InferencePipeline()
    # Pre-load resources to avoid redundant loading
    pipeline.vocab = vocab
    pipeline.embeddings = embeddings
    pipeline.ranker = ranker
    pipeline.reader = reader
    pipeline.device = DEVICE

    pipeline.run_inference()

    print("Done.")


if __name__ == "__main__":
    main()
