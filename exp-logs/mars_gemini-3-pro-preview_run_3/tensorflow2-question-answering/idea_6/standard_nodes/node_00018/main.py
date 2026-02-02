import sys
import os
import torch
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from transformers import AutoTokenizer

# Ensure library is in path
sys.path.append(".")

# Import library modules
from library.config import PathConfig, ModelConfig, TrainingConfig
from library.utils import set_seed, setup_logger, parse_html, HTML_TAGS
from library.train import run_training
from library.evaluate import predict_submission, load_models, get_original_token_indices

# --- Configuration Overrides for Fast Baseline ---
# Limit training data to ensure completion within 2 hours
TrainingConfig.SUBSET_SIZE = 2000
TrainingConfig.EPOCHS = 1
TrainingConfig.BATCH_SIZE = 16
TrainingConfig.NUM_WORKERS = 2

# Logger
logger = setup_logger("runfile")


def get_ground_truths(annotation):
    """Parses annotation to get sets of valid long and short answer spans."""
    # Long Answer
    la = annotation["long_answer"]
    la_truths = []
    if la["start_token"] != -1:
        la_truths.append((la["start_token"], la["end_token"]))

    # Short Answers
    sa_truths = []
    for sa in annotation["short_answers"]:
        sa_truths.append((sa["start_token"], sa["end_token"]))

    # Yes/No Answer (treated as a special string if present)
    # Note: The provided Reader model predicts spans, so it likely won't predict YES/NO.
    # We include it in GT for correctness of metric calculation.
    if annotation["yes_no_answer"] != "NONE":
        sa_truths.append(annotation["yes_no_answer"])

    return la_truths, sa_truths


def validate_and_analyze():
    """
    Performs inference on the validation set, computes Micro F1,
    and analyzes correlations between errors and input features.
    """
    logger.info("Starting Validation and Failure Analysis...")

    device = torch.device(TrainingConfig.DEVICE)
    ranker, reader = load_models(device)
    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.MODEL_NAME)

    # Load Validation Metadata
    if not os.path.exists(PathConfig.VAL_METADATA):
        logger.error("Validation metadata not found.")
        return

    val_df = pd.read_csv(PathConfig.VAL_METADATA)

    # Metrics counters
    tp = 0
    fp = 0
    fn = 0

    # Analysis data
    analysis_records = []

    # Inference Loop
    # We process one by one for simplicity given the complex candidate logic,
    # but use no_grad and GPU for speed.

    ranker.eval()
    reader.eval()

    with torch.no_grad():
        for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Validation"):
            offset = row["byte_offset"]

            # Read Raw Data
            with open(PathConfig.TRAIN_FILE, "rb") as f:
                f.seek(offset)
                line = f.readline()
                if not line:
                    continue
                record = json.loads(line.decode("utf-8"))

            question = record["question_text"]
            doc_text = record["document_text"]
            doc_tokens = doc_text.split()
            annotations = record["annotations"]  # List of annotations

            # Collect all valid ground truths from all annotators
            all_la_truths = set()
            all_sa_truths = set()

            for ann in annotations:
                la_t, sa_t = get_ground_truths(ann)
                for item in la_t:
                    all_la_truths.add(item)
                for item in sa_t:
                    all_sa_truths.add(item)

            # --- Prediction Pipeline ---

            # 1. Candidate Generation
            candidates = parse_html(doc_text)

            pred_la = None
            pred_sa = None

            if candidates:
                cand_texts = [c["text"] for c in candidates]

                # Ranker
                q_inputs = tokenizer(
                    [question],
                    max_length=ModelConfig.MAX_Q_LEN,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(device)

                # Batch candidates
                cand_embeddings = []
                chunk_size = 32
                for i in range(0, len(cand_texts), chunk_size):
                    batch_texts = cand_texts[i : i + chunk_size]
                    c_inputs = tokenizer(
                        batch_texts,
                        max_length=ModelConfig.MAX_CTX_LEN,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)
                    c_emb = ranker(c_inputs["input_ids"], c_inputs["attention_mask"])
                    cand_embeddings.append(c_emb)

                if cand_embeddings:
                    cand_embeddings = torch.cat(cand_embeddings, dim=0)
                    q_emb = ranker(q_inputs["input_ids"], q_inputs["attention_mask"])

                    q_emb_norm = torch.nn.functional.normalize(q_emb, p=2, dim=1)
                    c_emb_norm = torch.nn.functional.normalize(
                        cand_embeddings, p=2, dim=1
                    )
                    scores = torch.mm(q_emb_norm, c_emb_norm.transpose(0, 1)).squeeze(0)

                    best_score, best_idx = torch.max(scores, dim=0)
                    best_score = best_score.item()
                    best_idx = best_idx.item()
                    best_candidate = candidates[best_idx]

                    # Threshold check
                    if best_score >= ModelConfig.RANKER_THRESHOLD:
                        pred_la = (
                            best_candidate["start_token"],
                            best_candidate["end_token"],
                        )

                        # Reader
                        context_text = best_candidate["text"]
                        reader_inputs = tokenizer(
                            question,
                            context_text,
                            truncation="only_second",
                            max_length=ModelConfig.MAX_CTX_LEN,
                            return_offsets_mapping=True,
                            return_token_type_ids=True,
                            padding="max_length",
                            return_tensors="pt",
                        )
                        r_input_ids = reader_inputs["input_ids"].to(device)
                        r_mask = reader_inputs["attention_mask"].to(device)
                        r_token_type = reader_inputs["token_type_ids"].to(device)
                        offset_mapping = (
                            reader_inputs["offset_mapping"][0].cpu().numpy()
                        )

                        start_logits, end_logits = reader(
                            r_input_ids, r_mask, r_token_type
                        )

                        start_probs = (
                            torch.softmax(start_logits, dim=1).cpu().numpy()[0]
                        )
                        end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()[0]
                        start_idx = np.argmax(start_probs)
                        end_idx = np.argmax(end_probs)
                        confidence = start_probs[start_idx] * end_probs[end_idx]

                        if (
                            start_idx <= end_idx
                            and confidence >= ModelConfig.SHORT_ANSWER_THRESHOLD
                        ):
                            token_types = r_token_type.cpu().numpy()[0]
                            if (
                                token_types[start_idx] == 1
                                and token_types[end_idx] == 1
                            ):
                                char_start = offset_mapping[start_idx][0]
                                char_end = offset_mapping[end_idx][1]
                                clean_tokens = context_text.split()

                                # Map chars to clean tokens
                                current_char = 0
                                clean_token_start_idx = -1
                                clean_token_end_idx = -1
                                for i, token in enumerate(clean_tokens):
                                    token_len = len(token)
                                    if (
                                        current_char
                                        <= char_start
                                        < current_char + token_len + 1
                                    ):
                                        clean_token_start_idx = i
                                    if (
                                        current_char
                                        <= char_end
                                        <= current_char + token_len + 1
                                    ):
                                        clean_token_end_idx = i
                                    if current_char + token_len == char_end:
                                        clean_token_end_idx = i
                                    current_char += token_len + 1

                                if clean_token_start_idx != -1:
                                    if clean_token_end_idx == -1:
                                        clean_token_end_idx = len(clean_tokens) - 1

                                    # Map to original
                                    original_indices = get_original_token_indices(
                                        doc_tokens,
                                        best_candidate["start_token"],
                                        best_candidate["end_token"],
                                    )
                                    if clean_token_start_idx < len(
                                        original_indices
                                    ) and clean_token_end_idx < len(original_indices):
                                        final_start = original_indices[
                                            clean_token_start_idx
                                        ]
                                        final_end = (
                                            original_indices[clean_token_end_idx] + 1
                                        )
                                        pred_sa = (final_start, final_end)

            # --- Metric Calculation ---

            # Long Answer
            la_correct = False
            if pred_la is not None:
                if pred_la in all_la_truths:
                    tp += 1
                    la_correct = True
                else:
                    fp += 1

            if not la_correct and len(all_la_truths) > 0:
                fn += 1

            # Short Answer
            sa_correct = False
            if pred_sa is not None:
                if pred_sa in all_sa_truths:
                    tp += 1
                    sa_correct = True
                else:
                    fp += 1

            if not sa_correct and len(all_sa_truths) > 0:
                fn += 1

            # --- Failure Analysis Data ---
            # Define error as failure to match existing ground truth
            # If GT exists and we missed it, or we predicted when no GT -> Error
            is_error = 0
            if (len(all_la_truths) > 0 and not la_correct) or (
                pred_la is not None and not la_correct
            ):
                is_error = 1
            if (len(all_sa_truths) > 0 and not sa_correct) or (
                pred_sa is not None and not sa_correct
            ):
                is_error = 1

            analysis_records.append(
                {
                    "error": is_error,
                    "doc_len": len(doc_tokens),
                    "q_len": len(question.split()),
                }
            )

    # Compute Micro F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    print(f"Final Validation Metric: {f1}")

    # Compute Correlations
    if analysis_records:
        df_an = pd.DataFrame(analysis_records)
        corr_doc = df_an["error"].corr(df_an["doc_len"])
        corr_q = df_an["error"].corr(df_an["q_len"])
        print(f"Correlation Error vs Doc Length: {corr_doc}")
        print(f"Correlation Error vs Question Length: {corr_q}")


if __name__ == "__main__":
    # 1. Train models
    run_training()

    # 2. Validate and Analyze
    validate_and_analyze()

    # 3. Generate Submission
    predict_submission()
