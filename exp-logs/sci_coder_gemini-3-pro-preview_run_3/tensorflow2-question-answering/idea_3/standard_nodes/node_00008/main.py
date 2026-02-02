import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import library modules
from library.config import Config
from library.trainer import run_training
from library.inference import generate_predictions
from library.models import SiameseRanker, ConditionalReader
from library.data_processing import DataProcessor, TextProcessor

# --------------------------------------------------------------------------
# 1. Configuration Overrides for Fast Baseline
# --------------------------------------------------------------------------
# Override config to ensure execution finishes quickly
Config.EPOCHS = 1
Config.TRAIN_SAMPLE_SIZE = 10000  # Reduced from 50k
Config.VAL_SAMPLE_SIZE = 2000  # Reduced for fast validation
Config.VOCAB_SIZE = 10000  # Smaller vocab for speed
Config.BATCH_SIZE = 32

# Set seeds
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


def main():
    # --------------------------------------------------------------------------
    # 2. Training
    # --------------------------------------------------------------------------
    print("Step 1: Training Models...")
    # load_cached_data=False forces regeneration of data with new sample sizes
    run_training(load_cached_data=False)

    # --------------------------------------------------------------------------
    # 3. Validation & Metric Calculation
    # --------------------------------------------------------------------------
    print("Step 2: Validation and Metric Calculation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Metadata
    if not os.path.exists(Config.VAL_METADATA):
        print("Validation metadata not found. Skipping validation.")
        return

    val_meta = pd.read_csv(Config.VAL_METADATA).head(Config.VAL_SAMPLE_SIZE)

    # Initialize Processor and Vocab
    processor = DataProcessor()
    if os.path.exists(Config.VOCAB_CACHE):
        processor.vocab.load(Config.VOCAB_CACHE)
    text_processor = processor.text_processor
    vocab = processor.vocab

    # Load Models
    ranker = SiameseRanker(vocab_size=len(vocab)).to(device)
    if os.path.exists(Config.RANKER_MODEL_PATH):
        ranker.load_state_dict(
            torch.load(Config.RANKER_MODEL_PATH, map_location=device)
        )
    ranker.eval()

    reader = ConditionalReader(vocab_size=len(vocab)).to(device)
    if os.path.exists(Config.READER_MODEL_PATH):
        reader.load_state_dict(
            torch.load(Config.READER_MODEL_PATH, map_location=device)
        )
    reader.eval()

    # Metrics counters
    tp_long, fp_long, fn_long = 0, 0, 0
    tp_short, fp_short, fn_short = 0, 0, 0

    # Analysis data
    analysis_data = []

    # Helper for inference
    def prepare_tensor(tokens, max_len):
        indices = vocab.encode(tokens)
        indices = indices[:max_len]
        pad_len = max_len - len(indices)
        indices += [vocab.token_to_idx[Config.PAD_TOKEN]] * pad_len
        return torch.tensor([indices], dtype=torch.long).to(device)

    # Validation Loop
    # We read from TRAIN_FILE because validation split comes from there
    with open(Config.TRAIN_FILE, "rb") as f:
        for _, row in tqdm(val_meta.iterrows(), total=len(val_meta), disable=True):
            offset = row["byte_offset"]
            f.seek(offset)
            line = f.readline()
            if not line:
                continue

            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue

            # --- Ground Truth Extraction ---
            annotations = data.get("annotations", [])
            gt_long_spans = []
            gt_short_spans = []

            for ann in annotations:
                la = ann["long_answer"]
                if la["start_token"] != -1:
                    gt_long_spans.append((la["start_token"], la["end_token"]))

                for sa in ann["short_answers"]:
                    gt_short_spans.append((sa["start_token"], sa["end_token"]))
                # Note: Model architecture provided does not handle YES/NO, so we focus on spans

            # --- Inference ---
            q_text = data["question_text"]
            q_tokens = text_processor.tokenize(q_text)
            q_tensor = prepare_tensor(q_tokens, Config.MAX_Q_LEN)

            doc_text = data["document_text"]
            doc_tokens = doc_text.split()
            candidates = data["long_answer_candidates"]

            pred_long_span = None
            pred_short_span = None

            if candidates:
                # Batch candidates
                cand_tensors = []
                cand_clean_tokens_list = []
                cand_maps = []
                valid_cand_indices = []

                for idx, cand in enumerate(candidates):
                    raw_span = doc_tokens[cand["start_token"] : cand["end_token"]]
                    clean_tokens, idx_map = text_processor.clean_and_map_indices(
                        raw_span
                    )
                    if not clean_tokens:
                        continue

                    cand_clean_tokens_list.append(clean_tokens)
                    cand_maps.append(idx_map)
                    valid_cand_indices.append(idx)

                    indices = vocab.encode(clean_tokens)
                    indices = indices[: Config.MAX_CTX_LEN]
                    pad_len = Config.MAX_CTX_LEN - len(indices)
                    indices += [vocab.token_to_idx[Config.PAD_TOKEN]] * pad_len
                    cand_tensors.append(torch.tensor(indices, dtype=torch.long))

                if cand_tensors:
                    ctx_batch = torch.stack(cand_tensors).to(device)
                    q_batch = q_tensor.repeat(len(ctx_batch), 1)

                    with torch.no_grad():
                        scores = ranker(q_batch, ctx_batch)

                    best_score, best_idx_in_batch = torch.max(scores, dim=0)

                    if best_score.item() >= Config.RANKER_THRESHOLD:
                        # Long Answer Prediction
                        best_cand_orig_idx = valid_cand_indices[best_idx_in_batch]
                        best_cand = candidates[best_cand_orig_idx]
                        pred_long_span = (
                            best_cand["start_token"],
                            best_cand["end_token"],
                        )

                        # Short Answer Prediction
                        best_clean_tokens = cand_clean_tokens_list[best_idx_in_batch]
                        best_map = cand_maps[best_idx_in_batch]

                        q_reader = q_tensor
                        ctx_reader = ctx_batch[best_idx_in_batch].unsqueeze(0)

                        with torch.no_grad():
                            s_logits, e_logits = reader(q_reader, ctx_reader)

                        actual_len = min(len(best_clean_tokens), Config.MAX_CTX_LEN)
                        s_probs = F.softmax(s_logits[0, :actual_len], dim=0)
                        e_probs = F.softmax(e_logits[0, :actual_len], dim=0)

                        # Greedy search
                        best_s, best_e, _ = (-1, -1, -1.0)
                        curr_max = -1.0
                        s_list = s_probs.tolist()
                        e_list = e_probs.tolist()

                        for i in range(len(s_list)):
                            for j in range(i, min(len(s_list), i + 30)):
                                sc = s_list[i] * e_list[j]
                                if sc > curr_max:
                                    curr_max = sc
                                    best_s, best_e = i, j

                        # Map back to global
                        raw_span_tokens = doc_tokens[
                            best_cand["start_token"] : best_cand["end_token"]
                        ]
                        raw_s_rel = -1
                        raw_e_rel = -1

                        # Reverse map logic
                        for r_idx, token in enumerate(raw_span_tokens):
                            if best_map.get(
                                r_idx
                            ) == best_s and not text_processor.is_html_tag(token):
                                raw_s_rel = r_idx
                                break
                        for r_idx, token in enumerate(raw_span_tokens):
                            if best_map.get(
                                r_idx
                            ) == best_e and not text_processor.is_html_tag(token):
                                raw_e_rel = r_idx
                                break

                        if raw_s_rel != -1 and raw_e_rel != -1:
                            pred_short_span = (
                                best_cand["start_token"] + raw_s_rel,
                                best_cand["start_token"]
                                + raw_e_rel
                                + 1,  # Exclusive end
                            )

            # --- Evaluation ---
            # Long Answer
            is_correct_long = False
            if pred_long_span:
                if pred_long_span in gt_long_spans:
                    tp_long += 1
                    is_correct_long = True
                else:
                    fp_long += 1
            else:
                if gt_long_spans:
                    fn_long += 1
                # else TN (ignored in F1)

            # Short Answer
            is_correct_short = False
            if pred_short_span:
                if pred_short_span in gt_short_spans:
                    tp_short += 1
                    is_correct_short = True
                else:
                    fp_short += 1
            else:
                if gt_short_spans:
                    fn_short += 1

            # --- Analysis Data ---
            error_val = 0
            if (gt_long_spans and not is_correct_long) or (
                not gt_long_spans and pred_long_span
            ):
                error_val += 1
            if (gt_short_spans and not is_correct_short) or (
                not gt_short_spans and pred_short_span
            ):
                error_val += 1

            analysis_data.append(
                {"doc_len": len(doc_tokens), "q_len": len(q_tokens), "error": error_val}
            )

    # Calculate Micro F1
    total_tp = tp_long + tp_short
    total_fp = fp_long + fp_short
    total_fn = fn_long + fn_short

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print(f"Final Validation Metric: {f1}")

    # --------------------------------------------------------------------------
    # 4. Failure Analysis
    # --------------------------------------------------------------------------
    print("Step 3: Failure Analysis...")
    df_analysis = pd.DataFrame(analysis_data)
    if not df_analysis.empty and df_analysis["error"].std() > 0:
        corr_doc = df_analysis["doc_len"].corr(df_analysis["error"])
        corr_q = df_analysis["q_len"].corr(df_analysis["error"])
        print(f"Correlation Error vs Doc Length: {corr_doc}")
        print(f"Correlation Error vs Question Length: {corr_q}")
    else:
        print("Insufficient variance for correlation analysis.")

    # --------------------------------------------------------------------------
    # 5. Submission
    # --------------------------------------------------------------------------
    print("Step 4: Generating Submission...")
    generate_predictions(sample_size=None)


if __name__ == "__main__":
    main()
