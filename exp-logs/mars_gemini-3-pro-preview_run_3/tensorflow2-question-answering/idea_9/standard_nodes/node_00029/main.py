import os
import json
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# Import library modules
from library.config import Config
from library.preprocessing import TextPreprocessor
from library.data_loader import NQRankerDataset, NQReaderDataset, collate_fn
from library.train_engine import train_ranker, train_reader, set_seed
from library.eval_engine import EvalEngine
from library.ranker_model import DecomposableAttentionRanker
from library.reader_model import GatedConvReader


def setup_fast_baseline_config():
    """
    Overrides Config parameters for a fast baseline run.
    """
    print("Configuring fast baseline parameters...")
    Config.NUM_EPOCHS = 1
    Config.SAMPLE_SIZE = 10000  # Limit training samples for speed
    Config.BATCH_SIZE = 32  # Adjust for memory safety
    Config.MAX_DOC_LEN = 128  # Reduce sequence length for speed
    Config.MAX_Q_LEN = 20

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


def evaluate_on_validation(ranker, reader, preprocessor, device):
    """
    Runs end-to-end inference on the validation set to compute Micro F1
    and collect data for failure analysis.
    """
    print("\n--- Starting Validation Assessment ---")

    # Load validation metadata
    val_meta_df = pd.read_csv(Config.VAL_METADATA)
    # Sample validation set for speed if necessary, but requirements say "entire hold-out validation set"
    # However, to keep within 2 hours with training, we might need to be careful.
    # Given the constraints, we will process the full validation set defined in metadata
    # (which is already a split).

    ranker.eval()
    reader.eval()

    tp = 0
    fp = 0
    fn = 0

    analysis_data = []

    # We process sample by sample for the validation logic to handle the complex
    # logic of selecting candidates and matching spans.
    # For speed, we could batch, but the logic connecting Ranker output to Reader input
    # is complex. We will implement a simplified batched approach or sequential if fast enough.
    # Given the time limit, sequential processing of the validation set (approx 55k samples)
    # might be too slow if we run the full models.
    # We will use a subset for the "fast baseline" demonstration if the set is huge,
    # but the prompt asks for "entire hold-out validation set".
    # We will proceed with the full set but optimize where possible.

    # Actually, let's load the raw file and iterate efficiently.

    with open(Config.TRAIN_FILE, "rb") as f:
        for idx, row in val_meta_df.iterrows():
            offset = row["byte_offset"]
            f.seek(offset)
            line = f.readline()
            if not line:
                continue

            try:
                entry = json.loads(line.decode("utf-8"))

                # Ground Truth Extraction
                gt_long_answers = []
                gt_short_answers = []

                annotations = entry.get("annotations", [])
                for ann in annotations:
                    la = ann.get("long_answer", {})
                    if la.get("start_token", -1) != -1:
                        gt_long_answers.append((la["start_token"], la["end_token"]))

                    for sa in ann.get("short_answers", []):
                        gt_short_answers.append((sa["start_token"], sa["end_token"]))

                    # Yes/No answers are not handled by this specific architecture (span extraction),
                    # so we treat them as no short answer span for this baseline.

                # Prepare Input
                q_text = entry.get("question_text", "")
                doc_text = entry.get("document_text", "")
                doc_tokens = preprocessor.tokenize(doc_text)
                q_tokens = preprocessor.tokenize(q_text)

                candidates = entry.get("long_answer_candidates", [])
                top_level_candidates = [c for c in candidates if c["top_level"]]

                if not top_level_candidates:
                    # No candidates, model predicts nothing
                    pred_long = None
                    pred_short = None
                else:
                    # 1. Rank Candidates
                    # Prepare batch for ranker
                    q_indices = (
                        preprocessor.text_to_indices(q_text, Config.MAX_Q_LEN)
                        .unsqueeze(0)
                        .to(device)
                    )

                    cand_texts = []
                    cand_objs = []

                    # Optimization: Limit number of candidates processed per document to top N to save time
                    max_cands = 10

                    for cand in top_level_candidates[:max_cands]:
                        start = cand["start_token"]
                        end = cand["end_token"]
                        c_text = " ".join(doc_tokens[start:end])
                        cand_texts.append(c_text)
                        cand_objs.append(cand)

                    if not cand_texts:
                        pred_long = None
                        pred_short = None
                    else:
                        # Batchify candidate docs
                        doc_indices_list = [
                            preprocessor.text_to_indices(t, Config.MAX_DOC_LEN)
                            for t in cand_texts
                        ]
                        doc_tensor = torch.stack(doc_indices_list).to(device)

                        # Expand Q to match candidates
                        q_tensor = q_indices.repeat(len(cand_texts), 1)

                        with torch.no_grad():
                            logits = ranker(q_tensor, doc_tensor)
                            scores = torch.sigmoid(logits).cpu().numpy().flatten()

                        best_idx = np.argmax(scores)
                        best_score = scores[best_idx]
                        best_cand_obj = cand_objs[best_idx]
                        best_cand_text = cand_texts[best_idx]

                        # 2. Read Answer
                        # Input: Q + Paragraph
                        input_text = " ".join(
                            q_tokens + preprocessor.tokenize(best_cand_text)
                        )
                        input_indices = (
                            preprocessor.text_to_indices(
                                input_text, Config.MAX_Q_LEN + Config.MAX_DOC_LEN
                            )
                            .unsqueeze(0)
                            .to(device)
                        )

                        with torch.no_grad():
                            start_logits, end_logits = reader(input_indices)
                            start_probs = (
                                torch.softmax(start_logits, dim=1).cpu().numpy()[0]
                            )
                            end_probs = (
                                torch.softmax(end_logits, dim=1).cpu().numpy()[0]
                            )

                        # Find best span
                        best_span_score = -1.0
                        best_rel_start = -1
                        best_rel_end = -1

                        q_len = len(q_tokens)
                        seq_len = len(input_indices[0])

                        # Constrained search
                        for s in range(q_len, seq_len):
                            for e in range(s, min(s + Config.MAX_ANSWER_LEN, seq_len)):
                                score = start_probs[s] * end_probs[e]
                                if score > best_span_score:
                                    best_span_score = score
                                    best_rel_start = s
                                    best_rel_end = e

                        joint_conf = best_score * best_span_score

                        # Thresholding
                        if joint_conf >= Config.CONFIDENCE_THRESHOLD:
                            # Long Answer Prediction
                            pred_long = (
                                best_cand_obj["start_token"],
                                best_cand_obj["end_token"],
                            )

                            # Short Answer Prediction
                            # Map relative to global
                            local_start = best_rel_start - q_len
                            local_end = best_rel_end - q_len  # inclusive local end

                            global_start = best_cand_obj["start_token"] + local_start
                            global_end = (
                                best_cand_obj["start_token"] + local_end + 1
                            )  # exclusive global end

                            pred_short = (global_start, global_end)
                        else:
                            pred_long = None
                            pred_short = None

                # Compute Metrics
                # Long Answer Logic
                has_gt_long = len(gt_long_answers) > 0
                has_pred_long = pred_long is not None

                match_long = False
                if has_pred_long:
                    for gt in gt_long_answers:
                        if pred_long == gt:
                            match_long = True
                            break

                if match_long:
                    tp += 1
                elif has_pred_long and not match_long:
                    fp += 1
                elif not has_pred_long and has_gt_long:
                    fn += 1
                # TN (both none) is ignored in F1

                # Short Answer Logic
                has_gt_short = len(gt_short_answers) > 0
                has_pred_short = pred_short is not None

                match_short = False
                if has_pred_short:
                    for gt in gt_short_answers:
                        if pred_short == gt:
                            match_short = True
                            break

                if match_short:
                    tp += 1
                elif has_pred_short and not match_short:
                    fp += 1
                elif not has_pred_short and has_gt_short:
                    fn += 1

                # Failure Analysis Data
                is_error = 0
                if (
                    (has_gt_long and not match_long)
                    or (has_gt_short and not match_short)
                    or (has_pred_long and not match_long)
                ):
                    is_error = 1

                analysis_data.append(
                    {
                        "q_len": len(q_tokens),
                        "doc_len": len(doc_tokens),
                        "error": is_error,
                    }
                )

            except Exception as e:
                continue

    # Compute Micro F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print(f"Final Validation Metric: {f1}")

    return pd.DataFrame(analysis_data)


def perform_failure_analysis(df):
    """
    Calculates correlation between error magnitude and input features.
    """
    print("\n--- Failure Analysis ---")
    if df.empty:
        print("No data for failure analysis.")
        return

    # Correlation with Question Length
    corr_q, _ = pearsonr(df["q_len"], df["error"])
    print(f"Correlation between Error and Question Length: {corr_q}")

    # Correlation with Document Length
    corr_d, _ = pearsonr(df["doc_len"], df["error"])
    print(f"Correlation between Error and Document Length: {corr_d}")


def main():
    # 1. Setup
    setup_fast_baseline_config()
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Preprocessing
    preprocessor = TextPreprocessor()
    vocab = preprocessor.build_vocabulary(load_cached_data=True)
    embeddings = preprocessor.load_embeddings(load_cached_data=True)

    # 3. Data Loading
    # Ranker Datasets
    ranker_train_ds = NQRankerDataset(
        Config.TRAIN_METADATA,
        Config.TRAIN_FILE,
        preprocessor,
        is_train=True,
        load_cached_data=True,
        sample_size=Config.SAMPLE_SIZE,
    )
    ranker_val_ds = NQRankerDataset(
        Config.VAL_METADATA,
        Config.TRAIN_FILE,
        preprocessor,
        is_train=False,
        load_cached_data=True,
        sample_size=Config.SAMPLE_SIZE // 5,
    )

    ranker_train_loader = DataLoader(
        ranker_train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    ranker_val_loader = DataLoader(
        ranker_val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Reader Datasets
    reader_train_ds = NQReaderDataset(
        Config.TRAIN_METADATA,
        Config.TRAIN_FILE,
        preprocessor,
        is_train=True,
        load_cached_data=True,
        sample_size=Config.SAMPLE_SIZE,
    )
    reader_val_ds = NQReaderDataset(
        Config.VAL_METADATA,
        Config.TRAIN_FILE,
        preprocessor,
        is_train=False,
        load_cached_data=True,
        sample_size=Config.SAMPLE_SIZE // 5,
    )

    reader_train_loader = DataLoader(
        reader_train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    reader_val_loader = DataLoader(
        reader_val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # 4. Training
    print("\n--- Training Ranker ---")
    ranker_model = train_ranker(
        ranker_train_loader, ranker_val_loader, embeddings, device
    )

    print("\n--- Training Reader ---")
    reader_model = train_reader(
        reader_train_loader, reader_val_loader, embeddings, device
    )

    # 5. Validation Assessment & Failure Analysis
    # We use the trained models to evaluate on the validation set
    analysis_df = evaluate_on_validation(
        ranker_model, reader_model, preprocessor, device
    )
    perform_failure_analysis(analysis_df)

    # 6. Submission Generation
    print("\n--- Generating Submission ---")
    eval_engine = EvalEngine()
    # Ensure the eval engine uses the models we just trained (reloading weights saved during training)
    # The EvalEngine constructor loads from disk, and train functions save to disk.
    # Cite debug_lesson_8: Disable Caching for Inference to Prevent Stale Artifacts
    eval_engine.predict_sample(load_cached_data=False)

    print("Runfile execution completed.")


if __name__ == "__main__":
    main()
