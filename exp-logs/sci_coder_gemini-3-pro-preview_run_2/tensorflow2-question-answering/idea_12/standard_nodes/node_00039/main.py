import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import scipy.stats

from library.config import Config
from library.utils import set_seed, load_glove_embeddings
from library.data_processing import build_vocab
from library.dataset import get_dataloaders, preprocess_and_cache, YESNO_MAP
from library.model import FiLMNetwork
from library.trainer import Trainer
from library.inference import Predictor


def get_best_span(start_logits, end_logits):
    """
    Finds the best valid span (start <= end) maximizing the sum of logits.
    Copied logic from Predictor for use in validation loop.
    """
    top_k = 5
    start_probs = F.softmax(start_logits, dim=0)
    end_probs = F.softmax(end_logits, dim=0)

    top_start_indices = torch.topk(
        start_probs, k=min(top_k, len(start_probs))
    ).indices.tolist()

    best_score = -float("inf")
    best_span = (0, 0)

    for s_idx in top_start_indices:
        max_span_len = 30
        e_search_end = min(len(end_probs), s_idx + max_span_len)

        if s_idx >= e_search_end:
            continue

        valid_end_logits = end_logits[s_idx:e_search_end]
        best_rel_e = torch.argmax(valid_end_logits).item()
        e_idx = s_idx + best_rel_e

        score = start_logits[s_idx].item() + end_logits[e_idx].item()

        if score > best_score:
            best_score = score
            best_span = (s_idx, e_idx)

    return best_span


def validate_and_analyze(model, vocab, val_loader, device):
    print("Loading validation metadata for ground truth...")
    # Load validation dataframe to get full ground truth labels
    val_df = preprocess_and_cache(
        Config.VAL_META,
        Config.TRAIN_FILE,
        vocab,
        Config.VAL_CACHE,
        is_train=True,
        filter_negatives=False,
    )
    # Create a lookup map for fast access
    val_labels_map = val_df.set_index("example_id").to_dict("index")

    model.eval()

    # Stats for Micro F1
    # We track TP, FP, FN for Long and Short answers separately
    # Logic:
    #   TP: GT is not empty AND Pred matches GT
    #   FP: Pred is not empty AND (GT is empty OR Pred != GT)
    #   FN: GT is not empty AND (Pred is empty OR Pred != GT)
    # Note: If Pred != GT, it counts as both FP (wrong prediction) and FN (missed truth)

    stats = {"long": {"tp": 0, "fp": 0, "fn": 0}, "short": {"tp": 0, "fp": 0, "fn": 0}}

    # Data for failure analysis
    analysis_data = []  # list of dicts

    idx_to_yesno = {v: k for k, v in YESNO_MAP.items()}

    print(f"Running validation on {len(val_loader)} examples...")

    with torch.no_grad():
        for batch in val_loader:
            example_id = batch["example_id"][0]
            q_input = batch["q_input"].to(device)
            candidates = batch["candidates"].to(device)

            # Ground Truth
            gt_row = val_labels_map.get(example_id)
            if not gt_row:
                continue

            gt_la_idx = int(gt_row["label_la_idx"])
            gt_short_span = gt_row["label_short_span"]  # [start, end] absolute
            gt_yn_val = int(gt_row["label_yn"])
            gt_yn_str = idx_to_yesno.get(gt_yn_val, "NONE")

            # Get candidates from dataframe to have access to all of them (offsets)
            # The loader might truncate to 30, but we need raw offsets for answer reconstruction
            raw_candidates = gt_row["candidates"]  # list of [start, end]

            # Model Inference
            candidates = candidates.squeeze(0)  # Remove batch dim -> (N, Ctx_Len)
            num_cands = candidates.size(0)

            pred_la_idx = -1
            pred_short_str = (
                ""  # formatted string for comparison if needed, or just logic
            )

            # Predictions
            has_pred_long = False
            has_pred_short = False

            # Predicted values
            p_la_idx = -1
            p_short_abs_start = -1
            p_short_abs_end = -1
            p_yn_str = "NONE"

            if num_cands > 0:
                q_input_expanded = q_input.repeat(num_cands, 1)
                outputs = model(q_input_expanded, candidates)

                rank_logits = outputs["rank_logits"].squeeze(1)
                rank_probs = torch.sigmoid(rank_logits)

                best_score, best_idx_tensor = torch.max(rank_probs, dim=0)
                best_idx = best_idx_tensor.item()

                if best_score.item() >= Config.LONG_ANSWER_THRESHOLD:
                    # Long Answer Predicted
                    has_pred_long = True
                    p_la_idx = best_idx

                    # Check Yes/No
                    yesno_logits = outputs["yesno_logits"][best_idx]
                    yesno_probs = F.softmax(yesno_logits, dim=0)
                    yesno_class = torch.argmax(yesno_probs).item()
                    p_yn_str = idx_to_yesno.get(yesno_class, "NONE")

                    if p_yn_str in ["YES", "NO"]:
                        has_pred_short = True
                    else:
                        # Span Prediction
                        start_logits = outputs["start_logits"][best_idx]
                        end_logits = outputs["end_logits"][best_idx]
                        rel_start, rel_end = get_best_span(start_logits, end_logits)

                        # Convert to absolute
                        # raw_candidates is list of [start, end]
                        # If best_idx is within range of raw_candidates (it should be unless truncated)
                        # Note: val_loader truncates to 30. If model picks index 29, it maps to raw_candidates[29]
                        if best_idx < len(raw_candidates):
                            cand_start, cand_end = raw_candidates[best_idx]
                            p_short_abs_start = cand_start + rel_start
                            p_short_abs_end = (
                                cand_start + rel_end + 1
                            )  # Inclusive to Exclusive for comparison?
                            # NQ short answer is token indices.
                            # Let's align with GT format. GT `label_short_span` is [start, end].
                            # In `preprocess_and_cache`, short_span is taken directly from JSON `start_token`, `end_token`.
                            # In NQ JSON, `end_token` is exclusive (Python slice style).
                            # Our `rel_end` from `get_best_span` is inclusive index.
                            # So `abs_end` should be `cand_start + rel_end + 1`.
                            has_pred_short = True

            # --- Evaluate Long Answer ---
            # GT exists?
            gt_has_long = gt_la_idx != -1

            if gt_has_long:
                if has_pred_long and p_la_idx == gt_la_idx:
                    stats["long"]["tp"] += 1
                elif has_pred_long and p_la_idx != gt_la_idx:
                    stats["long"]["fp"] += 1
                    stats["long"]["fn"] += 1
                else:  # not has_pred_long
                    stats["long"]["fn"] += 1
            else:
                if has_pred_long:
                    stats["long"]["fp"] += 1
                # else TN

            # --- Evaluate Short Answer ---
            # GT exists?
            gt_has_short = False
            if gt_yn_str in ["YES", "NO"]:
                gt_has_short = True
            elif gt_short_span[0] != -1:
                gt_has_short = True

            is_short_correct = False

            if gt_has_short:
                if has_pred_short:
                    # Check match
                    match = False
                    # Case 1: Yes/No match
                    if gt_yn_str in ["YES", "NO"]:
                        if p_yn_str == gt_yn_str:
                            match = True
                    # Case 2: Span match
                    elif gt_short_span[0] != -1 and p_yn_str not in ["YES", "NO"]:
                        # Compare spans
                        # GT is [start, end] (exclusive end)
                        # Pred is [p_short_abs_start, p_short_abs_end] (exclusive end)
                        if (
                            p_short_abs_start == gt_short_span[0]
                            and p_short_abs_end == gt_short_span[1]
                        ):
                            match = True

                    if match:
                        stats["short"]["tp"] += 1
                        is_short_correct = True
                    else:
                        stats["short"]["fp"] += 1
                        stats["short"]["fn"] += 1
                else:
                    stats["short"]["fn"] += 1
            else:
                if has_pred_short:
                    stats["short"]["fp"] += 1
                else:
                    is_short_correct = True  # TN is correct behavior

            # --- Failure Analysis Data ---
            # Error = 1 if either Long or Short was incorrect (based on F1 logic, if we missed TP or got FP)
            # For correlation, let's define error as "Did not get perfect score".
            # Perfect score means:
            #   If GT exists: Pred matches (TP)
            #   If GT doesn't exist: Pred doesn't exist (TN)

            long_ok = False
            if gt_has_long:
                if has_pred_long and p_la_idx == gt_la_idx:
                    long_ok = True
            else:
                if not has_pred_long:
                    long_ok = True

            short_ok = False
            if gt_has_short:
                if has_pred_short:
                    # Re-check match logic
                    if gt_yn_str in ["YES", "NO"]:
                        if p_yn_str == gt_yn_str:
                            short_ok = True
                    elif gt_short_span[0] != -1 and p_yn_str not in ["YES", "NO"]:
                        if (
                            p_short_abs_start == gt_short_span[0]
                            and p_short_abs_end == gt_short_span[1]
                        ):
                            short_ok = True
            else:
                if not has_pred_short:
                    short_ok = True

            error_val = 0.0
            if not (long_ok and short_ok):
                error_val = 1.0

            analysis_data.append(
                {
                    "q_len": len(gt_row["q_indices"]),
                    "num_cands": len(raw_candidates),
                    "error": error_val,
                }
            )

    # Compute Micro F1
    # F1 = 2 * P * R / (P + R) = 2 * TP / (2 * TP + FP + FN)

    # Aggregate counts for Micro F1
    total_tp = stats["long"]["tp"] + stats["short"]["tp"]
    total_fp = stats["long"]["fp"] + stats["short"]["fp"]
    total_fn = stats["long"]["fn"] + stats["short"]["fn"]

    denom = 2 * total_tp + total_fp + total_fn

    if denom == 0:
        final_metric = 0.0
    else:
        final_metric = 2 * total_tp / denom

    print(f"Final Validation Metric (Micro F1): {final_metric}")

    # Correlation Analysis
    df_analysis = pd.DataFrame(analysis_data)
    if not df_analysis.empty and df_analysis["error"].std() > 0:
        corr_q, _ = scipy.stats.pearsonr(df_analysis["q_len"], df_analysis["error"])
        corr_c, _ = scipy.stats.pearsonr(df_analysis["num_cands"], df_analysis["error"])
        print("Failure Analysis Correlations (Error vs Features):")
        print(f"  Question Length: {corr_q:.4f}")
        print(f"  Num Candidates:  {corr_c:.4f}")
    else:
        print("Failure Analysis: Insufficient variance to calculate correlations.")


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Explicitly invalidate caches to prevent loading stale/corrupt data (Cite debug_lesson_2)
    if os.path.exists(Config.TRAIN_CACHE):
        print(f"Removing stale cache: {Config.TRAIN_CACHE}")
        os.remove(Config.TRAIN_CACHE)
    if os.path.exists(Config.VAL_CACHE):
        print(f"Removing stale cache: {Config.VAL_CACHE}")
        os.remove(Config.VAL_CACHE)

    # Override for fast baseline execution
    Config.NUM_EPOCHS = 1

    # 2. Data Loading
    print("Building vocabulary...")
    vocab = build_vocab(load_cached_data=True)

    print("Loading embeddings...")
    embedding_matrix = load_glove_embeddings(vocab.stoi, Config.EMBEDDING_DIM)

    print("Preparing DataLoaders...")
    train_loader, val_loader = get_dataloaders(vocab, load_cached_data=True)

    # 3. Model
    print("Initializing model...")
    # Unfreeze embeddings because we are using random initialization (Cite debug_lesson_3)
    model = FiLMNetwork(embedding_matrix, freeze_embeddings=False)

    # 4. Training
    print("Starting training...")
    trainer = Trainer(model, device)
    trainer.train(train_loader, val_loader, num_epochs=Config.NUM_EPOCHS)

    # 5. Validation & Analysis
    print("\n--- Validation & Failure Analysis ---")
    # Load best model for validation
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    validate_and_analyze(model, vocab, val_loader, device)

    # 6. Submission
    print("\n--- Generating Submission ---")
    predictor = Predictor(device=device, model_path=Config.MODEL_SAVE_PATH)
    results = predictor.generate_predictions(threshold=Config.LONG_ANSWER_THRESHOLD)
    predictor.save_submission(results)


if __name__ == "__main__":
    main()
