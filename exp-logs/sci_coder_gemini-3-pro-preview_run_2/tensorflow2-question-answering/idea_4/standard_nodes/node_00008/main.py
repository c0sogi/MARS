import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import collections
from tqdm import tqdm

from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint, compute_f1
from library.vocab import build_vocab, build_embedding_matrix
from library.data import NQDataset, collate_fn
from library.model import SiameseBiLSTM
from library.engine import train_one_epoch, evaluate


def get_best_span(start_logits, end_logits):
    """
    Finds the best valid span (start <= end) from logits.
    Simple greedy approach: argmax start, then argmax end >= start.
    """
    start_idx = torch.argmax(start_logits).item()
    # Mask end positions before start_idx
    end_probs = torch.softmax(end_logits, dim=0)
    end_probs[:start_idx] = 0
    end_idx = torch.argmax(end_probs).item()
    return start_idx, end_idx


def run_inference(model, dataloader, dataset, device, is_test=False, threshold=0.5):
    model.eval()
    results = collections.defaultdict(list)

    # Yes/No map inverse
    yn_map_inv = {0: "NONE", 1: "YES", 2: "NO"}

    with torch.no_grad():
        for batch in dataloader:
            if not batch:
                continue

            q_input_ids = batch["q_input_ids"].to(device)
            c_input_ids = batch["c_input_ids"].to(device)

            outputs = model(q_input_ids, c_input_ids)

            # Move to CPU
            rank_scores = outputs["rank_score"].cpu().numpy()
            start_logits = outputs["span_start_logits"].cpu()
            end_logits = outputs["span_end_logits"].cpu()
            yn_logits = outputs["yn_logits"].cpu()

            example_ids = batch["example_ids"]
            global_cand_idxs = batch["global_cand_idxs"]

            for i, eid in enumerate(example_ids):
                # Retrieve candidate metadata from dataset using global index
                # We need to access the raw data map in the dataset
                raw_entry = dataset.data_map[eid]
                cand_idx = global_cand_idxs[i]
                candidate = raw_entry["long_answer_candidates"][cand_idx]

                # Get span
                s_idx, e_idx = get_best_span(start_logits[i], end_logits[i])

                # Get Yes/No
                yn_idx = torch.argmax(yn_logits[i]).item()
                yn_pred = yn_map_inv[yn_idx]

                results[eid].append(
                    {
                        "score": rank_scores[i],
                        "cand_info": candidate,
                        "rel_span": (s_idx, e_idx),
                        "yn_pred": yn_pred,
                        "q_len": len(raw_entry["question_text"].split()),
                    }
                )

    # Process aggregated results
    final_predictions = {}
    metrics_data = []  # For failure analysis: (f1, q_len, c_len)

    # Ground truth for validation
    gt_map = {}
    if not is_test:
        for eid, entry in dataset.data_map.items():
            anns = entry.get("annotations", [])
            if not anns:
                continue
            ann = anns[0]

            # Long Answer GT
            la = ann.get("long_answer", {})
            la_idx = la.get("candidate_index", -1)
            la_token_span = None
            if la_idx != -1:
                c = entry["long_answer_candidates"][la_idx]
                la_token_span = (c["start_token"], c["end_token"])

            # Short Answer GT
            sa_list = ann.get("short_answers", [])
            sa_spans = [(s["start_token"], s["end_token"]) for s in sa_list]

            # Yes/No GT
            yn_gt = ann.get("yes_no_answer", "NONE")

            gt_map[eid] = {
                "long_span": la_token_span,
                "short_spans": sa_spans,
                "yes_no": yn_gt,
            }

    total_f1 = 0.0
    count = 0

    for eid, candidates in results.items():
        # Sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best_cand = candidates[0]

        # Default Predictions
        pred_long_span_str = ""
        pred_short_span_str = ""

        # Apply Threshold
        if best_cand["score"] >= threshold:
            # Long Answer Prediction (Document Token Indices)
            c_start = best_cand["cand_info"]["start_token"]
            c_end = best_cand["cand_info"]["end_token"]
            pred_long_span = (c_start, c_end)
            pred_long_span_str = f"{c_start}:{c_end}"

            # Short Answer Prediction
            yn = best_cand["yn_pred"]
            if yn != "NONE":
                pred_short_span_str = yn
                # For F1 calculation, YES/NO are treated as tokens
                pred_short_tokens = [yn]
            else:
                # Span prediction
                rel_s, rel_e = best_cand["rel_span"]
                # Convert to document absolute indices
                # rel_e is inclusive index from get_best_span logic?
                # Wait, get_best_span returns indices.
                # NQ format expects start:end where end is exclusive.
                # Model predicts start and end token indices.
                # So span is doc_tokens[abs_start : abs_end + 1]

                abs_start = c_start + rel_s
                abs_end = c_start + rel_e + 1  # +1 for exclusive upper bound

                # Sanity check
                if abs_start < c_end:
                    pred_short_span_str = f"{abs_start}:{abs_end}"
                    # Retrieve tokens for F1
                    doc_tokens = dataset.data_map[eid]["doc_tokens"]
                    pred_short_tokens = doc_tokens[abs_start:abs_end]
                else:
                    pred_short_tokens = []
        else:
            pred_long_span = None
            pred_short_tokens = []

        final_predictions[eid] = {
            "long": pred_long_span_str,
            "short": pred_short_span_str,
        }

        # Validation Metrics
        if not is_test and eid in gt_map:
            gt = gt_map[eid]

            # Long Answer F1
            # If both None: 1.0. If one None: 0.0. Else overlap.
            # Task description says "match exactly the token indices".
            # So exact match for Long Answer.
            if gt["long_span"] is None and pred_long_span is None:
                f1_long = 1.0
            elif gt["long_span"] is not None and pred_long_span is not None:
                # Tuple comparison
                f1_long = 1.0 if gt["long_span"] == pred_long_span else 0.0
            else:
                f1_long = 0.0

            # Short Answer F1
            # "Short answers are always contained within...".
            # If YES/NO, match string. Else match tokens.
            # Multi-label ground truth: max F1 over gold set.
            f1_short = 0.0

            # Handle Yes/No GT
            if gt["yes_no"] != "NONE":
                # If GT is YES/NO, prediction must match string
                if pred_short_span_str == gt["yes_no"]:
                    f1_short = 1.0
            elif gt["short_spans"]:
                # Span comparison
                # If we predicted YES/NO but GT is span -> 0.0
                # If we predicted span, compute token F1
                if pred_short_span_str not in ["YES", "NO", ""]:
                    # pred_short_tokens populated above
                    best_s_f1 = 0.0
                    for s_span in gt["short_spans"]:
                        # Get GT tokens
                        doc_tokens = dataset.data_map[eid]["doc_tokens"]
                        gt_tokens = doc_tokens[s_span[0] : s_span[1]]
                        s_f1 = compute_f1(pred_short_tokens, gt_tokens)
                        if s_f1 > best_s_f1:
                            best_s_f1 = s_f1
                    f1_short = best_s_f1
                elif pred_short_span_str == "" and not gt["short_spans"]:
                    f1_short = 1.0
            else:
                # No short answer in GT
                if pred_short_span_str == "":
                    f1_short = 1.0

            # "Micro F1": In NQ challenges, usually average of F1 per instance?
            # The prompt says "Metric: Micro F1". Usually this implies aggregating TP/FP/FN over all instances.
            # However, standard NQ eval scripts often average the F1 of (Long) and F1 of (Short).
            # Let's assume average of (Long F1 + Short F1) / 2 per example, then mean over examples.
            # Or is it global Micro? Given the complexity, we will calculate per-example average F1.

            instance_f1 = (f1_long + f1_short) / 2.0
            total_f1 += instance_f1
            count += 1

            # For failure analysis
            c_len = (
                best_cand["cand_info"]["end_token"]
                - best_cand["cand_info"]["start_token"]
            )
            metrics_data.append(
                {"f1": instance_f1, "q_len": best_cand["q_len"], "c_len": c_len}
            )

    if not is_test:
        avg_f1 = total_f1 / count if count > 0 else 0.0
        return avg_f1, metrics_data
    else:
        return final_predictions


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    # Build Vocab
    # Explicitly invalidate cache to ensure full vocab is built (Cite debug_lesson_2)
    vocab = build_vocab(Config.TRAIN_DATA_PATH, load_cached_data=False)

    # Build Embeddings
    # Explicitly invalidate cache to match new vocab (Cite debug_lesson_2)
    embedding_matrix = build_embedding_matrix(vocab, load_cached_data=False)

    # Datasets
    # Remove debug_limit to use full datasets for valid metric calculation
    print("Initializing Datasets...")
    train_dataset = NQDataset(Config.TRAIN_META_PATH, vocab, mode="train")
    val_dataset = NQDataset(Config.VAL_META_PATH, vocab, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # 3. Model
    model = SiameseBiLSTM(embedding_matrix).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 4. Training
    print("Starting training...")
    best_val_loss = float("inf")

    # Train for limited epochs for baseline speed
    epochs = 3

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch + 1)
        val_loss, val_metrics = evaluate(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                os.path.join(Config.WORKING_DIR, "best_model.pth"),
            )

    # 5. Validation Assessment & Failure Analysis
    print("\n--- Validation Assessment & Failure Analysis ---")
    # Load best model
    checkpoint = load_checkpoint(
        os.path.join(Config.WORKING_DIR, "best_model.pth"), model
    )

    # Run full validation inference logic
    val_score, val_analysis_data = run_inference(
        model,
        val_loader,
        val_dataset,
        device,
        is_test=False,
        threshold=Config.LONG_ANSWER_CONFIDENCE_THRESHOLD,
    )

    print(f"Final Validation Metric: {val_score}")

    # Failure Analysis
    df_analysis = pd.DataFrame(val_analysis_data)
    df_analysis["error"] = 1.0 - df_analysis["f1"]

    corr_q = df_analysis["error"].corr(df_analysis["q_len"])
    corr_c = df_analysis["error"].corr(df_analysis["c_len"])

    print("Correlation between Error (1-F1) and Input Features:")
    print(f"  Question Length: {corr_q:.4f}")
    print(f"  Candidate Length: {corr_c:.4f}")

    # 6. Submission Generation
    print("\n--- Generating Submission ---")
    test_dataset = NQDataset(Config.TEST_META_PATH, vocab, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    predictions = run_inference(
        model,
        test_loader,
        test_dataset,
        device,
        is_test=True,
        threshold=Config.LONG_ANSWER_CONFIDENCE_THRESHOLD,
    )

    # Format for CSV
    # sample_submission.csv format:
    # example_id,PredictionString
    # -123_long,start:end
    # -123_short,YES

    output_rows = []
    # Ensure we cover all IDs in test set
    all_test_ids = test_dataset.metadata["example_id"].astype(str).unique()

    for eid in all_test_ids:
        preds = predictions.get(eid, {"long": "", "short": ""})

        # Long row
        output_rows.append(
            {"example_id": f"{eid}_long", "PredictionString": preds["long"]}
        )

        # Short row
        output_rows.append(
            {"example_id": f"{eid}_short", "PredictionString": preds["short"]}
        )

    submission_df = pd.DataFrame(output_rows)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
