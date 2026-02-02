import sys
import os
import torch
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from sklearn.metrics import f1_score

# Import library modules
from library.config import Config
from library.vocab_manager import VocabManager
from library.data_loader import get_data_loaders
from library.model import WindowMaxPoolingNetwork
from library.solver import Solver


def set_seed(seed):
    """Sets fixed random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_micro_f1(preds, gts):
    """
    Computes Micro F1 score.
    preds: list of prediction strings (e.g., "10:20", "YES", "")
    gts: list of ground truth strings
    """
    tp = 0
    fp = 0
    fn = 0

    for p, g in zip(preds, gts):
        # Normalize
        p = str(p).strip()
        g = str(g).strip()

        # If both are empty (no answer predicted, no answer exists) -> True Negative (doesn't affect F1 usually, but for NQ blank match is good?)
        # NQ Metric typically:
        # If GT is empty and Pred is empty: Match (TP=1? No, usually treated as correct non-answer)
        # Actually, standard F1 formulation:
        # If GT is present:
        #   If Pred == GT: TP++
        #   Else: FN++ (missed), FP++ (wrong pred) -> Wait, standard definition:
        #   Precision = TP / (TP + FP)
        #   Recall = TP / (TP + FN)

        # Simplified logic for NQ Micro F1 over all items:
        # Treat each question as having a set of answers.
        # Here we have 1 GT and 1 Pred per item.

        if g == "":
            if p != "":
                fp += 1
            # else: TN, ignore
        else:
            if p == g:
                tp += 1
            else:
                fn += 1  # Missed the correct one
                if p != "":
                    fp += 1  # And predicted a wrong one

    epsilon = 1e-9
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1


def evaluate_model(model, val_loader, config, device):
    """
    Runs inference on validation set and computes Micro F1.
    Also returns a DataFrame for failure analysis.
    """
    model.eval()

    # Store data for aggregation
    # example_id -> list of window predictions
    grouped_preds = defaultdict(list)

    # Store Ground Truths: example_id -> {'long': str, 'short': str, 'q_len': int}
    ground_truths = {}

    print("Evaluating on validation set...")

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            question_ids = batch["question_ids"].to(device)

            # Forward pass
            window_score, start_logits, end_logits, yes_no_logits = model(
                input_ids, question_ids
            )

            # Probs and Preds
            window_probs = torch.sigmoid(window_score).squeeze(-1).cpu().numpy()
            pred_starts = torch.argmax(start_logits, dim=1).cpu().numpy()
            pred_ends = torch.argmax(end_logits, dim=1).cpu().numpy()
            pred_yes_no = torch.argmax(yes_no_logits, dim=1).cpu().numpy()

            # Metadata
            example_ids = batch["example_id"]
            candidate_indices = batch["candidate_index"].numpy()
            global_starts = batch["global_start"].numpy()
            window_indices = batch["window_index"].numpy()

            # Labels for GT reconstruction
            label_window = batch["label_window"].numpy()
            label_start = batch["label_start"].numpy()
            label_end = batch["label_end"].numpy()
            label_yes_no = batch["label_yes_no"].numpy()

            # Question length (count non-padding)
            q_lens = (batch["question_ids"] != 0).sum(dim=1).numpy()

            for i in range(len(example_ids)):
                eid = example_ids[i]

                # Store Prediction Candidate
                grouped_preds[eid].append(
                    {
                        "score": window_probs[i],
                        "cand_idx": candidate_indices[i],
                        "rel_start": pred_starts[i],
                        "rel_end": pred_ends[i],
                        "yn_class": pred_yes_no[i],
                        "global_w_start": global_starts[i],
                    }
                )

                # Construct Ground Truth if this is a positive window
                # Note: A question might have multiple positive windows (overlapping).
                # We only need to capture the GT once per ID.
                if eid not in ground_truths:
                    # Initialize with empty (assuming unanswerable until found)
                    ground_truths[eid] = {
                        "long": "",
                        "short": "",
                        "q_len": int(q_lens[i]),
                    }

                if label_window[i] == 1.0:
                    # This window contains valid answer info
                    # Reconstruct GT strings
                    # Note: We don't have the raw candidate start/end here easily for Long GT string
                    # without loading the heavy JSON again.
                    # However, for the metric, we compare our prediction string to the GT string.
                    # Since we are in a constrained environment, we will use a proxy:
                    # We will compare (CandidateIndex) for Long and (GlobalStart:GlobalEnd) for Short.
                    # This is exact enough for internal validation.

                    gt_long_str = str(
                        candidate_indices[i]
                    )  # Proxy for "Start:End" of candidate

                    gt_short_str = ""
                    if label_yes_no[i] == 1:
                        gt_short_str = "YES"
                    elif label_yes_no[i] == 2:
                        gt_short_str = "NO"
                    else:
                        # Span
                        # Check if span labels are valid (not 0,0 unless it's actually 0,0)
                        # In NQ, if yes_no is NONE, and it's a positive window, it must be a span or just long.
                        # If label_start == 0 and label_end == 0, it might be just long answer.
                        # We check if has_short_answer logic was applied in WindowProcessor.
                        # WindowProcessor sets label_window=1 for Long-only too.
                        # We need to distinguish.
                        # But wait, WindowProcessor sets label_start/end only if target_short_start != -1.
                        # If they are 0 and 0, it could be the first token or no short answer.
                        # We'll assume if label_start != label_end, it's a span.
                        if label_start[i] != label_end[i] or (
                            label_start[i] == 0
                            and label_end[i] == 0
                            and label_yes_no[i] == 0
                        ):
                            # To be safe, we rely on the fact that if it was just Long,
                            # WindowProcessor sets labels to 0.
                            # But 0:0 is a valid span (1 token).
                            # Let's assume if it's a positive window and not YES/NO, it's a short span
                            # IF the span is valid in context.
                            # Actually, simpler: We can't perfectly reconstruct the string "100:105"
                            # without the raw text mapping if we don't trust the relative mapping.
                            # But `global_w_start + label_start` IS the global index.
                            s = global_starts[i] + label_start[i]
                            e = global_starts[i] + label_end[i]
                            gt_short_str = f"{s}:{e}"

                    # Update GT (Long answer is always relevant if window is relevant)
                    ground_truths[eid]["long"] = gt_long_str
                    # Only update short if we found a specific short answer type
                    if (
                        gt_short_str != "0:0"
                    ):  # Avoid overwriting with default if we processed a better window before
                        ground_truths[eid]["short"] = gt_short_str
                    elif label_yes_no[i] != 0:
                        ground_truths[eid]["short"] = gt_short_str

    # Aggregation
    final_long_preds = []
    final_short_preds = []
    final_long_gts = []
    final_short_gts = []

    analysis_data = []

    for eid, preds in grouped_preds.items():
        # Find best window
        best_pred = max(preds, key=lambda x: x["score"])

        # Prediction Strings
        pred_long_str = ""
        pred_short_str = ""

        if best_pred["score"] >= config.LONG_ANSWER_CONFIDENCE_THRESHOLD:
            # Long Answer Proxy: Candidate Index
            pred_long_str = str(best_pred["cand_idx"])

            # Short Answer
            if best_pred["yn_class"] == 1:
                pred_short_str = "YES"
            elif best_pred["yn_class"] == 2:
                pred_short_str = "NO"
            else:
                s = best_pred["global_w_start"] + best_pred["rel_start"]
                e = best_pred["global_w_start"] + best_pred["rel_end"]
                if e >= s:
                    pred_short_str = f"{s}:{e}"

        # GT Strings
        gt_data = ground_truths.get(eid, {"long": "", "short": "", "q_len": 0})
        gt_long_str = gt_data["long"]
        gt_short_str = gt_data["short"]

        # Collect for Metric
        final_long_preds.append(pred_long_str)
        final_long_gts.append(gt_long_str)

        final_short_preds.append(pred_short_str)
        final_short_gts.append(gt_short_str)

        # Collect for Analysis
        # Calculate instance F1 (approximate: 1.0 if match, 0.0 if not)
        # Average of long and short match
        long_match = 1.0 if pred_long_str == gt_long_str else 0.0
        short_match = 1.0 if pred_short_str == gt_short_str else 0.0

        # Handle the case where both are empty (True Negative) -> Perfect score
        if pred_long_str == "" and gt_long_str == "":
            long_match = 1.0
        if pred_short_str == "" and gt_short_str == "":
            short_match = 1.0

        avg_f1 = (long_match + short_match) / 2.0
        error = 1.0 - avg_f1

        analysis_data.append(
            {"example_id": eid, "error": error, "q_len": gt_data["q_len"]}
        )

    # Compute Global Micro F1
    # Combine lists
    all_preds = final_long_preds + final_short_preds
    all_gts = final_long_gts + final_short_gts

    total_f1 = compute_micro_f1(all_preds, all_gts)

    return total_f1, pd.DataFrame(analysis_data)


def perform_failure_analysis(df):
    """
    Correlates error with features.
    """
    if df.empty:
        print("No data for failure analysis.")
        return

    print("\n--- Failure Analysis ---")
    print(f"Analyzed {len(df)} validation examples.")

    # Correlation with Question Length
    corr_q = df["error"].corr(df["q_len"])
    print(f"Correlation between Error and Question Length: {corr_q:.4f}")

    # We could analyze more if we had doc length, but q_len is readily available.
    if abs(corr_q) > 0.1:
        print("Observation: Significant correlation found.")
    else:
        print("Observation: No significant linear correlation with question length.")


def main():
    # 1. Configuration
    config = Config()
    # Modify for Fast Baseline
    config.NUM_EPOCHS = 1
    # config.DEBUG = True # Uncomment to run on tiny subset for testing logic
    config.setup()

    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Vocabulary & Embeddings
    print("Initializing Vocabulary...")
    vocab_manager = VocabManager(config)
    vocab_manager.build_vocab(load_cached_data=True)

    # 3. Data Loaders
    print("Preparing Data Loaders...")
    train_loader, val_loader, test_loader = get_data_loaders(
        config, vocab_manager, load_cached_data=True
    )

    # 4. Model
    print("Initializing Model...")
    embedding_matrix = vocab_manager.get_embedding_matrix()
    model = WindowMaxPoolingNetwork(embedding_matrix, config)

    # 5. Training
    print("Starting Training...")
    solver = Solver(model, config, device=device)
    solver.train(train_loader, val_loader)

    # 6. Validation & Metrics
    print("\nStarting Validation Assessment...")
    val_f1, val_errors_df = evaluate_model(model, val_loader, config, device)
    print(f"Final Validation Metric: {val_f1:.16f}")

    # 7. Failure Analysis
    perform_failure_analysis(val_errors_df)

    # 8. Submission
    print("\nGenerating Submission...")
    solver.inference(test_loader)
    print("Done.")


if __name__ == "__main__":
    main()
