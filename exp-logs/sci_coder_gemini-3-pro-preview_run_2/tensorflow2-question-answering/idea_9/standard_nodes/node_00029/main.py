import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from typing import List, Dict, Set

# Import library modules
from library.config import Config
from library.utils import build_vocab, create_embedding_matrix
from library.data import get_dataloaders, YES_NO_MAP
from library.model import SingleStreamNetwork
from library.engine import Engine, IDX_TO_YES_NO


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_f1(
    predictions: Dict[str, str], ground_truths: Dict[str, List[str]]
) -> float:
    """
    Computes Micro F1 score.
    predictions: dict {example_id_type: prediction_string}
    ground_truths: dict {example_id_type: [valid_answer_string_1, valid_answer_string_2, ...]}
    """
    tp = 0
    fp = 0
    fn = 0

    # The set of all keys is the union of both
    all_keys = set(predictions.keys()) | set(ground_truths.keys())

    for key in all_keys:
        pred = predictions.get(key, "")
        gts = ground_truths.get(key, [])

        # Normalize empty strings
        if pred == "nan":
            pred = ""

        # If GT is empty list, it means no answer.
        # In NQ, "no answer" is often represented as blank string in submission.
        # If GT list is empty, treat as [""] for comparison if we consider blank as a valid label for "no answer"
        # However, standard F1 usually considers positives.
        # For NQ Kaggle metric:
        # If there is no ground truth answer, and prediction is blank -> Match (TP? or TN?)
        # Actually, the metric is F1.
        # Let's follow standard interpretation:
        # If GT has answers and Pred matches one -> TP
        # If GT has answers and Pred does not match -> FN (and FP if Pred was not blank)
        # If GT has NO answers (empty list) and Pred is blank -> TN (ignored in F1)
        # If GT has NO answers and Pred is NOT blank -> FP

        # Adjust GT for "no answer" cases to be explicit
        if not gts:
            gts = [""]

        matched = False
        if pred in gts:
            matched = True

        # Logic for TP/FP/FN
        # Case 1: Target is blank (No Answer)
        if "" in gts and len(gts) == 1:
            if pred == "":
                # True Negative. Doesn't affect F1 numerator or denominator usually,
                # but in some implementations it counts as accuracy.
                # Strict F1 ignores TN.
                pass
            else:
                # False Positive
                fp += 1
        # Case 2: Target is not blank
        else:
            if pred == "":
                # False Negative
                fn += 1
            else:
                if matched:
                    tp += 1
                else:
                    # Predicted something wrong
                    fp += 1
                    fn += 1  # And missed the real one

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def get_validation_predictions(engine, dataloader, val_df):
    """
    Runs inference on validation set and returns a dictionary of predictions.
    Reuses logic from engine.predict but returns dict instead of writing CSV.
    """
    engine.model.eval()
    all_predictions = {}

    # Map example_id to candidates for coordinate retrieval
    val_data_map = val_df.set_index("example_id")["candidates"].to_dict()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(engine.device)
            outputs = engine.model(input_ids)

            ranking_scores = torch.sigmoid(outputs["ranking_logits"]).cpu().numpy()
            start_logits = outputs["start_logits"].cpu().numpy()
            end_logits = outputs["end_logits"].cpu().numpy()
            yes_no_logits = outputs["yes_no_logits"].cpu().numpy()

            # In validation collator (is_train=True), we don't get example_ids/cand_indices directly in batch
            # We need to reconstruct or modify collator.
            # However, we cannot modify library files.
            # But wait, NQCollator returns 'example_ids' only if is_train=False.
            # For validation, we instantiated val_loader with is_train=True in data.py to get labels for loss.
            # This means we CANNOT map predictions back to example_ids easily using the provided val_loader
            # because the collator swallows the IDs.

            # WORKAROUND:
            # We will instantiate a separate "inference" validation loader where is_train=False.
            # This allows us to get example_ids and candidate_indices to map predictions.
            pass

    # Since we need to create a new loader, we'll do it outside this function or handle it here.
    return {}  # Placeholder, logic moved to main execution


def main():
    # 1. Configuration Overrides for Fast Baseline
    Config.DATASET_FRACTION = 0.05  # Use 5% of data
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 32  # Safe batch size

    # Setup
    set_seed(Config.SEED)
    Config.setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    # Load vocab first to ensure consistency
    # We force load_cached_data=True, assuming cache exists or will be built
    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        load_cached_data=True, debug=True
    )

    # Create a specific validation loader for inference (is_train=False)
    # This is necessary because the default val_loader (is_train=True) doesn't return example_ids needed for mapping
    from library.data import NQDataset, NQCollator

    # Load processed val dataframe directly
    val_df = pd.read_parquet(Config.PROCESSED_VAL_DATA_PATH)

    # Cite debug_lesson_5: Explicitly convert NumPy arrays from Parquet back to lists
    # to prevent broadcasting errors during concatenation in the Collator.
    for col in ["question_ids", "document_ids", "candidates", "short_answers"]:
        if col in val_df.columns:
            val_df[col] = val_df[col].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else list(x)
            )

    val_inf_ds = NQDataset(val_df, is_train=False)
    val_inf_collate = NQCollator(vocab, is_train=False)
    val_inf_loader = torch.utils.data.DataLoader(
        val_inf_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=val_inf_collate,
        num_workers=2,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = SingleStreamNetwork(vocab).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    engine = Engine(model, device, optimizer)

    # 4. Training Loop
    print("Starting Training...")
    for epoch in range(Config.NUM_EPOCHS):
        train_loss = engine.train_one_epoch(train_loader)
        val_metrics = engine.validate(val_loader)
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_metrics['val_loss']:.4f}"
        )

        # Save checkpoint
        torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # 5. Validation Assessment & F1 Calculation
    print("Running Validation Inference for F1...")
    engine.model.eval()

    val_preds_map = {}  # example_id -> list of predictions

    with torch.no_grad():
        for batch in val_inf_loader:
            input_ids = batch["input_ids"].to(device)
            outputs = engine.model(input_ids)

            ranking_scores = torch.sigmoid(outputs["ranking_logits"]).cpu().numpy()
            start_logits = outputs["start_logits"].cpu().numpy()
            end_logits = outputs["end_logits"].cpu().numpy()
            yes_no_logits = outputs["yes_no_logits"].cpu().numpy()

            example_ids = batch["example_ids"]
            cand_indices = batch["candidate_indices"]
            input_seqs = input_ids.cpu().numpy()
            sep_id = vocab[Config.SEP_TOKEN]

            for i in range(len(example_ids)):
                eid = example_ids[i]
                c_idx = cand_indices[i]
                rank_score = ranking_scores[i]

                # Find SEP offset
                try:
                    sep_pos = np.where(input_seqs[i] == sep_id)[0][0]
                    cand_offset = sep_pos + 1
                except:
                    cand_offset = 0

                s_idx = np.argmax(start_logits[i])
                e_idx = np.argmax(end_logits[i])
                yn_idx = np.argmax(yes_no_logits[i])

                if eid not in val_preds_map:
                    val_preds_map[eid] = []

                val_preds_map[eid].append(
                    {
                        "cand_idx": c_idx,
                        "rank_score": rank_score,
                        "s_idx": s_idx,
                        "e_idx": e_idx,
                        "offset": cand_offset,
                        "yn_idx": yn_idx,
                    }
                )

    # Construct final prediction strings
    final_preds = {}  # key: example_id_type, value: string

    # Need candidate offsets for mapping
    # val_df has 'candidates' column: list of [start, end, top_level]
    val_candidates_map = val_df.set_index("example_id")["candidates"].to_dict()

    for eid, preds in val_preds_map.items():
        # Sort by rank score
        preds.sort(key=lambda x: x["rank_score"], reverse=True)
        best = preds[0]

        long_str = ""
        short_str = ""

        if best["rank_score"] >= Config.LONG_ANSWER_THRESHOLD:
            candidates = val_candidates_map.get(str(eid), [])
            if best["cand_idx"] < len(candidates):
                c_start, c_end, _ = candidates[best["cand_idx"]]
                long_str = f"{c_start}:{c_end}"

                # Short Answer
                s_seq = best["s_idx"]
                e_seq = best["e_idx"]
                offset = best["offset"]

                if s_seq >= offset and e_seq >= offset and s_seq <= e_seq:
                    global_s = c_start + (s_seq - offset)
                    global_e = c_start + (e_seq - offset)

                    if global_e < c_end:
                        yn_label = IDX_TO_YES_NO.get(best["yn_idx"], "NONE")
                        if yn_label in ["YES", "NO"]:
                            short_str = yn_label
                        else:
                            short_str = f"{global_s}:{global_e + 1}"

        final_preds[f"{eid}_long"] = long_str
        final_preds[f"{eid}_short"] = short_str

    # Build Ground Truth Dictionary
    ground_truths = {}

    # val_df has labels but we need to parse them into strings
    # The 'short_answers' column in val_df (from preprocess_data) contains list of (start, end)
    # The 'long_answer_index' contains index.
    # We need to map these to strings.

    for _, row in val_df.iterrows():
        eid = str(row["example_id"])
        candidates = row["candidates"]

        # Long Answer GT
        la_idx = row["long_answer_index"]
        la_gts = []
        if la_idx != -1 and la_idx < len(candidates):
            ls, le, _ = candidates[la_idx]
            la_gts.append(f"{ls}:{le}")
        else:
            la_gts.append("")
        ground_truths[f"{eid}_long"] = la_gts

        # Short Answer GT
        sa_gts = []
        # Check Yes/No first
        yn_val = row["yes_no_answer"]  # 0=NONE, 1=YES, 2=NO
        yn_str = IDX_TO_YES_NO.get(yn_val, "NONE")

        if yn_str in ["YES", "NO"]:
            sa_gts.append(yn_str)
        else:
            # Check spans
            shorts = row["short_answers"]  # list of [start, end]
            if shorts:
                for s, e in shorts:
                    sa_gts.append(
                        f"{s}:{e}"
                    )  # NQ raw is usually exclusive end, our logic assumes inclusive?
                    # The prompt sample says 105:200. Let's stick to raw integers.
            else:
                sa_gts.append("")

        ground_truths[f"{eid}_short"] = sa_gts

    # Compute Metric
    f1_score = compute_f1(final_preds, ground_truths)
    print(f"Final Validation Metric: {f1_score}")

    # 6. Failure Analysis
    print("Performing Failure Analysis...")
    errors = []
    doc_lens = []
    q_lens = []

    for _, row in val_df.iterrows():
        eid = str(row["example_id"])

        # Calculate F1 for this instance (average of long and short)
        # Construct mini-dicts for this instance
        inst_preds = {
            f"{eid}_long": final_preds.get(f"{eid}_long", ""),
            f"{eid}_short": final_preds.get(f"{eid}_short", ""),
        }
        inst_gts = {
            f"{eid}_long": ground_truths.get(f"{eid}_long", [""]),
            f"{eid}_short": ground_truths.get(f"{eid}_short", [""]),
        }

        inst_f1 = compute_f1(inst_preds, inst_gts)
        errors.append(1.0 - inst_f1)

        # Features
        doc_lens.append(len(row["document_ids"]))
        q_lens.append(len(row["question_ids"]))

    # Correlation
    if len(errors) > 1:
        corr_doc = np.corrcoef(errors, doc_lens)[0, 1]
        corr_q = np.corrcoef(errors, q_lens)[0, 1]
        print(f"Correlation (Error vs Doc Length): {corr_doc:.4f}")
        print(f"Correlation (Error vs Question Length): {corr_q:.4f}")
    else:
        print("Not enough data for correlation analysis.")

    # 7. Test Inference & Submission
    print("Generating Test Submission...")
    # Load test dataframe for candidate mapping
    test_df = pd.read_parquet(Config.PROCESSED_TEST_DATA_PATH)

    # Cite debug_lesson_5: Explicitly convert NumPy arrays from Parquet back to lists
    for col in ["question_ids", "document_ids", "candidates"]:
        if col in test_df.columns:
            test_df[col] = test_df[col].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else list(x)
            )

    engine.predict(test_loader, test_df)


if __name__ == "__main__":
    main()
