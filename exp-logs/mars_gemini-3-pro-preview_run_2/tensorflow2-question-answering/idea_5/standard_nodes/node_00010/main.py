import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from library.config import Config
from library.utils import set_seed
from library.data_processing import DataProcessor
from library.dataset import NQDataset
from library.model import FeedForwardDecomposableAttention
from library.trainer import Trainer
from library.inference import InferenceManager


def calculate_f1(true_positives, false_positives, false_negatives):
    if true_positives == 0:
        return 0.0
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def get_ground_truth(data_row):
    """
    Extracts ground truth strings from the raw data row.
    Returns: (long_ans_str, short_ans_str)
    """
    annotations = json.loads(data_row["annotations"])
    if not annotations:
        return "", ""

    ann = annotations[0]

    # Long Answer
    long_ans = ann.get("long_answer", {})
    l_idx = long_ans.get("candidate_index", -1)

    gt_long = ""
    if l_idx != -1:
        candidates = json.loads(data_row["long_answer_candidates"])
        if l_idx < len(candidates):
            cand = candidates[l_idx]
            gt_long = f"{cand['start_token']}:{cand['end_token']}"

    # Short Answer
    gt_short = ""
    yes_no = ann.get("yes_no_answer", "NONE")
    if yes_no != "NONE":
        gt_short = yes_no
    else:
        shorts = ann.get("short_answers", [])
        if shorts:
            s = shorts[0]
            gt_short = f"{s['start_token']}:{s['end_token']}"

    return gt_long, gt_short


def run_validation_assessment(config, model, processor):
    print("\n--- Starting Validation Assessment ---")

    # Use validation split
    val_dataset = NQDataset(config, processor, split="val", load_cached_data=True)

    # Batch size 1 to handle one example (with all its candidates) at a time
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=NQDataset.collate_fn,
        num_workers=0,
    )

    device = torch.device(config.DEVICE)
    model.eval()

    tp = 0
    fp = 0
    fn = 0

    # For Failure Analysis
    analysis_data = []

    idx_to_yn = {0: "YES", 1: "NO", 2: "NONE"}

    with torch.no_grad():
        for batch in val_loader:
            if not batch:
                continue

            q_input = batch["q_input"].to(device)
            c_input = batch["c_input"].to(device)
            example_ids = batch["example_ids"]
            cand_starts = batch["cand_starts"]
            cand_ends = batch["cand_ends"]

            curr_id = example_ids[0]

            # Forward
            outputs = model(q_input, c_input)

            # Ranking
            ranking_logits = outputs["ranking_logits"].squeeze(-1)
            ranking_scores = torch.sigmoid(ranking_logits)

            best_score, best_idx = torch.max(ranking_scores, dim=0)
            best_idx = best_idx.item()
            best_score = best_score.item()

            # Predictions
            pred_long = ""
            pred_short = ""

            if best_score >= config.CONFIDENCE_THRESHOLD:
                l_start = cand_starts[best_idx]
                l_end = cand_ends[best_idx]
                pred_long = f"{l_start}:{l_end}"

                # Short / Yes-No
                yn_logits = outputs["yn_logits"][best_idx]
                yn_idx = torch.argmax(yn_logits).item()
                yn_label = idx_to_yn.get(yn_idx, "NONE")

                if yn_label in ["YES", "NO"]:
                    pred_short = yn_label
                else:
                    start_logits = outputs["start_logits"][best_idx]
                    end_logits = outputs["end_logits"][best_idx]

                    s_rel = torch.argmax(start_logits).item()
                    e_rel = torch.argmax(end_logits).item()

                    if s_rel <= e_rel:
                        # Convert relative to absolute
                        # Note: NQ format is start_token:end_token (exclusive end usually in python,
                        # but NQ annotations are token based).
                        # Dataset logic: s_end_rel was rel_e - 1 (inclusive).
                        # So original end token index = l_start + e_rel + 1
                        s_abs = l_start + s_rel
                        e_abs = l_start + e_rel + 1
                        pred_short = f"{s_abs}:{e_abs}"

            # Ground Truth
            raw_row = val_dataset.id_to_data[curr_id]
            gt_long, gt_short = get_ground_truth(raw_row)

            # Update Metrics (Micro F1 Logic)
            # Check Long
            if pred_long == gt_long:
                if pred_long != "":
                    tp += 1
            else:
                if pred_long != "":
                    fp += 1
                if gt_long != "":
                    fn += 1

            # Check Short
            if pred_short == gt_short:
                if pred_short != "":
                    tp += 1
            else:
                if pred_short != "":
                    fp += 1
                if gt_short != "":
                    fn += 1

            # Failure Analysis Data
            # Error = 1 if either long or short is wrong
            is_error = 0
            if pred_long != gt_long or pred_short != gt_short:
                is_error = 1

            doc_len = len(raw_row["document_text"].split())
            q_len = len(raw_row["question_text"].split())

            analysis_data.append(
                {"error": is_error, "doc_len": doc_len, "q_len": q_len}
            )

    # Compute Final Metric
    final_f1 = calculate_f1(tp, fp, fn)
    print(f"Final Validation Metric: {final_f1}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    df_analysis = pd.DataFrame(analysis_data)
    if not df_analysis.empty:
        corr_doc = df_analysis["error"].corr(df_analysis["doc_len"])
        corr_q = df_analysis["error"].corr(df_analysis["q_len"])

        print("Correlation between Error and Input Features:")
        print(f"  Document Length: {corr_doc:.4f}")
        print(f"  Question Length: {corr_q:.4f}")
    else:
        print("No analysis data available.")


def main():
    # 1. Configuration
    config = Config()

    # --- FAST BASELINE SETTINGS ---
    config.DEBUG_SAMPLE_SIZE = 2000  # Limit samples for speed
    config.NUM_EPOCHS = 1  # Single epoch for baseline
    config.BATCH_SIZE = 32  # Adjust for memory safety
    # ------------------------------

    set_seed(config.SEED)
    config.display()

    # 2. Data Processing
    print("\n[1/5] Processing Data...")
    processor = DataProcessor(config)
    vocab = processor.build_vocab(load_cached_data=True)
    embedding_matrix = processor.create_embedding_matrix(load_cached_data=True)

    # 3. Datasets
    print("\n[2/5] Preparing Datasets...")
    train_dataset = NQDataset(config, processor, split="train", load_cached_data=True)
    val_dataset = NQDataset(config, processor, split="val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=NQDataset.collate_fn,
        num_workers=0,
    )

    # Validation loader for Trainer (batch-wise metrics)
    val_loader_batch = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=NQDataset.collate_fn,
        num_workers=0,
    )

    # 4. Model
    print("\n[3/5] Initializing Model...")
    embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)
    model = FeedForwardDecomposableAttention(config, embedding_tensor)

    # 5. Training
    print("\n[4/5] Training...")
    trainer = Trainer(model, train_loader, val_loader_batch, config)
    trainer.train()

    # 6. Validation Assessment & Failure Analysis
    run_validation_assessment(config, model, processor)

    # 7. Inference & Submission
    print("\n[5/5] Generating Submission...")
    inference_manager = InferenceManager(config)
    # Reload best model weights for inference
    if os.path.exists(config.MODEL_CHECKPOINT_PATH):
        checkpoint = torch.load(
            config.MODEL_CHECKPOINT_PATH, map_location=config.DEVICE
        )
        inference_manager.model.load_state_dict(checkpoint["model_state_dict"])

    # Run inference on full test set (or debug size if config persists, reset here if needed)
    # We keep config.DEBUG_SAMPLE_SIZE as set at start for consistency in this fast run
    submission_df = inference_manager.generate_predictions()
    inference_manager.save_submission(submission_df)

    print("\nRunfile execution completed.")


if __name__ == "__main__":
    main()
