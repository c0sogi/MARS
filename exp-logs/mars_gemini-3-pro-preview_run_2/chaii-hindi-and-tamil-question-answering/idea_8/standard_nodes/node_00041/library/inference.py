import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.dataset import prepare_test_features, QADataset
from library.model import CustomXLMRoberta
from library.utils import seed_everything


def get_best_span(
    start_logits,
    end_logits,
    answerable_prob,
    sequence_ids,
    offset_mapping,
    context_text,
    max_answer_len=60,
):
    """
    Finds the optimal span in a single window based on the ensemble score formula:
    Score = (Start_Logit + End_Logit) * Answerable_Prob
    """
    # Create context mask (1 for context tokens, 0 otherwise)
    # sequence_ids: 0=question, 1=context, -1=special
    context_mask = np.array([1 if s == 1 else 0 for s in sequence_ids])

    # Mask logits for non-context tokens to effectively exclude them
    min_val = -10000.0
    s_logits = np.where(context_mask, start_logits, min_val)
    e_logits = np.where(context_mask, end_logits, min_val)

    # Get top-k candidates to reduce complexity
    k = 20
    start_indices = np.argsort(s_logits)[-k:]
    end_indices = np.argsort(e_logits)[-k:]

    best_score = -1e9
    best_span_text = ""

    for start_idx in start_indices:
        for end_idx in end_indices:
            # Basic constraints
            if start_idx > end_idx:
                continue
            if end_idx - start_idx + 1 > max_answer_len:
                continue
            if context_mask[start_idx] == 0 or context_mask[end_idx] == 0:
                continue

            # Calculate score
            # We use the sum of logits weighted by the answerability probability
            score = (s_logits[start_idx] + e_logits[end_idx]) * answerable_prob

            if score > best_score:
                best_score = score
                try:
                    # Map tokens to character positions
                    char_start = offset_mapping[start_idx][0]
                    char_end = offset_mapping[end_idx][1]
                    best_span_text = context_text[char_start:char_end]
                except Exception:
                    continue

    return best_score, best_span_text


def inference_fn():
    # 1. Setup
    seed_everything(Config.seeds[0])
    device = Config.device
    os.makedirs(Config.working_dir, exist_ok=True)

    print(f"Starting Inference. Device: {device}")

    # 2. Load Test Data
    # Use metadata/test.csv if available, else fallback to input
    if os.path.exists(Config.test_path):
        test_path = Config.test_path
    else:
        test_path = os.path.join(Config.input_dir, "test.csv")

    print(f"Loading test data from {test_path}")
    df_test = pd.read_csv(test_path)

    # 3. Prepare Features
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    # prepare_test_features handles caching and returns list of dicts + dataframe
    features, _ = prepare_test_features(
        df_test, tokenizer=tokenizer, load_cached_data=True
    )

    test_dataset = QADataset(features, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 4. Ensemble Inference
    # Initialize accumulators for averaging
    num_samples = len(features)
    seq_len = Config.max_len

    avg_start_logits = np.zeros((num_samples, seq_len), dtype=np.float32)
    avg_end_logits = np.zeros((num_samples, seq_len), dtype=np.float32)
    avg_ans_probs = np.zeros((num_samples,), dtype=np.float32)

    models_used = 0

    for seed in Config.seeds:
        # Construct model path
        # We check for seed-specific checkpoints first
        p1 = os.path.join(Config.working_dir, f"best_model_seed_{seed}.pth")
        p2 = os.path.join(Config.working_dir, "best_model.pth")

        if os.path.exists(p1):
            model_path = p1
        elif os.path.exists(p2) and seed == Config.seeds[0]:
            # Fallback to generic best_model.pth only for the first seed iteration
            # to avoid using the same model multiple times in the average
            model_path = p2
        else:
            print(f"Checkpoint for seed {seed} not found. Skipping.")
            continue

        print(f"Loading model from {model_path}...")
        model = CustomXLMRoberta()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        models_used += 1

        # Batch Inference
        batch_start_preds = []
        batch_end_preds = []
        batch_ans_preds = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                s, e, a = model(input_ids, attention_mask)

                batch_start_preds.append(s.cpu().numpy())
                batch_end_preds.append(e.cpu().numpy())
                # Apply sigmoid to answerability logits to get probability
                batch_ans_preds.append(torch.sigmoid(a).squeeze(-1).cpu().numpy())

        # Accumulate results
        avg_start_logits += np.concatenate(batch_start_preds, axis=0)
        avg_end_logits += np.concatenate(batch_end_preds, axis=0)
        avg_ans_probs += np.concatenate(batch_ans_preds, axis=0)

        # Cleanup to save memory
        del model, state_dict, batch_start_preds, batch_end_preds, batch_ans_preds
        torch.cuda.empty_cache()
        gc.collect()

    if models_used == 0:
        print("Error: No models found for inference.")
        return

    # Compute Average
    avg_start_logits /= models_used
    avg_end_logits /= models_used
    avg_ans_probs /= models_used

    print(f"Inference complete using {models_used} models. Post-processing...")

    # 5. Post-Processing
    final_preds = {}  # example_id -> (score, text)

    for idx, feature in enumerate(features):
        eid = feature["example_id"]

        score, text = get_best_span(
            avg_start_logits[idx],
            avg_end_logits[idx],
            avg_ans_probs[idx],
            feature["sequence_ids"],
            feature["offset_mapping"],
            feature["context"],
        )

        # Maximize score across all windows belonging to the same document
        if eid not in final_preds or score > final_preds[eid][0]:
            final_preds[eid] = (score, text)

    # 6. Save Submission
    submission_rows = []
    all_ids = df_test["id"].unique()

    for eid in all_ids:
        # Default to empty string if no valid prediction found
        text = final_preds.get(eid, (0, ""))[1]
        submission_rows.append({"id": eid, "PredictionString": text})

    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
    print(sub_df.head())
