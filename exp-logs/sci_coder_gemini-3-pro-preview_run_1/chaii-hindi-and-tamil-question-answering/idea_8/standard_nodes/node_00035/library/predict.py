import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
from transformers import AutoTokenizer

from library.config import Config
from library.dataset import get_test_dataset
from library.modeling import CustomXLMRoberta
from library.utils import seed_everything


def predict(config: Config):
    """
    Runs the inference pipeline:
    1. Loads test data (sliding windows).
    2. Loads ensemble models.
    3. Aggregates logits across models.
    4. Reconstructs answers using Gated Scoring.
    5. Saves submission.csv.
    """
    # 1. Setup
    seed_everything(config.seed)
    device = config.device

    print("Initializing inference pipeline...")

    # We need the tokenizer to process the dataset
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Load test dataset
    # load_cached_data=True allows using cached parquet if available
    test_ds = get_test_dataset(config, tokenizer, load_cached_data=True)

    test_loader = DataLoader(
        test_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    # 2. Load Ensemble Models
    models = []
    print(f"Loading ensemble models for seeds: {config.ensemble_seeds}")

    for seed in config.ensemble_seeds:
        # Construct model path
        model_path = os.path.join(config.output_dir, f"model_seed_{seed}.pth")

        if os.path.exists(model_path):
            try:
                model = CustomXLMRoberta(config)
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                models.append(model)
                print(f"Successfully loaded model: {model_path}")
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
        else:
            print(f"Warning: Model not found at {model_path}. Skipping.")

    if not models:
        print("Error: No models loaded. Generating empty submission.")
        test_df = pd.read_csv(config.test_path)
        sub = pd.DataFrame({"id": test_df["id"], "PredictionString": ""})
        sub.to_csv(config.submission_path, index=False)
        return

    # 3. Inference Loop
    results = defaultdict(list)
    print(f"Starting inference on {len(test_loader)} batches...")

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Initialize accumulators for averaging
            avg_start_logits = torch.zeros(
                input_ids.size(0), input_ids.size(1), device=device
            )
            avg_end_logits = torch.zeros(
                input_ids.size(0), input_ids.size(1), device=device
            )
            avg_rel_logits = torch.zeros(input_ids.size(0), device=device)

            # Forward pass through all models
            for model in models:
                s, e, r = model(input_ids, attention_mask)
                avg_start_logits += s
                avg_end_logits += e
                avg_rel_logits += r

            # Average
            n_models = len(models)
            avg_start_logits /= n_models
            avg_end_logits /= n_models
            avg_rel_logits /= n_models

            # Move to CPU for storage
            avg_start_logits = avg_start_logits.cpu().numpy()
            avg_end_logits = avg_end_logits.cpu().numpy()
            avg_rel_logits = avg_rel_logits.cpu().numpy()
            batch_input_ids = batch["input_ids"].cpu().numpy()

            # Metadata from dataset
            offset_mapping = batch["offset_mapping"].numpy()
            example_ids = batch["example_id"]

            # Store results grouped by example_id
            for i, ex_id in enumerate(example_ids):
                results[ex_id].append(
                    {
                        "start_logits": avg_start_logits[i],
                        "end_logits": avg_end_logits[i],
                        "relevance_logit": avg_rel_logits[i],
                        "offset_mapping": offset_mapping[i],
                        "input_ids": batch_input_ids[i],
                    }
                )

    # 4. Post-processing & Reconstruction
    print("Post-processing predictions...")

    # Load original test CSV to get context text and ensure all IDs are covered
    test_df = pd.read_csv(config.test_path)

    final_preds = []

    for _, row in test_df.iterrows():
        ex_id = row["id"]
        context_text = str(row["context"])

        if ex_id not in results:
            # Fallback if ID somehow missing from processing
            final_preds.append({"id": ex_id, "PredictionString": ""})
            continue

        windows = results[ex_id]
        best_overall_score = -float("inf")
        best_prediction_string = ""

        for win in windows:
            start_logits = win["start_logits"]
            end_logits = win["end_logits"]
            rel_logit = win["relevance_logit"]
            offsets = win["offset_mapping"]
            input_ids = win["input_ids"]

            # Identify Context Boundaries
            # XLM-R structure: <s> (0) Question </s> (2) </s> (2) Context </s> (2)
            # We look for the sequence of separator tokens (id 2)
            sep_indices = np.where(input_ids == 2)[0]

            if len(sep_indices) >= 2:
                # Context starts after the second </s>.
                # The first </s> ends the question. The second </s> is the start of context block indicator.
                # Actually, XLM-R pair encoding is: <s> A </s> </s> B </s>
                # Indices of '2': [len(A)+1, len(A)+2, len(A)+len(B)+3]
                # Context B starts at sep_indices[1] + 1
                context_start_idx = sep_indices[1] + 1
                context_end_idx = (
                    sep_indices[2] if len(sep_indices) > 2 else len(input_ids) - 1
                )
            else:
                # Fallback: assume whole sequence is searchable (risky but better than crash)
                context_start_idx = 0
                context_end_idx = len(input_ids) - 1

            # Get top-k start and end indices to reduce search space
            top_start_indices = np.argsort(start_logits)[-config.n_best_size :]
            top_end_indices = np.argsort(end_logits)[-config.n_best_size :]

            for s_idx in top_start_indices:
                # Constraint: Start must be within context
                if s_idx < context_start_idx or s_idx >= context_end_idx:
                    continue

                for e_idx in top_end_indices:
                    # Constraint: End must be within context
                    if e_idx < context_start_idx or e_idx >= context_end_idx:
                        continue

                    # Constraint: End must be after Start
                    if s_idx > e_idx:
                        continue

                    # Constraint: Max answer length
                    length = e_idx - s_idx + 1
                    if length > config.max_answer_length:
                        continue

                    # Calculate Gated Score
                    # Score = (Start + End) + Relevance
                    score = start_logits[s_idx] + end_logits[e_idx] + rel_logit

                    if score > best_overall_score:
                        best_overall_score = score

                        # Extract Text
                        start_char = offsets[s_idx][0]
                        end_char = offsets[e_idx][1]

                        # Verify bounds in original string
                        if 0 <= start_char < len(context_text) and end_char <= len(
                            context_text
                        ):
                            best_prediction_string = context_text[start_char:end_char]

        final_preds.append({"id": ex_id, "PredictionString": best_prediction_string})

    # 5. Save Submission
    submission_df = pd.DataFrame(final_preds)
    print(f"Saving submission to {config.submission_path}")
    submission_df.to_csv(config.submission_path, index=False)
    print("Inference complete.")
