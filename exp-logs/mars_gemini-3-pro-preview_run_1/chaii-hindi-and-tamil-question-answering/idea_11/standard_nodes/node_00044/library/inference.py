import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.model import CustomXLMRoberta
from library.data import prepare_data
from library.utils import seed_everything


def get_best_span(
    start_logits, end_logits, relevance_logit, offset_mapping, context, max_len=40
):
    """
    Finds the optimal answer span within a single window based on logits.

    Args:
        start_logits (np.array): Logits for start position (Seq_Len,)
        end_logits (np.array): Logits for end position (Seq_Len,)
        relevance_logit (float): Logit indicating window relevance.
        offset_mapping (np.array): Token to character offsets (Seq_Len, 2).
        context (str): Original context string.
        max_len (int): Maximum allowed length for an answer in tokens.

    Returns:
        score (float): The combined score of the best span.
        text (str): The extracted answer text.
    """
    # Get top N candidates to avoid O(N^2) search over full sequence
    # We take top 20 start and end positions
    n_best = 20
    start_indices = np.argsort(start_logits)[::-1][:n_best]
    end_indices = np.argsort(end_logits)[::-1][:n_best]

    best_score = -float("inf")
    best_answer = ""

    for start_idx in start_indices:
        for end_idx in end_indices:
            # Basic validity checks
            if start_idx > end_idx:
                continue
            if end_idx - start_idx + 1 > max_len:
                continue

            # Check character offsets
            # offset_mapping[i] = [start_char, end_char]
            start_char = offset_mapping[start_idx][0]
            end_char = offset_mapping[end_idx][1]

            # Skip special tokens (often mapped to 0,0) if they result in empty or invalid spans
            # Note: We allow 0,0 if it's a valid start of text, but usually special tokens are <s> etc.
            # A simple check is if the span length in chars is 0, it's likely a special token or empty.
            if start_char == end_char:
                continue

            # Gated Scoring: (Start + End) + Relevance
            # We use the raw logits as per the strategy
            score = start_logits[start_idx] + end_logits[end_idx] + relevance_logit

            if score > best_score:
                best_score = score
                best_answer = context[start_char:end_char]

    return best_score, best_answer


def run_inference(load_cached_data=True):
    """
    Main inference function.

    1. Loads test data (raw metadata + tokenized windows).
    2. Loads the ensemble of models from checkpoints.
    3. Runs inference on all sliding windows.
    4. Aggregates scores and reconstructs answers.
    5. Saves submission.csv.
    """
    seed_everything(42)
    device = Config.DEVICE

    print("Initializing Inference Pipeline...")

    # 1. Load Data
    # We need the raw metadata to map example_idx back to Context and ID
    if not os.path.exists(Config.TEST_META):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_META}")

    test_meta = pd.read_csv(Config.TEST_META)

    # We use the shared data preparation function.
    # This ensures consistency in tokenization and windowing.
    # It returns (train_dataset, test_dataset). We only need test_dataset.
    print("Loading/Processing datasets...")
    _, test_dataset = prepare_data(load_cached_data=load_cached_data)

    data_loader = DataLoader(
        test_dataset,
        batch_size=Config.INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Load Models (Ensemble)
    models = []
    for seed in Config.SEEDS:
        checkpoint_path = os.path.join(Config.OUTPUT_DIR, f"model_seed_{seed}.pth")
        if os.path.exists(checkpoint_path):
            print(f"Loading model checkpoint: {checkpoint_path}")
            model = CustomXLMRoberta(Config.MODEL_NAME)
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            models.append(model)
        else:
            print(f"Warning: Checkpoint for seed {seed} not found. Skipping.")

    if not models:
        print("No models loaded. Aborting inference.")
        return

    print(f"Running inference with {len(models)} models...")

    # 3. Prediction Loop
    # Store best result per example: {example_idx: (score, answer_text)}
    results = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            offset_mapping = batch["offset_mapping"].numpy()
            example_indices = batch["example_idx"].numpy()

            # Accumulators for ensemble averaging
            avg_start_logits = None
            avg_end_logits = None
            avg_relevance_logits = None

            # Forward pass through all models
            for model in models:
                start_logits, end_logits, rel_logits = model(input_ids, attention_mask)

                # Move to CPU numpy
                s = start_logits.cpu().numpy()
                e = end_logits.cpu().numpy()
                r = rel_logits.cpu().numpy()

                if avg_start_logits is None:
                    avg_start_logits = s
                    avg_end_logits = e
                    avg_relevance_logits = r
                else:
                    avg_start_logits += s
                    avg_end_logits += e
                    avg_relevance_logits += r

            # Compute Average
            num_models = len(models)
            avg_start_logits /= num_models
            avg_end_logits /= num_models
            avg_relevance_logits /= num_models

            # Process each sample in the batch
            for i in range(len(example_indices)):
                ex_idx = example_indices[i]

                # Retrieve original context
                # We assume example_idx matches the row index in test_meta
                try:
                    context = test_meta.iloc[ex_idx]["context"]
                except IndexError:
                    # Should not happen if data is consistent
                    continue

                offsets = offset_mapping[i]

                # Extract best span from this window
                score, text = get_best_span(
                    avg_start_logits[i],
                    avg_end_logits[i],
                    avg_relevance_logits[i][0],  # Scalar
                    offsets,
                    context,
                )

                # Update global best for this example ID
                # If we haven't seen this example yet, or if this window provides a higher confidence score
                if ex_idx not in results or score > results[ex_idx][0]:
                    results[ex_idx] = (score, text)

    # 4. Generate Submission File
    print("Generating submission file...")
    submission_data = []

    # Iterate over the original test metadata to ensure we output every ID in order
    for idx, row in test_meta.iterrows():
        sample_id = row["id"]
        prediction_string = ""

        if idx in results:
            # results[idx] is (score, text)
            prediction_string = results[idx][1]

        # Clean the string (remove excessive whitespace)
        prediction_string = prediction_string.strip()

        submission_data.append({"id": sample_id, "PredictionString": prediction_string})

    submission_df = pd.DataFrame(submission_data)

    # Save to CSV
    # quoting=None defaults to minimal quoting, but pandas handles strings correctly.
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved successfully to {Config.SUBMISSION_FILE}")
