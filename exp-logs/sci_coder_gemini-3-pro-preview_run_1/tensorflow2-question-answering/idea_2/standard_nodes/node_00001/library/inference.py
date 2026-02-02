import os
import json
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import load_glove_embeddings, tokenize, format_submission
from library.data_loader import build_vocab, get_long_answer_loader
from library.modeling import (
    DEConvNet,
    WindowLogisticRegressor,
    extract_short_answer_inference,
)


def generate_predictions(load_cached_data=True, batch_size=Config.BATCH_SIZE):
    """
    Executes the two-stage inference pipeline on the test set.

    1. Runs DEConvNet to rank Long Answer candidates.
    2. Filters candidates based on Config.LONG_ANSWER_THRESHOLD.
    3. Runs WindowLogisticRegressor on selected Long Answers to extract Short Answers.
    4. Formats and saves the submission file.

    Args:
        load_cached_data (bool): Whether to load vocab/embeddings from cache.
        batch_size (int): Batch size for the Long Answer DataLoader.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # -------------------------------------------------------------------------
    # 1. Load Resources (Metadata, Vocab, Embeddings)
    # -------------------------------------------------------------------------
    print("Loading metadata and resources...")
    if not os.path.exists(Config.TEST_META_FILE):
        raise FileNotFoundError(f"Test metadata not found at {Config.TEST_META_FILE}")

    test_meta = pd.read_parquet(Config.TEST_META_FILE)

    # We use the training vocab/embeddings logic to ensure consistency
    # In inference, we typically expect these to exist from the training phase.
    # We pass the train metadata path logic to build_vocab if cache is missing,
    # but practically we expect cache to be present after training.
    if os.path.exists(Config.TRAIN_META_FILE):
        train_meta_for_vocab = pd.read_parquet(Config.TRAIN_META_FILE)
    else:
        # Fallback if training hasn't run (unlikely in pipeline)
        train_meta_for_vocab = pd.DataFrame()

    vocab = build_vocab(train_meta_for_vocab, load_cached_data=load_cached_data)
    embedding_matrix = load_glove_embeddings(
        vocab, Config.EMBEDDING_DIM, load_cached_data=load_cached_data
    )

    # -------------------------------------------------------------------------
    # 2. Load Models
    # -------------------------------------------------------------------------
    print("Loading models...")

    # Long Answer Model
    la_model = DEConvNet(embedding_matrix).to(device)
    if os.path.exists(Config.LONG_ANSWER_MODEL_PATH):
        la_model.load_state_dict(
            torch.load(Config.LONG_ANSWER_MODEL_PATH, map_location=device)
        )
        print(f"Loaded Long Answer model from {Config.LONG_ANSWER_MODEL_PATH}")
    else:
        print("WARNING: Long Answer model checkpoint not found. Using random weights.")
    la_model.eval()

    # Short Answer Model
    sa_model = WindowLogisticRegressor(input_dim=4).to(device)
    if os.path.exists(Config.SHORT_ANSWER_WEIGHTS_PATH):
        try:
            weights_dict = np.load(
                Config.SHORT_ANSWER_WEIGHTS_PATH, allow_pickle=True
            ).item()
            sa_model.linear.weight.data = torch.tensor(weights_dict["weights"]).to(
                device
            )
            sa_model.linear.bias.data = torch.tensor(weights_dict["bias"]).to(device)
            print(
                f"Loaded Short Answer weights from {Config.SHORT_ANSWER_WEIGHTS_PATH}"
            )
        except Exception as e:
            print(f"Error loading short answer weights: {e}")
    else:
        print("WARNING: Short Answer weights not found. Using random weights.")
    sa_model.eval()

    # -------------------------------------------------------------------------
    # 3. Stage 1: Long Answer Ranking
    # -------------------------------------------------------------------------
    print("Stage 1: Ranking Long Answers...")

    test_loader = get_long_answer_loader(
        test_meta,
        Config.TEST_DATA_FILE,
        vocab,
        split="test",
        batch_size=batch_size,
        shuffle=False,
        load_cached_data=load_cached_data,
    )

    # Dictionary to store the best candidate for each example
    # example_id -> (best_score, best_candidate_index)
    best_candidates = {}

    with torch.no_grad():
        for batch in test_loader:
            q = batch["question"].to(device)
            c = batch["candidate"].to(device)
            e_ids = batch["example_id"]
            c_idxs = batch["candidate_index"]

            scores = la_model(q, c).cpu().numpy()

            for eid, c_idx, score in zip(e_ids, c_idxs, scores):
                c_idx = int(c_idx)
                if eid not in best_candidates:
                    best_candidates[eid] = (-1.0, -1)

                if score > best_candidates[eid][0]:
                    best_candidates[eid] = (float(score), c_idx)

    # -------------------------------------------------------------------------
    # 4. Stage 2: Short Answer Extraction & Prediction Assembly
    # -------------------------------------------------------------------------
    print("Stage 2: Extracting Short Answers...")

    final_predictions = {}

    # Iterate through test metadata to access raw text for the selected candidates
    # We open the file once to avoid overhead
    with open(Config.TEST_DATA_FILE, "rb") as f:
        for _, row in test_meta.iterrows():
            eid = row["example_id"]

            # Initialize default empty prediction
            final_predictions[eid] = {"long": "", "short": ""}

            if eid not in best_candidates:
                continue

            score, cand_idx = best_candidates[eid]

            # Apply Long Answer Threshold (Alpha)
            if score < Config.LONG_ANSWER_THRESHOLD or cand_idx == -1:
                continue

            # Retrieve text data for the specific example
            offset = row["byte_offset"]
            f.seek(offset)
            line = f.readline()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            candidates = data.get("long_answer_candidates", [])
            if cand_idx >= len(candidates):
                continue

            # Construct Long Answer Prediction String
            cand = candidates[cand_idx]
            start_token = cand["start_token"]
            end_token = cand["end_token"]
            long_ans_str = f"{start_token}:{end_token}"

            # Prepare text for Short Answer Extraction
            doc_text = data.get("document_text", "")
            doc_tokens = tokenize(doc_text)
            q_text = data.get("question_text", "")
            q_tokens_set = set(tokenize(q_text.lower()))

            # Run Short Answer Model
            short_ans_str = extract_short_answer_inference(
                doc_tokens, q_tokens_set, start_token, end_token, sa_model
            )

            # Store final predictions
            final_predictions[eid] = {"long": long_ans_str, "short": short_ans_str}

    # -------------------------------------------------------------------------
    # 5. Save Submission
    # -------------------------------------------------------------------------
    print("Formatting and saving submission...")
    format_submission(final_predictions, Config.SUBMISSION_FILE)
    print("Inference completed successfully.")
