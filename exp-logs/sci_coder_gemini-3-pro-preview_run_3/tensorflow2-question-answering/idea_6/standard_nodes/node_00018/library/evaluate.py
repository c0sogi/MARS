import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from library.config import PathConfig, ModelConfig, TrainingConfig
from library.utils import setup_logger, parse_html, HTML_TAGS
from library.models import DualEncoderRanker, SimilarityProjectionReader

# Initialize logger
logger = setup_logger("evaluate")


def load_models(device):
    """
    Loads the trained Ranker and Reader models from checkpoints.
    """
    logger.info("Loading models...")

    # Load Ranker
    ranker = DualEncoderRanker()
    if os.path.exists(PathConfig.RANKER_MODEL_PATH):
        ranker.load_state_dict(
            torch.load(PathConfig.RANKER_MODEL_PATH, map_location=device)
        )
        logger.info(f"Ranker loaded from {PathConfig.RANKER_MODEL_PATH}")
    else:
        logger.warning(
            f"Ranker checkpoint not found at {PathConfig.RANKER_MODEL_PATH}. Using untrained model."
        )
    ranker.to(device)
    ranker.eval()

    # Load Reader
    reader = SimilarityProjectionReader()
    if os.path.exists(PathConfig.READER_MODEL_PATH):
        reader.load_state_dict(
            torch.load(PathConfig.READER_MODEL_PATH, map_location=device)
        )
        logger.info(f"Reader loaded from {PathConfig.READER_MODEL_PATH}")
    else:
        logger.warning(
            f"Reader checkpoint not found at {PathConfig.READER_MODEL_PATH}. Using untrained model."
        )
    reader.to(device)
    reader.eval()

    return ranker, reader


def get_original_token_indices(doc_tokens, candidate_start, candidate_end):
    """
    Maps indices from the cleaned (tag-stripped) text back to the original document token indices.
    Returns a list where list[i] is the original index of the i-th token in the cleaned text.
    """
    raw_slice = doc_tokens[candidate_start:candidate_end]
    mapping = []

    for i, token in enumerate(raw_slice):
        if token not in HTML_TAGS:
            mapping.append(candidate_start + i)

    return mapping


def predict_submission(subset_size=None):
    """
    Main inference function. Generates predictions for the test set and saves submission.csv.
    """
    device = torch.device(TrainingConfig.DEVICE)
    ranker, reader = load_models(device)
    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.MODEL_NAME)

    # Load Test Metadata
    if not os.path.exists(PathConfig.TEST_METADATA):
        logger.error(f"Test metadata not found at {PathConfig.TEST_METADATA}")
        return

    test_metadata = pd.read_csv(PathConfig.TEST_METADATA)
    if subset_size:
        test_metadata = test_metadata.head(subset_size)

    logger.info(f"Starting inference on {len(test_metadata)} examples...")

    results = []

    # Iterate through test examples
    for _, row in tqdm(
        test_metadata.iterrows(), total=len(test_metadata), desc="Inference"
    ):
        example_id = str(row["example_id"])
        offset = row["byte_offset"]

        # Read raw JSON line
        with open(PathConfig.TEST_FILE, "rb") as f:
            f.seek(offset)
            line = f.readline()
            if not line:
                continue
            record = json.loads(line.decode("utf-8"))

        question = record["question_text"]
        doc_text = record["document_text"]
        doc_tokens = doc_text.split()

        # 1. Candidate Generation
        candidates = parse_html(doc_text)

        if not candidates:
            # No valid text blocks found
            results.append({"example_id": f"{example_id}_long", "PredictionString": ""})
            results.append(
                {"example_id": f"{example_id}_short", "PredictionString": ""}
            )
            continue

        # 2. Ranking
        # Prepare batch for ranker
        cand_texts = [c["text"] for c in candidates]

        # Tokenize Question
        q_inputs = tokenizer(
            [question],
            max_length=ModelConfig.MAX_Q_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # Tokenize Candidates (Batch processing)
        # We process candidates in chunks if there are too many to avoid OOM
        cand_embeddings = []
        chunk_size = 32

        with torch.no_grad():
            q_emb = ranker(
                q_inputs["input_ids"], q_inputs["attention_mask"]
            )  # (1, Hidden)

            for i in range(0, len(cand_texts), chunk_size):
                batch_texts = cand_texts[i : i + chunk_size]
                c_inputs = tokenizer(
                    batch_texts,
                    max_length=ModelConfig.MAX_CTX_LEN,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(device)

                c_emb = ranker(c_inputs["input_ids"], c_inputs["attention_mask"])
                cand_embeddings.append(c_emb)

            if cand_embeddings:
                cand_embeddings = torch.cat(
                    cand_embeddings, dim=0
                )  # (Num_Cand, Hidden)

                # Compute Cosine Similarity
                # Normalize embeddings
                q_emb_norm = torch.nn.functional.normalize(q_emb, p=2, dim=1)
                c_emb_norm = torch.nn.functional.normalize(cand_embeddings, p=2, dim=1)

                scores = torch.mm(q_emb_norm, c_emb_norm.transpose(0, 1)).squeeze(
                    0
                )  # (Num_Cand)

                best_score, best_idx = torch.max(scores, dim=0)
                best_score = best_score.item()
                best_idx = best_idx.item()

                best_candidate = candidates[best_idx]
            else:
                best_score = -1.0
                best_candidate = None

        # 3. Decision Logic & Reading
        long_pred_str = ""
        short_pred_str = ""

        if best_candidate and best_score >= ModelConfig.RANKER_THRESHOLD:
            # Valid Long Answer found
            long_pred_str = (
                f"{best_candidate['start_token']}:{best_candidate['end_token']}"
            )

            # Prepare Reader Input
            context_text = best_candidate["text"]

            # Tokenize for Reader
            # We use stride to handle long contexts, but for simplicity in inference
            # we take the first window or the one with best ranker match.
            # Given ranker scored this text high, we assume the answer is within the truncation limit.
            reader_inputs = tokenizer(
                question,
                context_text,
                truncation="only_second",
                max_length=ModelConfig.MAX_CTX_LEN,
                return_offsets_mapping=True,
                return_token_type_ids=True,
                padding="max_length",
                return_tensors="pt",
            )

            # Move to device
            r_input_ids = reader_inputs["input_ids"].to(device)
            r_mask = reader_inputs["attention_mask"].to(device)
            r_token_type = reader_inputs["token_type_ids"].to(device)
            offset_mapping = (
                reader_inputs["offset_mapping"][0].cpu().numpy()
            )  # (Seq, 2)

            with torch.no_grad():
                start_logits, end_logits = reader(
                    input_ids=r_input_ids,
                    attention_mask=r_mask,
                    token_type_ids=r_token_type,
                )

            # Decode Span
            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()[0]
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()[0]

            start_idx = np.argmax(start_probs)
            end_idx = np.argmax(end_probs)

            confidence = start_probs[start_idx] * end_probs[end_idx]

            # Validate Span
            if (
                start_idx <= end_idx
                and confidence >= ModelConfig.SHORT_ANSWER_THRESHOLD
            ):
                # Map BERT tokens back to original document tokens

                # 1. Get char offsets in cleaned context
                # offset_mapping gives (start_char, end_char) in the combined sequence
                # We need to filter for context part (token_type_ids == 1)
                token_types = r_token_type.cpu().numpy()[0]

                if token_types[start_idx] == 1 and token_types[end_idx] == 1:
                    char_start = offset_mapping[start_idx][0]
                    char_end = offset_mapping[end_idx][1]

                    # 2. Map char offsets to whitespace tokens in cleaned text
                    clean_tokens = context_text.split()

                    # Build a map of char index -> clean token index
                    # This is linear scan but safe
                    current_char = 0
                    clean_token_start_idx = -1
                    clean_token_end_idx = -1

                    for i, token in enumerate(clean_tokens):
                        token_len = len(token)
                        # Token spans from current_char to current_char + token_len
                        # Check overlap with answer span

                        # If the answer starts within this token
                        if (
                            current_char <= char_start < current_char + token_len + 1
                        ):  # +1 for space
                            clean_token_start_idx = i

                        # If the answer ends within this token
                        if (
                            current_char <= char_end <= current_char + token_len + 1
                        ):  # approximate loose check
                            clean_token_end_idx = i

                        # If char_end is exactly at the end of token
                        if current_char + token_len == char_end:
                            clean_token_end_idx = i

                        current_char += token_len + 1  # +1 for space

                    # Fallback if indices not found (e.g. subword alignment issues)
                    if clean_token_start_idx == -1:
                        clean_token_start_idx = 0
                    if clean_token_end_idx == -1:
                        clean_token_end_idx = len(clean_tokens) - 1

                    # 3. Map clean token indices to original document indices
                    original_indices = get_original_token_indices(
                        doc_tokens,
                        best_candidate["start_token"],
                        best_candidate["end_token"],
                    )

                    if clean_token_start_idx < len(
                        original_indices
                    ) and clean_token_end_idx < len(original_indices):
                        final_start = original_indices[clean_token_start_idx]
                        final_end = (
                            original_indices[clean_token_end_idx] + 1
                        )  # Exclusive end for output?
                        # NQ format usually expects start:end. Example 6:18.
                        # If 6:18 means tokens 6 to 17 (inclusive), then python slice style is correct.

                        short_pred_str = f"{final_start}:{final_end}"

        results.append(
            {"example_id": f"{example_id}_long", "PredictionString": long_pred_str}
        )
        results.append(
            {"example_id": f"{example_id}_short", "PredictionString": short_pred_str}
        )

    # Save Submission
    PathConfig.ensure_dirs()
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(PathConfig.SUBMISSION_FILE, index=False)
    logger.info(f"Submission saved to {PathConfig.SUBMISSION_FILE}")
