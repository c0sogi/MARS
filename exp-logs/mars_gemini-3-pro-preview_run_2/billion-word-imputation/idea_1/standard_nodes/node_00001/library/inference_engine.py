import os
import csv
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from typing import List, Tuple, Dict

from library.config import Config
from library.model_factory import load_model_and_tokenizer


class InferenceDataset(Dataset):
    """
    Dataset to handle tokenization of candidate sentences for inference.
    """

    def __init__(self, texts: List[str], tokenizer, max_len: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize with truncation and padding
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def predict_submission(
    batch_size: int = Config.INFERENCE_BATCH_SIZE, subset_size: int = -1
):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        batch_size: Batch size for inference.
        subset_size: If > 0, only process this many test samples (for debugging).
    """
    # 1. Setup Resources
    device = Config.get_device()

    # Load model from checkpoint if available, else base (fallback)
    model_path = Config.MODEL_SAVE_PATH
    if not os.path.exists(model_path):
        print(f"Warning: Trained model not found at {model_path}. Loading base model.")
        model_path = Config.MODEL_NAME

    print(f"Loading model and tokenizer from {model_path}...")
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval()

    mask_token_id = tokenizer.mask_token_id
    mask_token = tokenizer.mask_token

    # 2. Load Data
    print(f"Loading test data from {Config.TEST_DATA_PATH}...")
    df_test = pd.read_parquet(Config.TEST_DATA_PATH)

    if subset_size > 0:
        df_test = df_test.head(subset_size)

    ids = df_test["id"].tolist()
    sentences = df_test["sentence"].tolist()
    total_samples = len(sentences)

    results = []

    # 3. Processing Loop (Chunked)
    # We process original sentences in chunks to manage the memory footprint
    # of the generated candidates.
    chunk_size = 100

    print(f"Starting inference on {total_samples} samples...")

    for i in range(0, total_samples, chunk_size):
        chunk_ids = ids[i : i + chunk_size]
        chunk_sentences = sentences[i : i + chunk_size]

        candidates = []
        # Map: candidate_idx -> (local_sentence_idx, insertion_index)
        candidate_map = []
        original_word_lists = []

        # Generate candidates for this chunk
        for local_idx, sent in enumerate(chunk_sentences):
            words = sent.split()
            original_word_lists.append(words)
            n_words = len(words)

            # Valid insertion indices: 1 to n_words (exclusive of first/last logic handled by range)
            # Logic: If words=['A', 'B'], len=2. range(1, 2) -> [1].
            # Insert at 1: ['A', '<mask>', 'B']. Correct.
            valid_indices = range(1, n_words)

            for ins_idx in valid_indices:
                # Construct candidate string
                new_words = words[:ins_idx] + [mask_token] + words[ins_idx:]
                cand_str = " ".join(new_words)

                candidates.append(cand_str)
                candidate_map.append((local_idx, ins_idx))

        # If no candidates (e.g. empty sentences), fill with defaults
        if not candidates:
            for local_idx, original_id in enumerate(chunk_ids):
                results.append(
                    {"id": original_id, "sentence": chunk_sentences[local_idx]}
                )
            continue

        # Run Inference on Candidates
        dataset = InferenceDataset(candidates, tokenizer, Config.MAX_SEQ_LEN)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

        # Track best prediction per local_idx
        # best_scores[local_idx] = (score, predicted_word, insertion_index)
        best_scores = {}

        candidate_global_ptr = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with autocast(enabled=Config.USE_FP16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits  # (B, Seq, Vocab)

                # Move to CPU for processing
                batch_input_ids = input_ids.detach().cpu()
                batch_logits = logits.detach().cpu()

                batch_len = input_ids.size(0)

                for b in range(batch_len):
                    local_idx, ins_idx = candidate_map[candidate_global_ptr]
                    candidate_global_ptr += 1

                    # Find mask index
                    # Note: If truncated, mask might be missing.
                    m_indices = (batch_input_ids[b] == mask_token_id).nonzero(
                        as_tuple=True
                    )[0]

                    if len(m_indices) == 0:
                        continue  # Skip invalid candidates

                    m_idx = m_indices[0].item()

                    # Get prediction
                    token_logits = batch_logits[b, m_idx, :]
                    score, pred_id = torch.max(token_logits, dim=0)

                    score = score.item()
                    pred_id = pred_id.item()

                    # Update best
                    if (
                        local_idx not in best_scores
                        or score > best_scores[local_idx][0]
                    ):
                        pred_word = tokenizer.decode(
                            [pred_id], skip_special_tokens=True
                        ).strip()
                        best_scores[local_idx] = (score, pred_word, ins_idx)

        # Reconstruct Sentences
        for local_idx in range(len(chunk_sentences)):
            original_id = chunk_ids[local_idx]

            if local_idx in best_scores:
                _, pred_word, ins_idx = best_scores[local_idx]
                words = original_word_lists[local_idx]
                words.insert(ins_idx, pred_word)
                final_sent = " ".join(words)
            else:
                # Fallback
                final_sent = chunk_sentences[local_idx]

            results.append({"id": original_id, "sentence": final_sent})

        # Logging
        if (i + chunk_size) % 5000 == 0:
            print(f"Processed {i + chunk_size} / {total_samples}...")

    # 4. Save Submission
    print(f"Saving submission to {Config.SUBMISSION_FILE}...")
    df_sub = pd.DataFrame(results)

    # Ensure ID is integer
    df_sub["id"] = df_sub["id"].astype(int)

    # Save with specific quoting to match requirements: id,"sentence"
    df_sub.to_csv(Config.SUBMISSION_FILE, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print("Submission generation completed.")
