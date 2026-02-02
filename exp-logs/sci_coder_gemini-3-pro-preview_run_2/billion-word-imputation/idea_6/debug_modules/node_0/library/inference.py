import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from library.config import Config
from library.utils import get_logger, escape_sentence_for_csv
from library.models import ModelFactory
from library.data import TestDataset

logger = get_logger("inference")


class BeamPipeline:
    """
    Implements the Probabilistic Beam-Search Cascade for inference.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.beam_k = Config.BEAM_K

        logger.info(f"Initializing BeamPipeline on {self.device} with K={self.beam_k}")

        # 1. Load Models
        logger.info("Loading Locator model...")
        self.locator = ModelFactory.get_locator_model()
        self.locator.load_state_dict(
            torch.load(Config.LOCATOR_MODEL_PATH, map_location=self.device)
        )
        self.locator.to(self.device)
        self.locator.eval()

        logger.info("Loading In-Filler model...")
        self.infiller = ModelFactory.get_infiller_model()
        self.infiller.load_state_dict(
            torch.load(Config.INFILLER_MODEL_PATH, map_location=self.device)
        )
        self.infiller.to(self.device)
        self.infiller.eval()

        # 2. Load Tokenizers
        # We need fast tokenizers to get offset mappings efficiently
        logger.info("Loading tokenizers...")
        try:
            self.loc_tokenizer = AutoTokenizer.from_pretrained(
                Config.LOCATOR_MODEL_NAME, use_fast=True
            )
        except Exception as e:
            logger.warning(f"Fast tokenizer for Locator not available: {e}")
            self.loc_tokenizer = AutoTokenizer.from_pretrained(
                Config.LOCATOR_MODEL_NAME, use_fast=False
            )

        try:
            self.inf_tokenizer = AutoTokenizer.from_pretrained(
                Config.INFILLER_MODEL_NAME, use_fast=True
            )
        except Exception as e:
            logger.warning(f"Fast tokenizer for Infiller not available: {e}")
            self.inf_tokenizer = AutoTokenizer.from_pretrained(
                Config.INFILLER_MODEL_NAME, use_fast=False
            )

        self.mask_token = self.inf_tokenizer.mask_token
        self.mask_token_id = self.inf_tokenizer.mask_token_id

    def predict(self, test_loader):
        """
        Runs the cascade inference on the test loader.
        """
        results = []

        logger.info("Starting inference loop...")

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                # We use raw text to handle offset mapping and reconstruction precisely
                raw_texts = batch["text"]
                row_ids = batch["id"]

                # -------------------------------------------------------
                # STAGE 1: LOCATOR (Structure)
                # -------------------------------------------------------
                # Re-tokenize to ensure we have access to offset_mapping
                loc_encodings = self.loc_tokenizer(
                    raw_texts,
                    padding=True,
                    truncation=True,
                    max_length=Config.MAX_LEN,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                )

                loc_input_ids = loc_encodings["input_ids"].to(self.device)
                loc_att_mask = loc_encodings["attention_mask"].to(self.device)
                offset_mapping = loc_encodings["offset_mapping"].cpu().numpy()

                # Forward pass
                loc_outputs = self.locator(
                    input_ids=loc_input_ids, attention_mask=loc_att_mask
                )
                loc_logits = loc_outputs.logits.squeeze(-1)  # (batch, seq_len)
                loc_probs = torch.sigmoid(loc_logits)

                # Mask out special tokens (CLS, SEP, PAD) to prevent invalid insertions
                special_tokens_mask = torch.tensor(
                    [
                        self.loc_tokenizer.get_special_tokens_mask(
                            val, already_has_special_tokens=True
                        )
                        for val in loc_input_ids.cpu().tolist()
                    ],
                    device=self.device,
                )
                valid_mask = (loc_att_mask == 1) & (special_tokens_mask == 0)

                # Apply mask (zero out invalid probs)
                loc_probs = loc_probs * valid_mask.float()

                # Get Top-K candidates
                # topk_probs: (batch, k), topk_indices: (batch, k)
                topk_probs, topk_indices = torch.topk(loc_probs, k=self.beam_k, dim=1)

                topk_probs = topk_probs.cpu().numpy()
                topk_indices = topk_indices.cpu().numpy()

                # -------------------------------------------------------
                # STAGE 2: HYPOTHESIS EXPANSION
                # -------------------------------------------------------
                hypotheses = []

                for b_idx, text in enumerate(raw_texts):
                    offsets = offset_mapping[b_idx]

                    for k in range(self.beam_k):
                        token_idx = topk_indices[b_idx][k]
                        prob_loc = topk_probs[b_idx][k]

                        # Determine insertion character position
                        # offsets[token_idx] = (start_char, end_char)
                        # We insert AFTER the token, so we use end_char.
                        try:
                            # Safety check for bounds
                            if token_idx < len(offsets):
                                char_end = offsets[token_idx][1]
                            else:
                                char_end = len(text)
                        except IndexError:
                            char_end = len(text)

                        # If char_end is 0 (special token case), handle gracefully
                        if char_end == 0 and len(text) > 0:
                            # Should be handled by valid_mask, but as fallback
                            char_end = len(text)

                        # Construct masked sentence
                        # We insert " <mask>" to ensure token separation
                        prefix = text[:char_end]
                        suffix = text[char_end:]
                        masked_text = f"{prefix} {self.mask_token}{suffix}"

                        hypotheses.append(
                            {
                                "original_id": row_ids[b_idx].item(),
                                "original_text": text,
                                "masked_text": masked_text,
                                "prob_loc": prob_loc,
                                "insert_pos": char_end,
                            }
                        )

                # -------------------------------------------------------
                # STAGE 3: IN-FILLER (Semantics)
                # -------------------------------------------------------
                # Process hypotheses in chunks to manage memory
                inf_texts = [h["masked_text"] for h in hypotheses]
                inf_predictions = []

                chunk_size = Config.INFILLER_BATCH_SIZE
                for i in range(0, len(inf_texts), chunk_size):
                    chunk_texts = inf_texts[i : i + chunk_size]

                    inf_encodings = self.inf_tokenizer(
                        chunk_texts,
                        padding=True,
                        truncation=True,
                        max_length=Config.MAX_LEN,
                        return_tensors="pt",
                    )

                    inf_input_ids = inf_encodings["input_ids"].to(self.device)
                    inf_att_mask = inf_encodings["attention_mask"].to(self.device)

                    inf_outputs = self.infiller(
                        input_ids=inf_input_ids, attention_mask=inf_att_mask
                    )
                    inf_logits = inf_outputs.logits  # (chunk, seq, vocab)

                    # Identify mask positions
                    mask_mask = inf_input_ids == self.mask_token_id
                    has_mask = mask_mask.any(dim=1)

                    # Storage for this chunk
                    chunk_probs = torch.zeros(len(chunk_texts), device=self.device)
                    chunk_word_ids = torch.zeros(
                        len(chunk_texts), dtype=torch.long, device=self.device
                    )

                    if has_mask.any():
                        # Get indices of valid rows
                        valid_indices = torch.nonzero(has_mask).squeeze()
                        if valid_indices.ndim == 0:
                            valid_indices = valid_indices.unsqueeze(0)

                        for v_idx in valid_indices:
                            # Find the column index of the mask
                            # We take the first mask if multiple (though unlikely with our constr)
                            col_idx = (
                                (inf_input_ids[v_idx] == self.mask_token_id)
                                .nonzero(as_tuple=True)[0][0]
                                .item()
                            )

                            # Get distribution over vocab
                            logits_vec = inf_logits[v_idx, col_idx, :]
                            probs_vec = torch.softmax(logits_vec, dim=0)

                            # Get best word
                            max_prob, max_id = torch.max(probs_vec, dim=0)

                            chunk_probs[v_idx] = max_prob
                            chunk_word_ids[v_idx] = max_id

                    inf_predictions.extend(
                        zip(chunk_probs.cpu().tolist(), chunk_word_ids.cpu().tolist())
                    )

                # -------------------------------------------------------
                # STAGE 4: JOINT RANKING & RECONSTRUCTION
                # -------------------------------------------------------
                # 1. Assign semantic scores back to hypotheses
                for h, (prob_word, word_id) in zip(hypotheses, inf_predictions):
                    h["prob_word"] = prob_word
                    h["word_id"] = word_id
                    # Joint Score
                    h["score"] = h["prob_loc"] * prob_word

                # 2. Select best hypothesis per original ID
                best_candidates = {}
                for h in hypotheses:
                    oid = h["original_id"]
                    if oid not in best_candidates:
                        best_candidates[oid] = h
                    else:
                        if h["score"] > best_candidates[oid]["score"]:
                            best_candidates[oid] = h

                # 3. Reconstruct final strings
                for oid in row_ids.tolist():
                    best = best_candidates[oid]
                    word_id = best["word_id"]

                    # Decode word
                    predicted_word = self.inf_tokenizer.decode(
                        [word_id], skip_special_tokens=True
                    ).strip()

                    # Insert into original text
                    orig = best["original_text"]
                    pos = best["insert_pos"]

                    # Construct: prefix + " " + word + suffix
                    # We add a space because we are inserting a missing word
                    final_sentence = f"{orig[:pos]} {predicted_word}{orig[pos:]}"

                    # Basic cleanup of potential double spaces introduced
                    # (e.g. if original was "Hello ." and we insert "world", we get "Hello world .")
                    # (e.g. if original was "Hello world" and we insert "new", we get "Hello new world")
                    # The simple f-string usually works given the task setup.
                    # We can normalize spaces just in case.
                    final_sentence = " ".join(final_sentence.split())

                    results.append((oid, final_sentence))

        return results

    def generate_submission(self, results):
        """
        Writes the results to the submission CSV file.
        """
        logger.info("Formatting submission...")
        df = pd.DataFrame(results, columns=["id", "sentence"])

        # Apply escaping
        df["sentence"] = df["sentence"].apply(escape_sentence_for_csv)

        output_path = Config.SUBMISSION_PATH
        logger.info(f"Saving submission to {output_path}")

        # Write manually to ensure strict CSV compliance with quoting
        with open(output_path, "w", encoding="utf-8") as f:
            f.write('id,"sentence"\n')
            for _, row in df.iterrows():
                f.write(f'{row["id"]},{row["sentence"]}\n')


def run_inference():
    """
    Main entry point for inference.
    """
    # 1. Load Data
    logger.info(f"Loading test metadata from {Config.TEST_METADATA_PATH}")
    df_test = pd.read_parquet(Config.TEST_METADATA_PATH)

    # 2. Create DataLoader
    # We pass a dummy tokenizer because TestDataset requires one,
    # but we will re-tokenize inside the pipeline to handle offsets correctly.
    dummy_tokenizer = AutoTokenizer.from_pretrained(Config.LOCATOR_MODEL_NAME)
    test_dataset = TestDataset(df_test, dummy_tokenizer)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.LOCATOR_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Run Pipeline
    pipeline = BeamPipeline()
    results = pipeline.predict(test_loader)

    # 4. Save
    pipeline.generate_submission(results)
    logger.info("Inference process finished.")
