import torch
import pandas as pd
import numpy as np
import os
import csv
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import List, Tuple, Optional

from library.config import Config
from library.utils import get_logger
from library.vocab_manager import build_vocabs, Vocab
from library.feature_engineering import FeatureEngineer
from library.knowledge_base import KnowledgeBase
from library.models import GatedBiLSTMTagger, Seq2SeqFallback
from library.datasets import TaggerDataset

logger = get_logger("inference")


class Seq2SeqInferenceDataset(Dataset):
    """
    Dataset wrapper for batching inputs to the Seq2Seq Fallback model during inference.
    """

    def __init__(
        self, tokens: List[str], class_ids: List[int], char_vocab: Vocab, max_len: int
    ):
        self.tokens = tokens
        self.class_ids = class_ids
        self.char_vocab = char_vocab
        self.max_len = max_len
        self.pad_id = char_vocab["<pad>"]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token = self.tokens[idx]
        cls_id = self.class_ids[idx]

        # Encode characters
        c_ids = [self.char_vocab[c] for c in token]

        # Pad or Truncate
        if len(c_ids) > self.max_len:
            c_ids = c_ids[: self.max_len]
        else:
            c_ids = c_ids + [self.pad_id] * (self.max_len - len(c_ids))

        return torch.tensor(c_ids, dtype=torch.long), torch.tensor(
            cls_id, dtype=torch.long
        )


class Predictor:
    """
    Orchestrates the end-to-end prediction pipeline:
    Feature Extraction -> Tagging -> KB Retrieval -> Seq2Seq Fallback -> Submission
    """

    def __init__(self):
        self.device = Config.DEVICE
        logger.info(f"Initializing Predictor on {self.device}...")

        # 1. Load Vocabularies
        self.word_vocab, self.char_vocab, self.class_vocab, self.bpe_tokenizer = (
            build_vocabs(load_cached_data=True)
        )

        # 2. Load Feature Engineer & Priors
        self.fe = FeatureEngineer()
        self.priors_df = self.fe.build_or_load_priors(
            self.class_vocab, load_cached_data=True
        )

        # 3. Load Knowledge Base
        self.kb = KnowledgeBase()
        self.kb.build(load_cached_data=True)

        # 4. Load Tagger Model
        logger.info("Loading Tagger Model...")
        self.tagger = GatedBiLSTMTagger(
            word_vocab_size=len(self.word_vocab),
            bpe_vocab_size=len(self.bpe_tokenizer),
            char_vocab_size=len(self.char_vocab),
            class_vocab_size=len(self.class_vocab),
            num_regex_feats=Config.NUM_REGEX_FEATURES,
            num_classes=len(self.class_vocab),
        ).to(self.device)

        if os.path.exists(Config.TAGGER_MODEL_PATH):
            self.tagger.load_state_dict(
                torch.load(Config.TAGGER_MODEL_PATH, map_location=self.device)
            )
            logger.info(f"Tagger weights loaded from {Config.TAGGER_MODEL_PATH}")
        else:
            logger.warning("Tagger checkpoint not found! Predictions will be random.")
        self.tagger.eval()

        # 5. Load Seq2Seq Model
        logger.info("Loading Seq2Seq Model...")
        self.seq2seq = Seq2SeqFallback(
            char_vocab_size=len(self.char_vocab),
            class_vocab_size=len(self.class_vocab),
            sos_idx=self.char_vocab["<sos>"],
            eos_idx=self.char_vocab["<eos>"],
            max_seq_len=Config.MAX_SEQ_LEN,
        ).to(self.device)

        if os.path.exists(Config.SEQ2SEQ_MODEL_PATH):
            self.seq2seq.load_state_dict(
                torch.load(Config.SEQ2SEQ_MODEL_PATH, map_location=self.device)
            )
            logger.info(f"Seq2Seq weights loaded from {Config.SEQ2SEQ_MODEL_PATH}")
        else:
            logger.warning("Seq2Seq checkpoint not found! Fallback will be random.")
        self.seq2seq.eval()

    def generate_submission(self, debug: bool = False):
        """
        Generates predictions for the test set and saves them to submission.csv.
        """
        logger.info("Starting submission generation...")

        # --- Step 1: Prepare Test Data ---
        # We use TaggerDataset to handle feature extraction efficiently
        test_dataset = TaggerDataset(
            data_path=Config.TEST_FILE,
            word_vocab=self.word_vocab,
            char_vocab=self.char_vocab,
            class_vocab=self.class_vocab,
            bpe_tokenizer=self.bpe_tokenizer,
            feature_engineer=self.fe,
            priors_df=self.priors_df,
            split="test",
            load_cached_data=True,
            debug=debug,
        )

        dataloader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE * 2,  # Inference is lighter on VRAM
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # --- Step 2: Tagger Inference ---
        logger.info("Running Tagger Inference...")
        all_preds = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Tagging"):
                word_ids = batch["word_ids"].to(self.device).unsqueeze(1)
                bpe_ids = batch["bpe_ids"].to(self.device).unsqueeze(1)
                char_ids = batch["char_ids"].to(self.device).unsqueeze(1)
                regex_feats = batch["regex_feats"].to(self.device).unsqueeze(1)
                prior_feats = batch["prior_feats"].to(self.device).unsqueeze(1)

                # Forward pass
                logits = self.tagger(
                    word_ids, bpe_ids, char_ids, regex_feats, prior_feats
                )
                logits = logits.squeeze(1)

                # Get class indices
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                all_preds.extend(preds)

        # --- Step 3: Hybrid Normalization Logic ---
        logger.info("Applying Hybrid Normalization (KB + Fallback)...")

        # Load raw test data to get tokens and IDs
        df_test = pd.read_csv(Config.TEST_FILE, dtype=str, keep_default_na=False)
        if debug:
            df_test = df_test.head(Config.DEBUG_SIZE)

        raw_tokens_list = df_test["before"].astype(str).tolist()
        row_ids_list = df_test["id"].tolist()

        assert len(all_preds) == len(
            raw_tokens_list
        ), "Mismatch between predictions and test set size"

        final_results = []  # List of (id, after)

        # Containers for Seq2Seq batch processing
        fallback_indices = []  # Indices in final_results to update later
        fallback_inputs = []  # (token, class_id) tuples

        for i, (token, pred_cls_idx, row_id) in enumerate(
            zip(raw_tokens_list, all_preds, row_ids_list)
        ):
            pred_cls_str = self.class_vocab.lookup_token(pred_cls_idx)

            # A. Knowledge Base Lookup (Deterministic Memory)
            kb_result = self.kb.query(token, pred_cls_str)

            if kb_result is not None:
                final_results.append((row_id, kb_result))
            else:
                # B. Heuristic Copy (Identity)
                # PLAIN and PUNCT rarely change. If not in KB, assume copy.
                if pred_cls_str in ["PLAIN", "PUNCT"]:
                    final_results.append((row_id, token))
                else:
                    # C. Neural Fallback (Generative)
                    # Placeholder, will be filled by Seq2Seq
                    final_results.append((row_id, None))
                    fallback_indices.append(i)
                    fallback_inputs.append((token, pred_cls_idx))

        # --- Step 4: Run Seq2Seq on Fallback Items ---
        if fallback_inputs:
            logger.info(f"Running Seq2Seq Fallback on {len(fallback_inputs)} tokens...")

            fb_tokens, fb_class_ids = zip(*fallback_inputs)

            # Use MAX_SEQ_LEN for input encoding to handle long tokens
            fb_dataset = Seq2SeqInferenceDataset(
                fb_tokens, fb_class_ids, self.char_vocab, Config.MAX_SEQ_LEN
            )

            fb_loader = DataLoader(
                fb_dataset,
                batch_size=Config.BATCH_SIZE * 2,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )

            generated_texts = []

            with torch.no_grad():
                for src_ids, class_ids in tqdm(fb_loader, desc="Seq2Seq"):
                    src_ids = src_ids.to(self.device)
                    class_ids = class_ids.to(self.device)

                    # Generate: (Batch, Max_Len)
                    output_ids = self.seq2seq.generate(src_ids, class_ids)
                    output_ids = output_ids.cpu().numpy()

                    # Decode IDs to String
                    for row in output_ids:
                        chars = []
                        for idx in row:
                            if idx == self.char_vocab["<eos>"]:
                                break
                            if idx == self.char_vocab["<sos>"]:
                                continue
                            if idx == self.char_vocab["<pad>"]:
                                continue
                            try:
                                chars.append(self.char_vocab.lookup_token(idx))
                            except:
                                pass
                        generated_texts.append("".join(chars))

            # Update final_results with generated text
            for idx, text in zip(fallback_indices, generated_texts):
                row_id = final_results[idx][0]
                final_results[idx] = (row_id, text)

        # --- Step 5: Save Submission ---
        logger.info("Saving submission...")
        submission_df = pd.DataFrame(final_results, columns=["id", "after"])

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

        # Save with quoting to handle special characters (e.g., "3" -> "three")
        submission_df.to_csv(
            Config.SUBMISSION_FILE, index=False, quoting=csv.QUOTE_NONNUMERIC
        )
        logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")
