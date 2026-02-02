import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader

from library.config import ProjectConfig, TrainingConfig, DataConfig, set_seed
from library.data_utils import (
    build_vocabularies,
    build_knowledge_base,
    load_dataset_raw,
    TaggerDataset,
    collate_fn_tagger,
)
from library.models import MultiGranularityTagger, Seq2SeqNormalizer


class CascadePredictor:
    """
    Hybrid prediction engine combining:
    1. Multi-Granularity Tagger (Bi-LSTM + CharCNN) for semantic classification.
    2. Deterministic Knowledge Base (KB) for O(1) retrieval of known tokens.
    3. Seq2Seq Neural Fallback for OOV normalization.
    """

    def __init__(self, device=None):
        self.device = device if device else torch.device(TrainingConfig.DEVICE)

        print("Initializing CascadePredictor...")

        # 1. Load Vocabularies & Knowledge Base
        # These functions handle caching internally as per library implementation
        self.vocab_words, self.vocab_chars, self.vocab_classes = build_vocabularies(
            load_cached_data=True
        )
        self.kb = build_knowledge_base(load_cached_data=True)

        # 2. Load Tagger Model
        print(f"Loading Tagger from {ProjectConfig.TAGGER_MODEL_PATH}...")
        self.tagger = MultiGranularityTagger(
            len(self.vocab_words), len(self.vocab_chars), len(self.vocab_classes)
        )
        tagger_state = torch.load(
            ProjectConfig.TAGGER_MODEL_PATH, map_location=self.device
        )
        self.tagger.load_state_dict(tagger_state)
        self.tagger.to(self.device)
        self.tagger.eval()

        # 3. Load Seq2Seq Model
        print(f"Loading Seq2Seq from {ProjectConfig.SEQ2SEQ_MODEL_PATH}...")
        self.seq2seq = Seq2SeqNormalizer(len(self.vocab_chars), len(self.vocab_classes))
        seq2seq_state = torch.load(
            ProjectConfig.SEQ2SEQ_MODEL_PATH, map_location=self.device
        )
        self.seq2seq.load_state_dict(seq2seq_state)
        self.seq2seq.to(self.device)
        self.seq2seq.eval()

        # Cache special tokens
        self.sos_idx = self.vocab_chars.stoi[DataConfig.SOS_TOKEN]
        self.eos_idx = self.vocab_chars.stoi[DataConfig.EOS_TOKEN]

    def predict_batch(self, batch):
        """
        Performs inference on a batch of data.

        Args:
            batch (dict): Output from collate_fn_tagger containing:
                          - word_ids, char_ids (tensors)
                          - raw_texts, ids (lists)

        Returns:
            list[dict]: List of {'id': str, 'after': str}
        """
        # Move inputs to device
        word_ids = batch["word_ids"].to(self.device)
        char_ids = batch["char_ids"].to(self.device)
        raw_texts = batch["raw_texts"]
        row_ids = batch["ids"]

        batch_size = len(raw_texts)
        results = [None] * batch_size

        # --- Step 1: Semantic Classification (Tagger) ---
        with torch.no_grad():
            tagger_logits = self.tagger(word_ids, char_ids)
            pred_class_indices = tagger_logits.argmax(dim=1).cpu().numpy()

        # --- Step 2: Hybrid Normalization Strategy ---
        # We segregate tokens into:
        # A. KB Hits / Trivial Copies (Immediate resolution)
        # B. Neural Fallback Candidates (Batch Seq2Seq)

        seq2seq_indices = []
        seq2seq_src_tensors = []
        seq2seq_class_indices = []

        for i in range(batch_size):
            raw_text = raw_texts[i]
            cls_idx = pred_class_indices[i]
            cls_name = self.vocab_classes.lookup_token(cls_idx)

            # Strategy A1: Knowledge Base Lookup
            # Key is (raw_token, predicted_class)
            kb_key = (raw_text, cls_name)

            if kb_key in self.kb:
                results[i] = self.kb[kb_key]

            # Strategy A2: Trivial Classes (PLAIN, PUNCT)
            # If not in KB but class is PLAIN/PUNCT, we assume copy.
            elif cls_name == "PLAIN" or cls_name == "PUNCT":
                results[i] = raw_text

            # Strategy B: Neural Fallback
            else:
                seq2seq_indices.append(i)
                # Use the char_ids from the batch (already padded/tensorized)
                # Note: batch['char_ids'] is (B, MaxLen).
                # We need the specific row.
                seq2seq_src_tensors.append(batch["char_ids"][i])
                seq2seq_class_indices.append(cls_idx)

        # --- Step 3: Run Seq2Seq on Candidates ---
        if seq2seq_indices:
            # Create mini-batch for Seq2Seq
            src_tensor = torch.stack(seq2seq_src_tensors).to(self.device)
            cls_tensor = torch.tensor(seq2seq_class_indices, dtype=torch.long).to(
                self.device
            )

            with torch.no_grad():
                # Predict returns indices (N, MaxLen)
                pred_seqs = self.seq2seq.predict(
                    src_tensor,
                    cls_tensor,
                    max_len=DataConfig.MAX_TOKEN_LEN,
                    sos_idx=self.sos_idx,
                    eos_idx=self.eos_idx,
                )

            pred_seqs_np = pred_seqs.cpu().numpy()

            # Decode and assign results
            for idx_in_batch, seq_indices in zip(seq2seq_indices, pred_seqs_np):
                chars = []
                for char_idx in seq_indices:
                    if char_idx == self.eos_idx:
                        break
                    token = self.vocab_chars.lookup_token(char_idx)
                    if token:
                        chars.append(token)

                normalized_text = "".join(chars)
                results[idx_in_batch] = normalized_text

        # --- Step 4: Format Output ---
        final_output = []
        for i in range(batch_size):
            # Fallback for safety (should not happen if logic is complete)
            if results[i] is None:
                results[i] = raw_texts[i]

            final_output.append({"id": row_ids[i], "after": results[i]})

        return final_output


def generate_submission_file(batch_size=256):
    """
    Generates the submission file for the test set.

    Args:
        batch_size (int): Batch size for inference.
    """
    print("\n=== Generating Submission ===")
    set_seed(TrainingConfig.SEED)
    device = torch.device(TrainingConfig.DEVICE)

    # Initialize Engine
    predictor = CascadePredictor(device=device)

    # Load Test Data
    print("Loading test data...")
    df_test = load_dataset_raw("test")

    # Prepare DataLoader
    # We use TaggerDataset as it provides the necessary input structure (words+chars)
    test_dataset = TaggerDataset(
        df_test,
        predictor.vocab_words,
        predictor.vocab_chars,
        predictor.vocab_classes,
        is_test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_tagger,
        num_workers=DataConfig.NUM_WORKERS,
    )

    all_predictions = []

    print(f"Starting inference on {len(df_test)} tokens...")

    # Inference Loop
    for batch_idx, batch in enumerate(test_loader):
        batch_preds = predictor.predict_batch(batch)
        all_predictions.extend(batch_preds)

        if (batch_idx + 1) % 100 == 0:
            print(f"Processed {batch_idx + 1} batches...")

    # Save to CSV
    df_sub = pd.DataFrame(all_predictions)

    # Ensure correct column order per submission format
    df_sub = df_sub[["id", "after"]]

    output_path = ProjectConfig.SUBMISSION_PATH
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Total predictions: {len(df_sub)}")
