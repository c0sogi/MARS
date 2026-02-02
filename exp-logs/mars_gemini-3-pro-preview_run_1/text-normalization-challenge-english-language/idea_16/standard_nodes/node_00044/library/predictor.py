import torch
import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
from library.config import Config
from library.dataset import get_tagger_loader
from library.models import MorphoBiLSTMTagger, CharSeq2Seq


class InferencePipeline:
    """
    Orchestrates the Text Normalization Inference Pipeline:
    1. Feature Extraction (handled by DataLoader)
    2. Tagging (MorphoBiLSTMTagger)
    3. Retrieval (Knowledge Base)
    4. Generation (CharSeq2Seq Fallback)
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        print(f"InferencePipeline initializing on device: {self.device}")

        # 1. Load Vocabularies
        self.word_vocab = self._load_vocab("vocab_words.json")
        self.class_vocab = self._load_vocab("vocab_classes.json")
        self.char_vocab = self._load_vocab("vocab_chars.json")
        self.seq2seq_vocab = self._load_vocab("vocab_seq2seq.json")

        # Create reverse lookups for decoding
        self.id2class = {v: k for k, v in self.class_vocab.items()}
        self.id2char_seq2seq = {v: k for k, v in self.seq2seq_vocab.items()}

        # 2. Load Knowledge Base
        self.kb = self._load_kb()

        # 3. Initialize Models
        self.tagger = self._load_tagger()
        self.seq2seq = self._load_seq2seq()

    def _load_vocab(self, filename):
        path = os.path.join(Config.VOCAB_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary {filename} not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_kb(self):
        """Loads the (token, class) -> normalized_text mapping."""
        kb_path = os.path.join(Config.CACHE_DIR, "knowledge_base.parquet")
        if not os.path.exists(kb_path):
            print(
                "Warning: Knowledge Base not found. Inference will rely solely on models."
            )
            return {}

        print(f"Loading Knowledge Base from {kb_path}...")
        df = pd.read_parquet(kb_path)
        # Create a dictionary for O(1) lookup
        # Key: (raw_token, class_name), Value: normalized_text
        kb_dict = {}
        for before, cls, after in zip(df["before"], df["class"], df["after"]):
            kb_dict[(str(before), str(cls))] = str(after)

        print(f"Knowledge Base loaded with {len(kb_dict)} entries.")
        return kb_dict

    def _load_tagger(self):
        print("Loading Tagger Model...")
        model = MorphoBiLSTMTagger(
            word_vocab_size=len(self.word_vocab),
            class_vocab_size=len(self.class_vocab),
            char_vocab_size=len(self.char_vocab),
        )
        path = os.path.join(Config.CHECKPOINT_DIR, "tagger_best_model.pth")
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            model.load_state_dict(state)
            print(f"Loaded Tagger weights from {path}")
        else:
            print(
                f"Warning: Tagger checkpoint not found at {path}. using random weights."
            )

        model.to(self.device)
        model.eval()
        return model

    def _load_seq2seq(self):
        print("Loading Seq2Seq Model...")
        model = CharSeq2Seq(
            char_vocab_size=len(self.seq2seq_vocab), num_classes=len(self.class_vocab)
        )
        path = os.path.join(Config.CHECKPOINT_DIR, "seq2seq_best_model.pth")
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            model.load_state_dict(state)
            print(f"Loaded Seq2Seq weights from {path}")
        else:
            print(
                f"Warning: Seq2Seq checkpoint not found at {path}. using random weights."
            )

        model.to(self.device)
        model.eval()
        return model

    def run_inference(self):
        """
        Main execution method:
        1. Loads test data.
        2. Predicts classes.
        3. Looks up KB or runs Seq2Seq.
        4. Saves submission.
        """
        print("Starting Inference on Test Set...")

        # Load Raw Test Data (for mapping indices back to tokens and writing submission)
        df_test = pd.read_csv(Config.TEST_DATA_PATH, keep_default_na=False)
        df_test["before"] = df_test["before"].astype(str)

        # Initialize predictions list
        final_predictions = [""] * len(df_test)

        # Buffers for OOV (Out-Of-Vocabulary) items requiring Seq2Seq
        oov_indices = []  # Index in df_test
        oov_tokens = []  # Raw text
        oov_classes = []  # Predicted class ID

        # 1. Get DataLoader
        # This triggers feature extraction (cached) and handles batching
        loader = get_tagger_loader("test", batch_size=Config.BATCH_SIZE, shuffle=False)

        print("Phase 1: Tagging and KB Retrieval...")
        with torch.no_grad():
            for batch in tqdm(loader, desc="Tagging"):
                word_ids = batch["word_ids"].to(self.device)
                char_features = batch["char_features"].to(self.device)
                regex_features = batch["regex_features"].to(self.device)
                lengths = batch["lengths"]

                # Tagger Forward Pass
                logits = self.tagger(word_ids, char_features, regex_features, lengths)
                # [Batch, Seq, NumClasses]

                # Get Class Predictions
                preds = torch.argmax(logits, dim=2).cpu().numpy()

                # Process Batch
                # batch["original_indices"] is a list of arrays (one per sentence)
                batch_orig_indices = batch["original_indices"]

                for i, sent_indices in enumerate(batch_orig_indices):
                    # sent_indices: indices in df_test corresponding to this sentence
                    # sent_preds: predictions for this sentence (truncate padding)
                    sent_preds = preds[i][: len(sent_indices)]

                    for idx_tensor, class_idx in zip(sent_indices, sent_preds):
                        idx = idx_tensor.item()
                        token = df_test.at[idx, "before"]
                        class_name = self.id2class.get(class_idx, "PLAIN")

                        # Strategy: KB Lookup -> Copy (if PLAIN) -> Seq2Seq (if Complex)

                        # 1. KB Lookup
                        kb_key = (token, class_name)
                        if kb_key in self.kb:
                            final_predictions[idx] = self.kb[kb_key]
                        else:
                            # 2. Logic for Unknowns
                            if len(token) == 0:
                                final_predictions[idx] = ""
                            elif class_name in ["PLAIN", "PUNCT"]:
                                # For PLAIN/PUNCT, normalization is usually identity
                                final_predictions[idx] = token
                            else:
                                # 3. Queue for Seq2Seq
                                oov_indices.append(idx)
                                oov_tokens.append(token)
                                oov_classes.append(class_idx)

        print(
            f"Phase 1 Complete. {len(oov_indices)} tokens queued for Seq2Seq generation."
        )

        # Phase 2: Seq2Seq Generation
        if len(oov_indices) > 0:
            self._run_seq2seq_fallback(
                oov_indices, oov_tokens, oov_classes, final_predictions
            )

        # Save Submission
        print("Saving submission...")
        df_test["after"] = final_predictions

        # Format: id,after
        # We explicitly select columns to match submission format
        output_path = Config.SUBMISSION_PATH
        df_test[["id", "after"]].to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")

    def _run_seq2seq_fallback(self, indices, tokens, class_idxs, final_predictions):
        print("Phase 2: Running Seq2Seq Fallback...")

        batch_size = Config.BATCH_SIZE
        num_samples = len(indices)

        # Process in batches to avoid OOM
        for i in range(0, num_samples, batch_size):
            batch_tokens = tokens[i : i + batch_size]
            batch_class_idxs = class_idxs[i : i + batch_size]
            batch_indices = indices[i : i + batch_size]

            # Prepare Inputs
            # Convert tokens to char IDs
            src_ids_list = []
            unk_id = self.seq2seq_vocab.get("<UNK>")
            for t in batch_tokens:
                ids = [self.seq2seq_vocab.get(c, unk_id) for c in str(t)]
                src_ids_list.append(torch.tensor(ids, dtype=torch.long))

            src_lens = torch.tensor([len(s) for s in src_ids_list], dtype=torch.long)
            src_ids = torch.nn.utils.rnn.pad_sequence(
                src_ids_list, batch_first=True, padding_value=0
            ).to(self.device)

            class_ids = torch.tensor(batch_class_idxs, dtype=torch.long).to(self.device)

            # Inference
            with torch.no_grad():
                # Returns [Batch, MaxGenLen] indices
                generated_ids = self.seq2seq(src_ids, src_lens, class_ids, tgt_ids=None)

            # Decode
            generated_ids = generated_ids.cpu().numpy()

            for j, gen_seq in enumerate(generated_ids):
                chars = []
                for char_id in gen_seq:
                    if char_id == 3:  # EOS
                        break
                    if char_id > 3:  # Skip PAD(0), UNK(1), SOS(2), EOS(3)
                        chars.append(self.id2char_seq2seq.get(char_id, ""))

                pred_str = "".join(chars)

                # Update final predictions
                original_idx = batch_indices[j]
                final_predictions[original_idx] = pred_str


def generate_submission():
    """
    Entry point for generating the submission file.
    """
    pipeline = InferencePipeline()
    pipeline.run_inference()
