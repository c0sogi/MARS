import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from library.config import Config
from library.utils import set_seed, get_device, save_submission
from library.data_processing import prepare_data, TaggerDataset
from library.models_tagger import QuadHybridBiLSTM
from library.models_seq2seq import CharTransformer


class InferenceSeq2SeqDataset(Dataset):
    """
    Lightweight dataset for Seq2Seq inference on specific tokens.
    """

    def __init__(self, tokens, classes, vocab_chars, vocab_classes):
        self.tokens = tokens
        self.classes = classes
        self.vocab_chars = vocab_chars
        self.vocab_classes = vocab_classes
        self.config = Config()
        self.max_len = self.config.MAX_TOKEN_CHAR_LEN
        self.pad_id = self.vocab_chars.get_id("<PAD>")
        self.unk_id = self.vocab_chars.get_id("<UNK>")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token = str(self.tokens[idx])
        cls_name = str(self.classes[idx])

        # Source Chars
        src_indices = [self.vocab_chars.get_id(c, self.unk_id) for c in token]
        src_indices = src_indices[: self.max_len]

        # Padding
        src_pad_len = self.max_len - len(src_indices)
        src_ids = src_indices + [self.pad_id] * src_pad_len

        # Class ID
        class_id = self.vocab_classes.get_id(cls_name, 0)

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "class_id": torch.tensor(class_id, dtype=torch.long),
        }


class NormalizationPipeline:
    """
    End-to-end inference pipeline for Text Normalization.
    """

    def __init__(self):
        self.config = Config()
        self.device = get_device()
        set_seed(self.config.SEED)

        # Placeholders for resources
        self.vocab_words = None
        self.vocab_chars = None
        self.vocab_classes = None
        self.bpe_tokenizer = None
        self.kb_map = None

        # Models
        self.tagger_model = None
        self.seq2seq_model = None

    def load_resources(self, load_cached_data=True):
        """
        Loads data artifacts and models.
        """
        print("Loading resources and data artifacts...")
        # 1. Load Data Artifacts via prepare_data
        # This handles vocabs, BPE, KB, and test data preprocessing/grouping
        artifacts = prepare_data(load_cached_data=load_cached_data)

        self.vocab_words = artifacts["vocab_words"]
        self.vocab_chars = artifacts["vocab_chars"]
        self.vocab_classes = artifacts["vocab_classes"]
        self.bpe_tokenizer = artifacts["bpe_tokenizer"]

        # Convert KB dataframe to dictionary for O(1) lookup
        # Key: (before, class), Value: after
        print("Indexing Knowledge Base...")
        kb_df = artifacts["kb_df"]
        self.kb_map = dict(zip(zip(kb_df["before"], kb_df["class"]), kb_df["after"]))

        self.test_grouped = artifacts["test_grouped"]

        # 2. Load Tagger Model
        print(f"Loading Tagger model from {self.config.TAGGER_MODEL_PATH}...")
        self.tagger_model = QuadHybridBiLSTM(
            num_classes=len(self.vocab_classes),
            vocab_words=self.vocab_words,
            vocab_chars=self.vocab_chars,
            vocab_bpe_size=self.config.BPE_VOCAB_SIZE,
        )
        if os.path.exists(self.config.TAGGER_MODEL_PATH):
            self.tagger_model.load_state_dict(
                torch.load(self.config.TAGGER_MODEL_PATH, map_location=self.device)
            )
        else:
            print(
                "WARNING: Tagger model checkpoint not found. Predictions will be random (untrained)."
            )
        self.tagger_model.to(self.device)
        self.tagger_model.eval()

        # 3. Load Seq2Seq Model
        print(f"Loading Seq2Seq model from {self.config.SEQ2SEQ_MODEL_PATH}...")
        self.seq2seq_model = CharTransformer(
            vocab_chars_size=len(self.vocab_chars),
            vocab_classes_size=len(self.vocab_classes),
        )
        if os.path.exists(self.config.SEQ2SEQ_MODEL_PATH):
            self.seq2seq_model.load_state_dict(
                torch.load(self.config.SEQ2SEQ_MODEL_PATH, map_location=self.device)
            )
        else:
            print(
                "WARNING: Seq2Seq model checkpoint not found. Predictions will be random (untrained)."
            )
        self.seq2seq_model.to(self.device)
        self.seq2seq_model.eval()

    def predict_classes(self):
        """
        Runs the Tagger on the test set to get class predictions.
        Returns a flat list of (id, token, predicted_class).
        """
        print("Running Tagger Inference...")
        dataset = TaggerDataset(
            self.test_grouped,
            self.vocab_words,
            self.vocab_chars,
            self.vocab_classes,
            self.bpe_tokenizer,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
        )

        flat_predictions = []
        global_idx = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Tagging"):
                # Move to device
                word_ids = batch["word_ids"].to(self.device)
                char_ids = batch["char_ids"].to(self.device)
                bpe_ids = batch["bpe_ids"].to(self.device)
                features = batch["features"].to(self.device)

                # Forward
                logits = self.tagger_model(word_ids, char_ids, bpe_ids, features)
                preds = torch.argmax(logits, dim=2).cpu().numpy()  # (B, S)

                batch_size = word_ids.size(0)

                # Map back to original tokens
                for b in range(batch_size):
                    row = self.test_grouped.iloc[global_idx]
                    sent_ids = row["id"]  # List of IDs for this sentence
                    sent_tokens = row["before"]  # List of tokens

                    # Get predictions for this sentence
                    # Note: Dataset truncates to MAX_SENT_LEN.
                    # We assume len(sent_ids) <= MAX_SENT_LEN based on EDA.
                    sent_preds = preds[b]

                    valid_len = len(sent_ids)
                    # Safety check
                    if valid_len > self.config.MAX_SENT_LEN:
                        # This case should be rare/non-existent given EDA (max 233)
                        valid_len = self.config.MAX_SENT_LEN

                    for k in range(valid_len):
                        cls_idx = sent_preds[k]
                        cls_name = self.vocab_classes.get_token(cls_idx)
                        flat_predictions.append((sent_ids[k], sent_tokens[k], cls_name))

                    global_idx += 1

        return flat_predictions

    def generate_sequences(self, tokens, classes):
        """
        Runs the Seq2Seq model on a list of tokens.
        """
        print(f"Running Seq2Seq Generation for {len(tokens)} tokens...")
        dataset = InferenceSeq2SeqDataset(
            tokens, classes, self.vocab_chars, self.vocab_classes
        )

        loader = DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
        )

        generated_texts = []
        sos_id = self.vocab_chars.get_id("<SOS>")
        eos_id = self.vocab_chars.get_id("<EOS>")

        with torch.no_grad():
            for batch in tqdm(loader, desc="Generating"):
                src_ids = batch["src_ids"].to(self.device)
                class_id = batch["class_id"].to(self.device)

                # Greedy Decode
                output_ids = self.seq2seq_model.predict(
                    src_ids,
                    class_id,
                    max_len=self.config.MAX_TOKEN_CHAR_LEN,
                    sos_id=sos_id,
                    eos_id=eos_id,
                )

                # Convert IDs to Strings
                output_ids = output_ids.cpu().numpy()
                for seq in output_ids:
                    chars = []
                    for idx in seq:
                        if idx == sos_id:
                            continue
                        if idx == eos_id:
                            break
                        if idx == self.vocab_chars.get_id("<PAD>"):
                            continue
                        chars.append(self.vocab_chars.get_token(idx))
                    generated_texts.append("".join(chars))

        return generated_texts

    def predict(self):
        """
        Main execution method.
        """
        # 1. Load Resources
        self.load_resources()

        # 2. Predict Classes
        flat_predictions = self.predict_classes()

        # 3. Apply Hybrid Logic
        final_results = []  # List of (id, after)

        # Lists for batch generation
        gen_indices = []
        gen_tokens = []
        gen_classes = []

        print("Applying Hybrid Normalization Logic...")
        for i, (uid, token, cls) in enumerate(flat_predictions):
            # Strategy 1: Knowledge Base Lookup
            if (token, cls) in self.kb_map:
                final_results.append(self.kb_map[(token, cls)])

            # Strategy 2: Copy if PLAIN/PUNCT
            elif cls == "PLAIN" or cls == "PUNCT":
                final_results.append(token)

            # Strategy 3: Neural Generation (Fallback)
            else:
                # Placeholder, will fill later
                final_results.append(None)
                gen_indices.append(i)
                gen_tokens.append(token)
                gen_classes.append(cls)

        # 4. Run Generation for Fallback cases
        if gen_indices:
            generated_texts = self.generate_sequences(gen_tokens, gen_classes)

            # Fill back into results
            for idx, text in zip(gen_indices, generated_texts):
                final_results[idx] = text

        # 5. Format Submission
        ids = [x[0] for x in flat_predictions]

        # Ensure no None values remain
        final_results = [res if res is not None else "" for res in final_results]

        return ids, final_results


def generate_submission():
    """
    Wrapper function to run the pipeline and save submission.
    """
    pipeline = NormalizationPipeline()
    ids, predictions = pipeline.predict()

    config = Config()
    save_submission(ids, predictions, config.SUBMISSION_FILE)
