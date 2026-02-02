import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import get_device, set_seed
from library.preprocessing import build_vocabularies, build_knowledge_base
from library.data_loader import get_tagger_loaders
from library.model_tagger import MultiGranularityTagger
from library.model_seq2seq import TransformerFallback


class NormalizationPipeline:
    def __init__(self, debug=Config.DEBUG):
        self.debug = debug
        self.device = get_device()
        set_seed(Config.SEED)

        print(f"Initializing Inference Pipeline (Device: {self.device})...")

        # 1. Load Raw Test Data for ID->Text Mapping
        # The DataLoader provides features and IDs, but we need raw text for KB lookup and Seq2Seq
        print("Loading test metadata for text lookup...")
        self.df_test_meta = pd.read_csv(Config.TEST_FILE)
        # Create a dictionary for fast lookup: id -> before
        self.id_to_text = pd.Series(
            self.df_test_meta.before.values, index=self.df_test_meta.id
        ).to_dict()

        # 2. Load Vocabularies & Tokenizer
        # We reuse get_tagger_loaders logic to get vocabs, ignoring the loaders for now
        # (We will get the test loader specifically later)
        print("Loading vocabularies...")
        (
            _,
            _,
            self.test_loader,
            self.word_vocab,
            self.char_vocab,
            self.class_vocab,
            self.bpe_tokenizer,
        ) = get_tagger_loaders(debug=debug, load_cached=True)

        # 3. Load Knowledge Base
        print("Loading Knowledge Base...")
        # We pass a dummy df because we expect to load from cache
        self.kb = build_knowledge_base(None, load_cached=True)

        # 4. Initialize & Load Tagger Model
        print("Loading Tagger Model...")
        self.tagger = MultiGranularityTagger(
            word_vocab_size=len(self.word_vocab),
            char_vocab_size=len(self.char_vocab),
            bpe_vocab_size=Config.BPE_VOCAB_SIZE,
            class_vocab_size=len(self.class_vocab),
            pad_idx=Config.PAD_IDX,
        ).to(self.device)

        tagger_path = Config.TAGGER_MODEL_PATH
        if os.path.exists(tagger_path):
            self.tagger.load_state_dict(
                torch.load(tagger_path, map_location=self.device)
            )
            self.tagger.eval()
        else:
            raise FileNotFoundError(f"Tagger model not found at {tagger_path}")

        # 5. Initialize & Load Seq2Seq Model
        print("Loading Seq2Seq Model...")
        self.seq2seq = TransformerFallback(
            char_vocab_size=len(self.char_vocab),
            class_vocab_size=len(self.class_vocab),
            pad_idx=Config.PAD_IDX,
        ).to(self.device)

        seq2seq_path = Config.SEQ2SEQ_MODEL_PATH
        if os.path.exists(seq2seq_path):
            self.seq2seq.load_state_dict(
                torch.load(seq2seq_path, map_location=self.device)
            )
            self.seq2seq.eval()
        else:
            print(
                f"Warning: Seq2Seq model not found at {seq2seq_path}. Fallback generation may fail."
            )
            # We don't raise error here to allow partial debugging if needed,
            # but usually this is critical.

        # Pre-compute special indices for Seq2Seq
        self.sos_idx = self.char_vocab.stoi[Config.SOS_TOKEN]
        self.eos_idx = self.char_vocab.stoi[Config.EOS_TOKEN]
        self.unk_char_idx = self.char_vocab.stoi[Config.UNK_TOKEN]

    def _encode_seq2seq_src(self, text_list):
        """
        Encodes a list of raw strings into a padded tensor for Seq2Seq source.
        """
        batch_src_ids = []
        for text in text_list:
            chars = list(str(text))[: Config.MAX_SEQ2SEQ_LEN]
            ids = [self.char_vocab.stoi.get(c, self.unk_char_idx) for c in chars]
            batch_src_ids.append(torch.tensor(ids, dtype=torch.long))

        # Pad
        src_padded = torch.nn.utils.rnn.pad_sequence(
            batch_src_ids, batch_first=True, padding_value=Config.PAD_IDX
        )
        return src_padded

    def _greedy_decode(self, src_ids, class_ids):
        """
        Performs greedy decoding for a batch of source sequences.
        Args:
            src_ids: (Batch, Src_Len)
            class_ids: (Batch)
        Returns:
            List of decoded strings.
        """
        batch_size = src_ids.size(0)
        max_len = Config.MAX_SEQ2SEQ_LEN

        # Initialize decoder input with SOS
        tgt = torch.full(
            (batch_size, 1), self.sos_idx, dtype=torch.long, device=self.device
        )

        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Loop
        for _ in range(max_len):
            # Forward pass
            # tgt shape: (Batch, Curr_Len)
            logits = self.seq2seq(src_ids, tgt, class_ids)

            # Get last token logits: (Batch, Vocab)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)  # (Batch)

            # Update finished mask if EOS is generated
            finished |= next_token == self.eos_idx

            # Append prediction
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

            if finished.all():
                break

        # Decode to strings
        decoded_strings = []
        tgt_cpu = tgt.cpu().numpy()

        for i in range(batch_size):
            # Skip SOS (index 0)
            indices = tgt_cpu[i, 1:]
            chars = []
            for idx in indices:
                if idx == self.eos_idx:
                    break
                # Skip padding if any (though greedy decode usually stops at EOS)
                if idx == Config.PAD_IDX:
                    continue

                char = self.char_vocab.lookup_token(idx)
                if char is not None:
                    chars.append(char)
            decoded_strings.append("".join(chars))

        return decoded_strings

    def run_inference(self):
        print("Starting Inference...")

        results = {}  # id -> predicted_text

        # Buffer for Seq2Seq processing
        # Stores tuples: (id, raw_text, class_idx)
        seq2seq_buffer = []
        seq2seq_batch_size = Config.BATCH_SIZE

        # 1. Tagger & KB Lookup Loop
        with torch.no_grad():
            for batch in self.test_loader:
                word_ids = batch["word_ids"].to(self.device)
                char_ids = batch["char_ids"].to(self.device)
                bpe_ids = batch["bpe_ids"].to(self.device)
                mask = batch["mask"].to(self.device)
                ids = batch["ids"]  # List of strings

                # Tagger Forward
                logits = self.tagger(word_ids, char_ids, bpe_ids, mask)
                pred_class_indices = torch.argmax(logits, dim=-1).cpu().numpy()

                # Process Batch
                for i, token_id in enumerate(ids):
                    # Get Raw Text
                    raw_text = str(self.id_to_text.get(token_id, ""))

                    # Get Predicted Class
                    class_idx = pred_class_indices[i]
                    class_name = self.class_vocab.lookup_token(class_idx)

                    # Strategy:
                    # 1. Check Knowledge Base
                    kb_result = self.kb.get(raw_text, class_name)

                    if kb_result is not None:
                        results[token_id] = kb_result
                    else:
                        # 2. Heuristic Fallback for PLAIN/PUNCT
                        # If not in KB, PLAIN usually maps to itself.
                        if class_name in ["PLAIN", "PUNCT"]:
                            results[token_id] = raw_text
                        else:
                            # 3. Queue for Seq2Seq
                            seq2seq_buffer.append((token_id, raw_text, class_idx))

                    # Process Seq2Seq buffer if full
                    if len(seq2seq_buffer) >= seq2seq_batch_size:
                        self._process_seq2seq_buffer(seq2seq_buffer, results)
                        seq2seq_buffer = []

            # Process remaining Seq2Seq buffer
            if len(seq2seq_buffer) > 0:
                self._process_seq2seq_buffer(seq2seq_buffer, results)

        # 2. Save Results
        print(f"Inference complete. Total predictions: {len(results)}")
        self._save_submission(results)

    def _process_seq2seq_buffer(self, buffer, results_dict):
        """
        Runs the Seq2Seq model on a batch of OOV tokens.
        Updates results_dict in-place.
        """
        if not buffer:
            return

        # Unpack
        ids, raw_texts, class_indices = zip(*buffer)

        # Prepare Inputs
        src_ids = self._encode_seq2seq_src(raw_texts).to(self.device)
        class_ids = torch.tensor(class_indices, dtype=torch.long).to(self.device)

        # Generate
        with torch.no_grad():
            generated_texts = self._greedy_decode(src_ids, class_ids)

        # Store
        for token_id, gen_text in zip(ids, generated_texts):
            results_dict[token_id] = gen_text

    def _save_submission(self, results):
        """
        Formats and saves the submission file.
        """
        # Ensure correct order based on sample submission or test file
        # We use the test metadata order
        submission_ids = self.df_test_meta["id"].tolist()

        # Create DataFrame
        # Fill missing with empty string or raw text?
        # Logic implies we should have a result for everything.
        # If missing (shouldn't happen), default to raw text.

        final_data = []
        for uid in submission_ids:
            if uid in results:
                final_data.append({"id": uid, "after": results[uid]})
            else:
                # Fallback safety
                raw = str(self.id_to_text.get(uid, ""))
                final_data.append({"id": uid, "after": raw})

        df_sub = pd.DataFrame(final_data)

        # Save
        out_path = Config.SUBMISSION_PATH
        print(f"Saving submission to {out_path}...")

        # Use quoting to handle text with special characters safely
        df_sub.to_csv(out_path, index=False)
        print("Done.")


def run_inference_pipeline():
    pipeline = NormalizationPipeline()
    pipeline.run_inference()
