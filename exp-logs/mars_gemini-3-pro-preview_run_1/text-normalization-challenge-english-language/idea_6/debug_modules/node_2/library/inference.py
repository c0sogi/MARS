import os
import torch
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any

from library.config import Config
from library.utils import get_device, load_checkpoint
from library.data_loader import get_data, get_tagger_loaders
from library.models_tagger import BiLSTM_CRF
from library.models_seq2seq import CharTransformer


class NormalizationPipeline:
    """
    Orchestrates the Text Normalization Inference Pipeline.
    1. Tagger (Bi-LSTM-CRF) -> Predicts token class (e.g., DATE, CARDINAL).
    2. Knowledge Base -> Looks up (token, class) for deterministic mapping.
    3. Seq2Seq (Transformer) -> Generates normalized text for OOV tokens.
    """

    def __init__(self, load_cached: bool = True):
        self.device = get_device()
        self.load_cached = load_cached

        print("Initializing Normalization Pipeline...")

        # 1. Load Data & Vocabularies
        # We need the test_grouped dataframe to map predictions back to submission IDs
        (
            self.vocab_tokens,
            self.vocab_chars,
            self.vocab_classes,
            _,  # train_grouped
            _,  # val_grouped
            self.test_grouped,
            _,  # seq2seq_train_df
        ) = get_data(load_cached=self.load_cached)

        # 2. Load Knowledge Base
        self.kb = self._load_knowledge_base()

        # 3. Load Models
        self.tagger = self._load_tagger()
        self.seq2seq = self._load_seq2seq()

    def _load_knowledge_base(self) -> Dict[tuple, str]:
        """Loads the Knowledge Base from parquet."""
        kb_path = Config.KNOWLEDGE_BASE_PATH
        if os.path.exists(kb_path):
            print(f"Loading Knowledge Base from {kb_path}...")
            kb_df = pd.read_parquet(kb_path)
            # Create a dict for O(1) lookup: (before, class) -> after
            return {
                (row["before"], row["class"]): row["after"]
                for _, row in kb_df.iterrows()
            }
        else:
            print(
                "Warning: Knowledge Base file not found. Inference will rely solely on models."
            )
            return {}

    def _load_tagger(self) -> BiLSTM_CRF:
        """Initializes and loads the Tagger model."""
        print("Loading Tagger Model...")
        model = BiLSTM_CRF(
            vocab_size=len(self.vocab_tokens),
            char_vocab_size=len(self.vocab_chars),
            num_classes=len(self.vocab_classes),
        ).to(self.device)

        try:
            load_checkpoint(Config.TAGGER_MODEL_PATH, model, device=self.device)
        except FileNotFoundError:
            print(
                f"Warning: Tagger checkpoint not found at {Config.TAGGER_MODEL_PATH}."
            )

        model.eval()
        return model

    def _load_seq2seq(self) -> CharTransformer:
        """Initializes and loads the Seq2Seq model."""
        print("Loading Seq2Seq Model...")
        model = CharTransformer(
            num_chars=len(self.vocab_chars), num_classes=len(self.vocab_classes)
        ).to(self.device)

        try:
            load_checkpoint(Config.SEQ2SEQ_MODEL_PATH, model, device=self.device)
        except FileNotFoundError:
            print(
                f"Warning: Seq2Seq checkpoint not found at {Config.SEQ2SEQ_MODEL_PATH}."
            )

        model.eval()
        return model

    def predict(self, batch_size: int = Config.BATCH_SIZE):
        """
        Runs the full inference pipeline and generates the submission file.
        """
        # Get Test Loader (aligned with self.test_grouped)
        _, _, test_loader, _, _, _ = get_tagger_loaders(
            batch_size=batch_size, load_cached=self.load_cached
        )

        results = []
        df_idx = 0

        print(f"Starting Inference on {len(self.test_grouped)} sentences...")

        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                token_ids = batch["token_ids"].to(self.device)
                char_ids = batch["char_ids"].to(self.device)
                mask = batch["mask"].to(self.device)

                # --- Step 1: Tagger Prediction ---
                # Returns (Batch, Seq_Len) indices
                pred_tags = self.tagger.decode(token_ids, char_ids, mask)

                # Retrieve corresponding raw data rows
                current_batch_size = token_ids.size(0)
                df_batch = self.test_grouped.iloc[df_idx : df_idx + current_batch_size]
                df_idx += current_batch_size

                # Prepare buffers for this batch
                batch_results = []
                oov_queue = []  # List of dicts to process with Seq2Seq

                # Iterate through sentences in the batch
                for b_i in range(current_batch_size):
                    row = df_batch.iloc[b_i]
                    raw_tokens = row["before"]  # List of strings
                    row_ids = row["id"]  # List of strings (sentence_id_token_id)

                    # Iterate through tokens in the sentence
                    # Note: raw_tokens length might be less than Config.MAX_SEQ_LEN (padding)
                    seq_len = len(raw_tokens)

                    for t_i in range(seq_len):
                        token_str = raw_tokens[t_i]
                        row_id = row_ids[t_i]

                        # Get predicted class
                        class_idx = pred_tags[b_i, t_i].item()
                        class_str = self.vocab_classes.lookup_token(class_idx)

                        norm_text = token_str  # Default fallback (copy)

                        # --- Step 2: Retrieval & Logic ---
                        kb_key = (token_str, class_str)

                        if kb_key in self.kb:
                            # 2a. Knowledge Base Hit
                            norm_text = self.kb[kb_key]
                        elif class_str == "PLAIN" or class_str == "PUNCT":
                            # 2b. Identity Class (Copy)
                            norm_text = token_str
                        else:
                            # 2c. OOV Complex Token -> Queue for Seq2Seq
                            oov_queue.append(
                                {
                                    "batch_res_idx": len(
                                        batch_results
                                    ),  # Index to update later
                                    "token_str": token_str,
                                    "class_idx": class_idx,
                                }
                            )
                            norm_text = "<PENDING>"  # Placeholder

                        batch_results.append({"id": row_id, "after": norm_text})

                # --- Step 3: Seq2Seq Generation for OOV ---
                if oov_queue:
                    self._process_oov_queue(oov_queue, batch_results)

                results.extend(batch_results)

        # Save Submission
        self._save_submission(results)

    def _process_oov_queue(self, oov_queue: List[Dict], batch_results: List[Dict]):
        """
        Runs Seq2Seq model on a batch of OOV tokens and updates batch_results in-place.
        """
        if not oov_queue:
            return

        # Prepare Inputs
        src_id_list = []
        class_id_list = []

        unk_idx = self.vocab_chars.stoi[Config.UNK_TOKEN]
        pad_idx = self.vocab_chars.stoi[Config.PAD_TOKEN]

        for item in oov_queue:
            token_str = item["token_str"]
            # Encode chars
            c_ids = [self.vocab_chars.stoi.get(c, unk_idx) for c in token_str]

            # Pad / Truncate to MAX_CHAR_LEN
            if len(c_ids) > Config.MAX_CHAR_LEN:
                c_ids = c_ids[: Config.MAX_CHAR_LEN]
            else:
                c_ids += [pad_idx] * (Config.MAX_CHAR_LEN - len(c_ids))

            src_id_list.append(c_ids)
            class_id_list.append(item["class_idx"])

        src_tensor = torch.tensor(src_id_list, dtype=torch.long).to(self.device)
        class_tensor = torch.tensor(class_id_list, dtype=torch.long).to(self.device)

        # Predict
        sos_idx = self.vocab_chars.stoi[Config.SOS_TOKEN]
        eos_idx = self.vocab_chars.stoi[Config.EOS_TOKEN]

        # generated_ids: (Batch, Max_Output_Len)
        generated_ids = self.seq2seq.predict(
            src_tensor,
            class_tensor,
            sos_idx,
            eos_idx,
            max_len=Config.SEQ2SEQ_MAX_OUTPUT_LEN,
        )

        # Decode and Update
        for i, row_ids in enumerate(generated_ids):
            # Convert indices to string
            chars = []
            for idx in row_ids:
                idx = idx.item()
                if idx == eos_idx:
                    break
                if idx == pad_idx:
                    continue
                chars.append(self.vocab_chars.lookup_token(idx))

            pred_str = "".join(chars)

            # Update the specific result entry
            target_idx = oov_queue[i]["batch_res_idx"]
            batch_results[target_idx]["after"] = pred_str

    def _save_submission(self, results: List[Dict]):
        """Saves the results to CSV."""
        df = pd.DataFrame(results)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save
        df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH} with {len(df)} rows.")
