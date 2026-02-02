import os
import torch
import pandas as pd
import csv
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import setup_logger, ensure_dir
from library.models import PriorAugmentedBiLSTMTagger, TransformerSeq2Seq
from library.data_loader import TaggerDataset, KnowledgeBase
from library.vocab import VocabManager
from library.features import RegexFeatureExtractor, GlobalPriorManager

logger = setup_logger("inference")


class InferencePipeline:
    """
    Orchestrates the Classify-Retrieve-Generate pipeline for inference.
    """

    def __init__(self, load_cached_data=True):
        self.device = Config.DEVICE
        self.load_cached_data = load_cached_data

        # 1. Load Managers and Vocabs
        logger.info("Loading vocabularies and managers...")
        self.vocab_manager = VocabManager()
        self.vocab_manager.build_or_load(load_cached_data=load_cached_data)

        self.prior_manager = GlobalPriorManager()
        # Priors are static lookup during inference, need to be loaded
        self.prior_manager.build_or_load(None, load_cached_data=True)

        self.kb = KnowledgeBase()
        self.kb.load()

        self.class_vocab = self.vocab_manager.get_class_vocab()
        self.char_vocab = self.vocab_manager.get_char_vocab()
        self.plain_class_idx = self.class_vocab["PLAIN"]

        # 2. Load Models
        self._load_models()

    def _load_models(self):
        logger.info("Loading models...")

        # Tagger
        regex_extractor = RegexFeatureExtractor()
        regex_dim = regex_extractor.get_feature_dim()
        num_classes = len(self.class_vocab)

        self.tagger = PriorAugmentedBiLSTMTagger(
            self.vocab_manager, regex_dim, num_classes
        ).to(self.device)

        if os.path.exists(Config.TAGGER_MODEL_PATH):
            state_dict = torch.load(Config.TAGGER_MODEL_PATH, map_location=self.device)
            self.tagger.load_state_dict(state_dict)
            self.tagger.eval()
            logger.info(f"Tagger loaded from {Config.TAGGER_MODEL_PATH}")
        else:
            logger.warning(
                "Tagger checkpoint not found. Inference will use random weights."
            )

        # Seq2Seq
        self.seq2seq = TransformerSeq2Seq(self.vocab_manager, num_classes).to(
            self.device
        )

        if os.path.exists(Config.SEQ2SEQ_MODEL_PATH):
            state_dict = torch.load(Config.SEQ2SEQ_MODEL_PATH, map_location=self.device)
            self.seq2seq.load_state_dict(state_dict)
            self.seq2seq.eval()
            logger.info(f"Seq2Seq loaded from {Config.SEQ2SEQ_MODEL_PATH}")
        else:
            logger.warning(
                "Seq2Seq checkpoint not found. Inference will use random weights."
            )

    def _load_raw_text_map(self, test_file):
        """
        Loads the test CSV to create a mapping from ID to raw text.
        Required because the DataLoader returns features/IDs, not raw text.
        """
        logger.info(f"Loading raw text mapping from {test_file}...")
        df = pd.read_csv(test_file)
        if "id" not in df.columns:
            df["id"] = df["sentence_id"].astype(str) + "_" + df["token_id"].astype(str)

        # Create dict: id -> before
        return dict(zip(df["id"], df["before"].astype(str)))

    def predict(
        self, test_file=Config.TEST_FILE, batch_size=Config.BATCH_SIZE, limit=None
    ):
        """
        Runs the inference pipeline on the test set.

        Args:
            test_file (str): Path to the test CSV.
            batch_size (int): Batch size for inference.
            limit (int, optional): Limit number of samples for debugging.
        """
        logger.info("Starting prediction pipeline...")

        # 1. Prepare Data
        id_to_token = self._load_raw_text_map(test_file)

        test_ds = TaggerDataset(
            test_file,
            self.vocab_manager,
            self.prior_manager,
            mode="test",
            load_cached_data=self.load_cached_data,
        )

        # Apply limit if requested (by slicing the dataset's internal dataframe if possible,
        # or just breaking the loop. TaggerDataset loads all, so we break loop).

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=TaggerDataset.collate_fn,
        )

        results = []  # List of (id, after)

        processed_count = 0

        with torch.no_grad():
            for batch in test_loader:
                if limit and processed_count >= limit:
                    break

                # Move inputs to device
                word_ids = batch["word_ids"].to(self.device)
                bpe_ids = batch["bpe_ids"].to(self.device)
                char_ids = batch["char_ids"].to(self.device)
                regex_feats = batch["regex_feats"].to(self.device)
                prior_feats = batch["prior_feats"].to(self.device)
                batch_ids = batch["ids"]  # List of lists of strings

                # --- Stage 1: Classify (Tagger) ---
                logits = self.tagger(
                    word_ids, bpe_ids, char_ids, regex_feats, prior_feats
                )
                pred_class_indices = torch.argmax(logits, dim=2).cpu().numpy()

                current_batch_size = word_ids.size(0)

                # Collect items for Seq2Seq fallback
                # Structure: (row_id, raw_token, class_idx)
                fallback_items = []

                # Iterate through batch
                for b in range(current_batch_size):
                    seq_len = len(batch_ids[b])
                    for s in range(seq_len):
                        row_id = batch_ids[b][s]

                        # Retrieve raw text
                        raw_token = id_to_token.get(row_id, "")

                        class_idx = pred_class_indices[b, s]
                        class_name = self.class_vocab.lookup_token(class_idx)

                        # --- Stage 2: Retrieve (Knowledge Base) ---
                        kb_result = self.kb.lookup(raw_token, class_name)

                        if kb_result is not None:
                            results.append((row_id, kb_result))
                        elif class_idx == self.plain_class_idx:
                            # PLAIN class usually implies copy, even if not in KB
                            results.append((row_id, raw_token))
                        else:
                            # --- Stage 3: Generate (Fallback Prep) ---
                            fallback_items.append((row_id, raw_token, class_idx))

                # --- Stage 3: Generate (Seq2Seq Execution) ---
                if fallback_items:
                    self._process_fallbacks(fallback_items, results)

                processed_count += current_batch_size

        return results

    def _process_fallbacks(self, fallback_items, results_list):
        """
        Runs Seq2Seq generation for a list of fallback items and appends to results.

        Args:
            fallback_items: List of (row_id, raw_token, class_idx)
            results_list: List to append (row_id, normalized_text)
        """
        # Prepare batch
        src_ids_list = []
        class_ids_list = []
        row_ids_list = []

        for row_id, token, class_idx in fallback_items:
            # Encode chars
            c_ids = [self.char_vocab[c] for c in token]
            src_ids_list.append(torch.tensor(c_ids, dtype=torch.long))
            class_ids_list.append(class_idx)
            row_ids_list.append(row_id)

        # Pad sequence
        src_padded = torch.nn.utils.rnn.pad_sequence(
            src_ids_list, batch_first=True, padding_value=0  # 0 is <pad> usually
        ).to(self.device)

        class_tensor = torch.tensor(class_ids_list, dtype=torch.long).to(self.device)

        # Generate
        generated_ids = self.seq2seq.generate(src_padded, class_tensor)
        generated_ids = generated_ids.cpu().numpy()

        # Decode
        for i, g_ids in enumerate(generated_ids):
            tokens = []
            for idx in g_ids:
                if idx == self.char_vocab["<eos>"]:
                    break
                if idx not in [self.char_vocab["<pad>"], self.char_vocab["<sos>"]]:
                    try:
                        tokens.append(self.char_vocab.lookup_token(idx))
                    except KeyError:
                        pass  # Ignore unknown chars

            norm_text = "".join(tokens)
            results_list.append((row_ids_list[i], norm_text))

    def save_submission(self, results, output_path=Config.SUBMISSION_PATH):
        """
        Saves the results to a CSV file.
        """
        ensure_dir(output_path)
        logger.info(f"Saving submission to {output_path}...")

        # Sort by ID is not strictly required but good for consistency.
        # IDs are strings "sentence_token", sorting might be lexicographical.
        # We write in the order processed or just write out.

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(["id", "after"])
            for row_id, after_text in results:
                writer.writerow([row_id, after_text])

        logger.info("Submission saved successfully.")


def run_inference(
    test_file=Config.TEST_FILE, output_path=Config.SUBMISSION_PATH, limit=None
):
    """
    Main entry point function to run inference.
    """
    pipeline = InferencePipeline(load_cached_data=True)
    results = pipeline.predict(test_file=test_file, limit=limit)
    pipeline.save_submission(results, output_path=output_path)
