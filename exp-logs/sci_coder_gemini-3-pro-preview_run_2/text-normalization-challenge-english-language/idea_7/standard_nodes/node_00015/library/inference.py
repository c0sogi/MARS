import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from library.config import Config
from library.router_model import TokenClassifier
from library.generator_model import Seq2SeqNormalizer
from library.rule_based_norm import apply_rule
from library.data_utils import process_router_data, get_router_dataloader
from library.trainer import InferenceDataset, collate_inference


class HybridPredictor:
    """
    Implements the hybrid inference pipeline (Idea 7).
    Routes tokens to either a deterministic rule engine or a neural generator
    based on a semantic classification step.
    """

    def __init__(
        self,
        router_checkpoint_dir: str = Config.ROUTER_CHECKPOINT_DIR,
        generator_checkpoint_dir: str = Config.GENERATOR_CHECKPOINT_DIR,
    ):
        """
        Initialize the predictor by loading models and tokenizers.

        Args:
            router_checkpoint_dir: Path to the saved Router model.
            generator_checkpoint_dir: Path to the saved Generator model.
        """
        self.device = Config.DEVICE

        # Load Router (Token Classification)
        print(f"Loading Router model from {router_checkpoint_dir}...")
        self.router_model = TokenClassifier.from_pretrained(router_checkpoint_dir)
        self.router_model.to(self.device)
        self.router_model.eval()
        self.router_tokenizer = AutoTokenizer.from_pretrained(
            Config.ROUTER_MODEL_NAME, add_prefix_space=True
        )

        # Load Generator (Seq2Seq)
        print(f"Loading Generator model from {generator_checkpoint_dir}...")
        self.generator_model = Seq2SeqNormalizer.from_pretrained(
            generator_checkpoint_dir
        )
        self.generator_model.to(self.device)
        self.generator_model.eval()
        self.generator_tokenizer = AutoTokenizer.from_pretrained(
            Config.GENERATOR_MODEL_NAME
        )

    def predict(
        self,
        load_cached_data: bool = True,
        batch_size: int = Config.ROUTER_VAL_BATCH_SIZE,
    ):
        """
        Runs the full prediction pipeline on the test set and saves the submission.

        Args:
            load_cached_data: Whether to use cached preprocessed data.
            batch_size: Batch size for the Router model inference.
        """
        Config.set_seed(Config.SEED)

        # ==========================================
        # 1. Load and Route (Router Model)
        # ==========================================
        print("Loading test data and running Router...")

        # Load grouped test data (Sentence Level)
        # process_router_data handles caching internally
        df_test_grouped = process_router_data(
            split="test", load_cached_data=load_cached_data
        )

        # Get DataLoader (Sequential, No Shuffle)
        # Note: We temporarily override Config batch size if needed, but get_router_dataloader
        # reads from Config.ROUTER_VAL_BATCH_SIZE for 'test' split.
        router_loader = get_router_dataloader(
            split="test", load_cached_data=load_cached_data
        )

        all_token_ids = []
        all_tokens = []
        all_pred_classes = []

        # Iterator to access raw tokens corresponding to batches
        df_iter = df_test_grouped.itertuples(index=False)

        with torch.no_grad():
            for batch in router_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Forward pass
                logits = self.router_model(input_ids, attention_mask).logits
                preds = torch.argmax(logits, dim=2).cpu().numpy()

                batch_len = input_ids.size(0)

                # Process each sentence in the batch
                for i in range(batch_len):
                    try:
                        row = next(df_iter)
                    except StopIteration:
                        break

                    raw_tokens = row.tokens
                    row_ids = row.token_ids

                    # Re-tokenize to align subword predictions to words
                    encoding = self.router_tokenizer(
                        raw_tokens,
                        is_split_into_words=True,
                        truncation=True,
                        max_length=Config.ROUTER_MAX_LEN,
                        return_attention_mask=False,
                    )
                    word_ids = encoding.word_ids()

                    sentence_preds = preds[i]
                    aligned_preds = []
                    prev_word_idx = None

                    # Alignment Logic: Take prediction of the first subword of each word
                    for j, word_idx in enumerate(word_ids):
                        if word_idx is None:
                            continue

                        if word_idx != prev_word_idx:
                            # Ensure we don't exceed raw tokens length (truncation case)
                            if word_idx < len(raw_tokens):
                                class_id = sentence_preds[j]
                                class_label = Config.ID2CLASS[class_id]
                                aligned_preds.append(class_label)
                            prev_word_idx = word_idx

                    # Fallback for truncated tokens: assume PLAIN
                    if len(aligned_preds) < len(raw_tokens):
                        aligned_preds.extend(
                            ["PLAIN"] * (len(raw_tokens) - len(aligned_preds))
                        )

                    all_token_ids.extend(row_ids)
                    all_tokens.extend(raw_tokens)
                    all_pred_classes.extend(aligned_preds)

        # ==========================================
        # 2. Hybrid Execution Strategy
        # ==========================================
        print("Applying Hybrid Normalization Rules...")

        final_results = {}
        gen_inputs = []
        gen_indices = []

        structured_set = Config.STRUCTURED_CLASSES
        unstructured_set = Config.UNSTRUCTURED_CLASSES

        for tid, token, cls in zip(all_token_ids, all_tokens, all_pred_classes):
            if cls == "PLAIN" or cls == "PUNCT":
                final_results[tid] = token
            elif cls in structured_set:
                # Path A: Deterministic Rules
                final_results[tid] = apply_rule(token, cls)
            elif cls in unstructured_set:
                # Path B: Queue for Generator
                # Format: "[CLASS] raw_token"
                gen_inputs.append(f"[{cls}] {token}")
                gen_indices.append(tid)
            else:
                # Fallback
                final_results[tid] = token

        # ==========================================
        # 3. Generator Inference (Unstructured)
        # ==========================================
        if gen_inputs:
            print(f"Running Generator on {len(gen_inputs)} complex tokens...")

            gen_dataset = InferenceDataset(
                gen_inputs, self.generator_tokenizer, max_len=Config.GEN_MAX_INPUT_LEN
            )

            gen_loader = DataLoader(
                gen_dataset,
                batch_size=Config.GEN_VAL_BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                collate_fn=collate_inference,
            )

            gen_outputs = []
            with torch.no_grad():
                for batch in gen_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)

                    generated_ids = self.generator_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=Config.GEN_MAX_TARGET_LEN,
                    )

                    decoded = self.generator_tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )
                    gen_outputs.extend(decoded)

            # Map generator outputs back to IDs
            for tid, output_text in zip(gen_indices, gen_outputs):
                final_results[tid] = output_text
        else:
            print("No unstructured tokens found. Skipping generator.")

        # ==========================================
        # 4. Save Submission
        # ==========================================
        print("Generating submission file...")

        submission_data = []
        # Iterate over all_token_ids to preserve original test set order
        for tid in all_token_ids:
            submission_data.append({"id": tid, "after": final_results.get(tid, "")})

        df_sub = pd.DataFrame(submission_data)

        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
