import os
import pandas as pd
import torch
from library.config import Config
from library.utils import get_device, is_digit_token
from library.data_processor import (
    get_tokenizer,
    SOS_TOKEN,
    EOS_TOKEN,
    construct_window_input,
    ContextAnchoredDataset,
    collate_fn,
)
from library.symbolic_agent import NgramMemory
from library.neural_agent import NeuralTrainer


class HybridPredictor:
    """
    Implements the Context-Anchored Hybrid Neuro-Symbolic System for text normalization.
    Orchestrates the Priority Cascade: Trigram -> Neural -> Bigram -> Unigram -> Identity.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = get_device()

        # --- 1. Load Symbolic Memory ---
        # NgramMemory handles its own caching (parquet files)
        print("Initializing Symbolic Memory...")
        self.memory = NgramMemory(config)
        self.memory.build_stats(load_cached_data=True)

        # --- 2. Load Neural Model ---
        print("Initializing Neural Model...")
        self.tokenizer = get_tokenizer(config)
        self.trainer = NeuralTrainer(config, self.tokenizer)

        # Attempt to load checkpoint
        if os.path.exists(config.model_checkpoint_path):
            print(f"Loading neural checkpoint from {config.model_checkpoint_path}...")
            self.trainer.load_model(config.model_checkpoint_path)
            self.model_ready = True
        else:
            print(
                f"Warning: No neural checkpoint found at {config.model_checkpoint_path}. Neural branch will be disabled."
            )
            self.model_ready = False

    def _generate_neural_preds(self, df_test: pd.DataFrame) -> dict:
        """
        Generates neural predictions for the subset of tokens containing digits.
        Constructs context-anchored inputs on the fly.

        Returns:
            dict: Mapping of 'sentence_id_token_id' -> 'normalized_text'
        """
        if not self.model_ready:
            return {}

        print("Preparing neural inference data...")

        # Group by sentence to allow context extraction
        # We assume df_test is sorted by sentence_id, token_id
        grouped_before = df_test.groupby("sentence_id")["before"].apply(list)
        grouped_ids = df_test.groupby("sentence_id")["token_id"].apply(list)

        samples = []

        # Iterate over sentences to find tokens needing neural normalization
        for sid in grouped_before.index:
            tokens = grouped_before[sid]
            t_ids = grouped_ids[sid]
            seq_len = len(tokens)

            for i in range(seq_len):
                token_text = str(tokens[i])

                # Heuristic: Only send tokens with digits to the neural model
                # This aligns with the "Context Starvation" fix strategy
                if is_digit_token(token_text):
                    input_str = construct_window_input(
                        tokens, i, self.config.context_window
                    )
                    full_id = f"{sid}_{t_ids[i]}"

                    samples.append({"id": full_id, "input_text": input_str})

        if not samples:
            return {}

        print(f"Running neural inference on {len(samples)} tokens...")

        # Create temporary dataset and dataloader
        df_samples = pd.DataFrame(samples)
        dataset = ContextAnchoredDataset(
            df_samples, self.tokenizer, self.config, is_test=True
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=lambda b: collate_fn(b, self.tokenizer.pad_token_id),
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # Inference Loop
        preds = {}
        self.trainer.model.eval()

        with torch.no_grad():
            for batch in dataloader:
                src = batch["src"].to(self.device)
                batch_ids = batch["id"]

                # Greedy generation
                generated_ids = self.trainer.model.generate(
                    src, max_len=self.config.max_seq_len, device=self.device
                )

                # Decode
                for i, seq in enumerate(generated_ids):
                    seq_list = seq.tolist()

                    # Truncate at EOS
                    if self.tokenizer.eos_token_id in seq_list:
                        eos_idx = seq_list.index(self.tokenizer.eos_token_id)
                        seq_list = seq_list[:eos_idx]

                    text = self.tokenizer.decode(seq_list, remove_special_tokens=True)
                    preds[batch_ids[i]] = text

        return preds

    def predict(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the full Hybrid Inference Cascade on the test dataframe.
        """
        # Pre-processing
        df_test["before"] = df_test["before"].fillna("").astype(str)
        # Sort to ensure correct context reconstruction
        df_test = df_test.sort_values(["sentence_id", "token_id"])

        # 1. Neural Branch (Batch Processing)
        neural_preds = self._generate_neural_preds(df_test)

        # 2. Sequential Cascade
        print(
            "Applying Priority Cascade (Trigram -> Neural -> Bigram -> Unigram -> Identity)..."
        )
        results = []

        # Group by sentence for context
        grouped = df_test.groupby("sentence_id")

        for sid, group in grouped:
            tokens = group["before"].tolist()
            t_ids = group["token_id"].tolist()
            seq_len = len(tokens)

            for i in range(seq_len):
                curr_tok = tokens[i]
                full_id = f"{sid}_{t_ids[i]}"

                # Context extraction
                prev_tok = tokens[i - 1] if i > 0 else SOS_TOKEN
                next_tok = tokens[i + 1] if i < seq_len - 1 else EOS_TOKEN

                # --- Step 1: Trigram (Specific Memory) ---
                norm = self.memory.query_trigram(prev_tok, curr_tok, next_tok)

                if norm is None:
                    # --- Step 2: Neural (Generalization) ---
                    # If the token was flagged for neural processing (has digits) and we have a prediction
                    if full_id in neural_preds:
                        norm = neural_preds[full_id]
                    else:
                        # --- Step 3: Bigram (General Memory) ---
                        norm = self.memory.query_bigram(prev_tok, curr_tok)

                        if norm is None:
                            # --- Step 4: Unigram (Fallback Memory) ---
                            norm = self.memory.query_unigram(curr_tok)

                            if norm is None:
                                # --- Step 5: Identity (Fallback) ---
                                norm = curr_tok

                results.append({"id": full_id, "after": norm})

        return pd.DataFrame(results)

    def generate_submission(self):
        """
        Orchestrates the full submission generation process.
        Loads test metadata, runs prediction, and saves to CSV.
        """
        test_path = os.path.join(self.config.metadata_dir, "test.csv")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test metadata not found at {test_path}")

        print(f"Loading test data from {test_path}...")
        df_test = pd.read_csv(test_path)

        # Run prediction
        df_preds = self.predict(df_test)

        # Save
        submission_path = self.config.submission_path
        print(f"Saving submission to {submission_path}...")

        # Ensure directory exists
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)

        df_preds.to_csv(submission_path, index=False)
        print("Submission generation complete.")


def run_inference_pipeline(config: Config):
    """
    Convenience wrapper to run the inference pipeline.
    """
    predictor = HybridPredictor(config)
    predictor.generate_submission()
