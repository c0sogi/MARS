import os
import torch
import pandas as pd
import numpy as np
import csv
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.data_factory import get_dataloader


class InferenceEngine:
    """
    Manages the inference process for Question Answering.
    Implements Global Confidence Aggregation and Majority Vote Ensembling.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL_NAME)
        self.test_metadata_path = Config.TEST_META_PATH

    def load_model(self, seed):
        """
        Loads a trained model checkpoint for a specific seed.
        """
        model_path = os.path.join(Config.QA_MODEL_DIR, f"model_seed_{seed}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Initialize model architecture
        model = AutoModelForTokenClassification.from_pretrained(
            Config.BASE_MODEL_NAME, num_labels=Config.NUM_LABELS
        )

        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()
        return model

    def extract_spans(self, input_ids, preds, probs):
        """
        Extracts candidate answer spans from a sequence of token predictions.

        Args:
            input_ids (np.array): Token IDs for the sequence.
            preds (np.array): Predicted class IDs (0=O, 1=B, 2=I).
            probs (np.array): Probability of the predicted class at each step.

        Returns:
            list: A list of tuples (confidence_score, decoded_text).
        """
        spans = []
        seq_len = len(input_ids)

        # Label mapping from data_factory: 0=O, 1=B, 2=I
        LABEL_B = 1
        LABEL_I = 2

        i = 0
        while i < seq_len:
            # A valid span must start with B-ANS
            if preds[i] == LABEL_B:
                start = i
                end = i

                # Initialize score accumulation
                score_sum = probs[i]
                count = 1

                # Extend span while subsequent tokens are I-ANS
                while (i + 1) < seq_len and preds[i + 1] == LABEL_I:
                    i += 1
                    end = i
                    score_sum += probs[i]
                    count += 1

                # Calculate confidence score (mean probability)
                avg_score = score_sum / count

                # Decode token IDs to string
                span_ids = input_ids[start : end + 1]
                text = self.tokenizer.decode(span_ids, skip_special_tokens=True).strip()

                # Filter out empty strings (e.g., if span was only special tokens)
                if text:
                    spans.append((avg_score, text))

            i += 1

        return spans

    def predict_single_model(self, model, dataloader):
        """
        Runs inference for a single model using Global Confidence Aggregation.

        Instead of taking the first span found, it collects all spans across
        all sliding windows for a document and selects the one with the
        highest confidence score.

        Returns:
            dict: Mapping {example_id: best_prediction_string}
        """
        # Dictionary to collect all candidates for each document
        # Structure: {example_id: [(score, text), ...]}
        candidates_map = defaultdict(list)

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                example_ids = batch["example_id"]  # List of document IDs

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # (Batch, Seq, NumLabels)

                # Calculate Probabilities and Predictions
                all_probs = torch.softmax(logits, dim=-1)
                all_preds = torch.argmax(all_probs, dim=-1)

                # Move to CPU for span extraction
                input_ids_cpu = input_ids.cpu().numpy()
                all_probs_cpu = all_probs.cpu().numpy()
                all_preds_cpu = all_preds.cpu().numpy()

                for idx, ex_id in enumerate(example_ids):
                    seq_preds = all_preds_cpu[idx]
                    seq_probs_full = all_probs_cpu[idx]

                    # Extract the probability of the *predicted* class for each token
                    # shape: (SeqLen,)
                    seq_confidences = np.array(
                        [seq_probs_full[t, seq_preds[t]] for t in range(len(seq_preds))]
                    )

                    # Extract spans from this window
                    spans = self.extract_spans(
                        input_ids_cpu[idx], seq_preds, seq_confidences
                    )

                    if spans:
                        candidates_map[ex_id].extend(spans)

        # Aggregation: Select best span per document
        final_preds = {}
        for ex_id, candidates in candidates_map.items():
            if not candidates:
                final_preds[ex_id] = ""
                continue

            # Sort by confidence score descending
            candidates.sort(key=lambda x: x[0], reverse=True)

            # Pick the top candidate
            final_preds[ex_id] = candidates[0][1]

        return final_preds

    def generate_submission(self, load_cached_data=True):
        """
        Orchestrates the full inference pipeline:
        1. Loads test data.
        2. Generates predictions from all seeded models.
        3. Applies Majority Vote Ensemble.
        4. Saves the submission file.
        """
        seed_everything(42)
        print("Initializing Inference Engine...")

        # 1. Load Test Data
        # shuffle=False is standard for inference
        test_loader = get_dataloader(
            mode="test",
            batch_size=Config.EVAL_BATCH_SIZE,
            shuffle=False,
            load_cached_data=load_cached_data,
        )

        # 2. Collect Predictions from Ensemble
        all_model_preds = []  # List of dicts

        for seed in Config.SEEDS:
            print(f"Running inference for Seed {seed}...")
            try:
                model = self.load_model(seed)
                preds = self.predict_single_model(model, test_loader)
                all_model_preds.append(preds)

                # Cleanup to save memory
                del model
                torch.cuda.empty_cache()
            except FileNotFoundError as e:
                print(f"Warning: {e}. Skipping this seed.")

        if not all_model_preds:
            raise RuntimeError(
                "No models were loaded successfully. Cannot generate submission."
            )

        # 3. Majority Vote Ensemble
        print("Performing Majority Vote Ensemble...")

        # Load Test Metadata to ensure we cover all IDs in the correct order
        if not os.path.exists(self.test_metadata_path):
            raise FileNotFoundError(
                f"Test metadata not found at {self.test_metadata_path}"
            )

        df_test = pd.read_csv(self.test_metadata_path)
        test_ids = df_test["id"].astype(str).tolist()

        final_submission_data = []

        for ex_id in test_ids:
            votes = []
            for model_pred_dict in all_model_preds:
                # Get prediction for this ID, default to empty string if missing
                votes.append(model_pred_dict.get(ex_id, ""))

            # Find the most common prediction
            # Counter.most_common(1) returns [(value, count)]
            counts = Counter(votes)
            best_answer, _ = counts.most_common(1)[0]

            final_submission_data.append({"id": ex_id, "PredictionString": best_answer})

        # 4. Save Submission
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df_submission = pd.DataFrame(final_submission_data)

        # Save to CSV
        # Standard CSV format handles quotes around strings if they contain delimiters.
        # The sample submission implies quoted strings, which pandas handles naturally.
        df_submission.to_csv(Config.SUBMISSION_PATH, index=False)

        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print("Head of submission:")
        print(df_submission.head())
