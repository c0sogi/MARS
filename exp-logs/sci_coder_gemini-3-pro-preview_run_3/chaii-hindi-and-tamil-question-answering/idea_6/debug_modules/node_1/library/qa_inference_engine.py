import os
import torch
import pandas as pd
import numpy as np
import csv
from collections import defaultdict, Counter
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

from library.configuration import Config
from library.utilities import set_seed
from library.qa_data_processing import get_qa_data


class QAInferenceEngine:
    """
    Manages prediction generation and ensembling for the Hindi/Tamil QA task.
    Encapsulates logic for span extraction, confidence scoring, and model inference.
    """

    @staticmethod
    def decode_span(tokenizer, token_ids):
        """
        Decodes a sequence of token IDs into a string using the tokenizer.
        Ensures special tokens are skipped and whitespace is trimmed.

        Args:
            tokenizer: The HuggingFace tokenizer.
            token_ids (list or np.array): The sequence of token IDs.

        Returns:
            str: The decoded string.
        """
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        return text.strip()

    @staticmethod
    def extract_candidates_from_batch(input_ids, logits, example_ids, tokenizer):
        """
        Extracts candidate answer spans from a batch of model predictions.
        Implements the logic to convert BIO tags into text spans with confidence scores.

        Args:
            input_ids (np.array): Input token IDs (Batch, Seq_Len).
            logits (np.array): Model logits (Batch, Seq_Len, 3).
            example_ids (list): List of document IDs corresponding to the batch.
            tokenizer: HuggingFace tokenizer.

        Returns:
            list: A list of dictionaries, each containing {example_id, text, score}.
        """
        # Convert logits to probabilities and predictions
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        # Move to CPU for processing
        probs = probs.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        input_ids = input_ids.detach().cpu().numpy()

        batch_candidates = []

        for i, ex_id in enumerate(example_ids):
            tokens = input_ids[i]
            sample_preds = preds[i]
            sample_probs = probs[i]

            current_span_start = None
            current_span_score = 0.0
            current_span_tokens = []

            # Iterate through the sequence to find spans
            # Labels: 0=O, 1=B-ANS, 2=I-ANS
            for t_idx, label in enumerate(sample_preds):
                if label == 1:  # B-ANS (Start of answer)
                    # If a span was already open, close it and save
                    if current_span_start is not None:
                        score = current_span_score / len(current_span_tokens)
                        text = QAInferenceEngine.decode_span(
                            tokenizer, tokens[current_span_start:t_idx]
                        )
                        if text:
                            batch_candidates.append(
                                {"example_id": ex_id, "text": text, "score": score}
                            )

                    # Start a new span
                    current_span_start = t_idx
                    current_span_score = sample_probs[t_idx][1]
                    current_span_tokens = [t_idx]

                elif label == 2:  # I-ANS (Inside answer)
                    if current_span_start is not None:
                        # Continue the current span
                        current_span_score += sample_probs[t_idx][2]
                        current_span_tokens.append(t_idx)
                    else:
                        # I-ANS without preceding B-ANS is ignored in strict mode
                        pass

                else:  # O (Outside) or Special Tokens
                    if current_span_start is not None:
                        # Close the current span
                        score = current_span_score / len(current_span_tokens)
                        text = QAInferenceEngine.decode_span(
                            tokenizer, tokens[current_span_start:t_idx]
                        )
                        if text:
                            batch_candidates.append(
                                {"example_id": ex_id, "text": text, "score": score}
                            )
                        current_span_start = None
                        current_span_score = 0.0
                        current_span_tokens = []

            # Handle case where span continues to the end of the sequence
            if current_span_start is not None:
                score = current_span_score / len(current_span_tokens)
                text = QAInferenceEngine.decode_span(
                    tokenizer, tokens[current_span_start:]
                )
                if text:
                    batch_candidates.append(
                        {"example_id": ex_id, "text": text, "score": score}
                    )

        return batch_candidates

    @staticmethod
    def predict_single_model(seed, test_ds, tokenizer):
        """
        Runs inference for a single model seed.
        Aggregates predictions from multiple sliding windows for each document
        and selects the span with the highest global confidence score.

        Args:
            seed (int): The random seed identifying the model checkpoint.
            test_ds (Dataset): The test dataset.
            tokenizer: The tokenizer used for decoding.

        Returns:
            dict: A mapping of {example_id: predicted_string}.
        """
        model_path = os.path.join(Config.QA_OUTPUT_DIR, f"model_seed_{seed}.pt")
        if not os.path.exists(model_path):
            print(
                f"Warning: Model for seed {seed} not found at {model_path}. Skipping."
            )
            return {}

        print(f"Loading model for seed {seed}...")

        # Determine configuration path (prefer TAPT if available)
        if os.path.exists(os.path.join(Config.TAPT_OUTPUT_DIR, "config.json")):
            config_path = Config.TAPT_OUTPUT_DIR
        else:
            config_path = Config.MODEL_CHECKPOINT

        # Load Model
        model = AutoModelForTokenClassification.from_pretrained(
            config_path, num_labels=3
        )
        state_dict = torch.load(model_path, map_location=Config.DEVICE)
        model.load_state_dict(state_dict)
        model.to(Config.DEVICE)
        model.eval()

        # Custom collate function to handle non-tensor fields (example_ids)
        def inference_collate(batch):
            input_ids = torch.stack([item["input_ids"] for item in batch])
            attention_mask = torch.stack([item["attention_mask"] for item in batch])
            example_ids = [item["example_id"] for item in batch]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "example_ids": example_ids,
            }

        loader = DataLoader(
            test_ds,
            batch_size=Config.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=inference_collate,
        )

        # Store all candidates for each document ID
        candidates_map = defaultdict(list)

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(Config.DEVICE)
                attention_mask = batch["attention_mask"].to(Config.DEVICE)
                example_ids = batch["example_ids"]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Extract spans
                batch_cands = QAInferenceEngine.extract_candidates_from_batch(
                    input_ids, logits, example_ids, tokenizer
                )

                # Aggregate
                for cand in batch_cands:
                    candidates_map[cand["example_id"]].append(
                        (cand["score"], cand["text"])
                    )

        # Select best candidate per ID (Global Confidence Aggregation)
        predictions = {}
        all_ids = set(test_ds.data["example_id"].unique())

        for eid in all_ids:
            cands = candidates_map.get(eid, [])
            if not cands:
                # Fallback if no spans detected (rare)
                predictions[eid] = ""
            else:
                # Sort by confidence score descending and pick top 1
                cands.sort(key=lambda x: x[0], reverse=True)
                predictions[eid] = cands[0][1]

        return predictions

    @staticmethod
    def majority_vote(predictions_list):
        """
        Performs majority voting on a list of prediction dictionaries.

        Args:
            predictions_list (list): List of dicts {id: prediction}.

        Returns:
            dict: The final consensus predictions {id: prediction}.
        """
        if not predictions_list:
            return {}

        final_preds = {}
        # Get all unique IDs
        all_ids = set().union(*[d.keys() for d in predictions_list])

        for eid in all_ids:
            votes = [d.get(eid, "") for d in predictions_list]
            # Find the most common string
            most_common = Counter(votes).most_common(1)[0][0]
            final_preds[eid] = most_common

        return final_preds


def generate_submission():
    """
    Main execution function to generate the submission file.
    Orchestrates data loading, multi-seed inference, ensembling, and file saving.
    """
    # 1. Load Data
    # get_qa_data handles caching. We only need the test dataset here.
    _, _, test_ds = get_qa_data(load_cached_data=True)

    # 2. Load Tokenizer
    # Use TAPT tokenizer if available for consistent decoding
    if os.path.exists(Config.TAPT_OUTPUT_DIR):
        print("Using TAPT tokenizer for inference.")
        tokenizer = AutoTokenizer.from_pretrained(Config.TAPT_OUTPUT_DIR)
    else:
        print("Using base tokenizer for inference.")
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)

    # 3. Generate Predictions for each Seed
    all_predictions = []
    for seed in Config.SEEDS:
        print(f"--- Generating predictions for Seed {seed} ---")
        preds = QAInferenceEngine.predict_single_model(seed, test_ds, tokenizer)
        if preds:
            all_predictions.append(preds)

    if not all_predictions:
        print("Error: No predictions generated. Please check model training.")
        return

    # 4. Ensemble Predictions (Majority Vote)
    print("Ensembling predictions via Majority Vote...")
    final_predictions = QAInferenceEngine.majority_vote(all_predictions)

    # 5. Save Submission File
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")

    # Prepare DataFrame
    submission_data = [
        {"id": eid, "PredictionString": pred} for eid, pred in final_predictions.items()
    ]
    df_sub = pd.DataFrame(submission_data)

    # Ensure correct column order
    df_sub = df_sub[["id", "PredictionString"]]

    # Save to CSV
    # Using QUOTE_NONNUMERIC to ensure string fields are quoted, matching the sample format.
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False, quoting=csv.QUOTE_NONNUMERIC)

    print("Submission generation complete.")
