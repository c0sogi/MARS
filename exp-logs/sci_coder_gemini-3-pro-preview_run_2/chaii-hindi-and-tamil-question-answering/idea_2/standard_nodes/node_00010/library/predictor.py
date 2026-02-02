import os
import collections
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.model_factory import get_model


class InferenceEngine:
    """
    Manages the inference process using an ensemble of trained models.
    Performs model loading, batched inference, logit averaging, and
    post-processing to map token logits back to text spans.
    """

    def __init__(self, device=None):
        self.device = device if device else Config.device
        self.models = []

    def load_ensemble(self, num_folds=Config.n_folds):
        """
        Loads the ensemble of K models from the checkpoints saved during training.

        Args:
            num_folds (int): Number of folds/models to load.
        """
        self.models = []
        print(f"Loading ensemble of {num_folds} models from {Config.output_dir}...")

        for fold_idx in range(num_folds):
            checkpoint_path = os.path.join(
                Config.output_dir, f"best_model_fold_{fold_idx}.pth"
            )

            if not os.path.exists(checkpoint_path):
                print(
                    f"Warning: Checkpoint {checkpoint_path} not found. Skipping fold."
                )
                continue

            try:
                # Initialize model architecture
                model = get_model()
                model.to(self.device)

                # Load weights
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.eval()

                self.models.append(model)
                print(f"Successfully loaded model for fold {fold_idx}")
            except Exception as e:
                print(f"Error loading fold {fold_idx}: {e}")

        if not self.models:
            print(
                "Error: No models were loaded. Initializing a random model as fallback."
            )
            # Fallback to avoid crash, though predictions will be random
            model = get_model()
            model.to(self.device)
            model.eval()
            self.models.append(model)

    def predict(self, test_loader, test_features_df, raw_test_df):
        """
        Runs inference on the test set using the loaded ensemble.
        Averages logits across all models before decoding.

        Args:
            test_loader: DataLoader containing test features.
            test_features_df: DataFrame with feature metadata (offset mappings).
            raw_test_df: DataFrame with raw context text.

        Returns:
            dict: A dictionary mapping example IDs to predicted answer strings.
        """
        if not self.models:
            self.load_ensemble()

        print("Starting ensemble inference...")
        all_start_logits = []
        all_end_logits = []

        # 1. Batched Inference Loop
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Accumulators for ensemble averaging
                batch_start_sum = None
                batch_end_sum = None

                for model in self.models:
                    # Mixed precision inference
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = model(
                            input_ids=input_ids, attention_mask=attention_mask
                        )

                    start = outputs.start_logits.detach().cpu().numpy()
                    end = outputs.end_logits.detach().cpu().numpy()

                    if batch_start_sum is None:
                        batch_start_sum = start
                        batch_end_sum = end
                    else:
                        batch_start_sum += start
                        batch_end_sum += end

                # Average logits
                num_models = len(self.models)
                avg_start = batch_start_sum / num_models
                avg_end = batch_end_sum / num_models

                all_start_logits.append(avg_start)
                all_end_logits.append(avg_end)

        if not all_start_logits:
            return {}

        # Concatenate results from all batches
        all_start_logits = np.concatenate(all_start_logits, axis=0)
        all_end_logits = np.concatenate(all_end_logits, axis=0)

        # 2. Post-processing
        return self._postprocess(
            all_start_logits, all_end_logits, test_features_df, raw_test_df
        )

    def _postprocess(self, start_logits_all, end_logits_all, features_df, raw_df):
        """
        Maps token-level logits back to text spans using offset mappings.
        Aggregates results from sliding windows for the same example.
        """
        print("Post-processing predictions...")

        # Map example_id to feature indices (handling sliding windows)
        example_to_features = collections.defaultdict(list)
        for idx, row in features_df.iterrows():
            example_to_features[row["example_id"]].append(idx)

        # Map raw ID to Context text for extraction
        raw_context_map = raw_df.set_index("id")["context"].to_dict()

        predictions = {}

        # Iterate over each example in the raw dataset
        for _, row in raw_df.iterrows():
            ex_id = row["id"]

            if ex_id not in example_to_features:
                predictions[ex_id] = ""
                continue

            feature_indices = example_to_features[ex_id]
            context_text = raw_context_map.get(ex_id, "")

            best_score = float("-inf")
            best_answer = ""

            # Search for best span across all chunks for this example
            for feat_idx in feature_indices:
                start_logits = start_logits_all[feat_idx]
                end_logits = end_logits_all[feat_idx]
                offsets = features_df.iloc[feat_idx]["offset_mapping"]

                # Get top-k start and end indices to reduce search space
                start_indexes = np.argsort(start_logits)[
                    -1 : -Config.n_best_size - 1 : -1
                ].tolist()
                end_indexes = np.argsort(end_logits)[
                    -1 : -Config.n_best_size - 1 : -1
                ].tolist()

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Validity checks
                        if start_index >= len(offsets) or end_index >= len(offsets):
                            continue
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        if end_index < start_index:
                            continue
                        if end_index - start_index + 1 > Config.max_answer_length:
                            continue

                        # Score is sum of start and end logits
                        score = start_logits[start_index] + end_logits[end_index]

                        if score > best_score:
                            best_score = score
                            try:
                                # Extract text using character offsets
                                start_char = offsets[start_index][0]
                                end_char = offsets[end_index][1]
                                best_answer = context_text[start_char:end_char]
                            except Exception:
                                pass

            predictions[ex_id] = best_answer

        return predictions


def generate_submission(predictions, output_path="./submission/submission.csv"):
    """
    Generates the final submission CSV file.

    Args:
        predictions (dict): Dictionary mapping IDs to prediction strings.
        output_path (str): Path to save the CSV.
    """
    print(f"Generating submission file at {output_path}...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert to DataFrame
    data = [{"id": k, "PredictionString": v} for k, v in predictions.items()]
    df = pd.DataFrame(data)

    # Save to CSV
    # Pandas handles quoting automatically for strings containing delimiters
    df.to_csv(output_path, index=False)
    print("Submission file generated successfully.")
