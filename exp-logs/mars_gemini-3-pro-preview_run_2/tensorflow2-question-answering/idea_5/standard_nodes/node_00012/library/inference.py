import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.data_processing import DataProcessor
from library.dataset import NQDataset
from library.model import FeedForwardDecomposableAttention


class InferenceManager:
    """
    Manages the inference process for the Natural Questions dataset.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)

        # Initialize Processor and load vocab/embeddings
        self.processor = DataProcessor(config)
        self.vocab = self.processor.build_vocab(load_cached_data=True)
        self.embedding_matrix = self.processor.create_embedding_matrix(
            load_cached_data=True
        )

        # Convert embedding matrix to tensor
        self.embedding_tensor = torch.tensor(self.embedding_matrix, dtype=torch.float32)

        # Initialize Model
        self.model = FeedForwardDecomposableAttention(config, self.embedding_tensor)
        self.model.to(self.device)
        self.model.eval()

        # Load Checkpoint
        if os.path.exists(config.MODEL_CHECKPOINT_PATH):
            load_checkpoint(
                config.MODEL_CHECKPOINT_PATH, self.model, device=config.DEVICE
            )
        else:
            print(
                f"Warning: No checkpoint found at {config.MODEL_CHECKPOINT_PATH}. Using random weights."
            )

        # Yes/No Map (Inverse of what is in Dataset)
        self.idx_to_yn = {0: "YES", 1: "NO", 2: "NONE"}

    def generate_predictions(self, debug_sample_size=None):
        """
        Generates predictions for the test set and saves them to the submission file.

        Args:
            debug_sample_size (int, optional): Override config to limit test set size.
        """
        # Override config for debugging if provided
        original_debug_size = self.config.DEBUG_SAMPLE_SIZE
        if debug_sample_size is not None:
            self.config.DEBUG_SAMPLE_SIZE = debug_sample_size

        # Prepare Dataset and DataLoader
        # Batch size 1 ensures we get all candidates for exactly one example per iteration
        # Cite debug_lesson_2: Explicitly invalidate cache to ensure data matches config
        test_dataset = NQDataset(
            self.config, self.processor, split="test", load_cached_data=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=NQDataset.collate_fn,
            num_workers=0,
        )

        results = []

        print(f"Starting inference on {len(test_dataset)} examples...")

        with torch.no_grad():
            for batch in test_loader:
                if not batch:
                    continue

                # Move inputs to device
                q_input = batch["q_input"].to(self.device)
                c_input = batch["c_input"].to(self.device)

                # Metadata for reconstruction
                example_ids = batch["example_ids"]
                cand_starts = batch["cand_starts"]
                cand_ends = batch["cand_ends"]

                # All candidates in this batch belong to the same example_id (due to batch_size=1)
                current_example_id = example_ids[0]

                # Model Forward Pass
                outputs = self.model(q_input, c_input)

                # 1. Ranking Scores
                ranking_logits = outputs["ranking_logits"].squeeze(
                    -1
                )  # (Num_Candidates,)
                ranking_scores = torch.sigmoid(ranking_logits)

                # Find best candidate
                best_score, best_idx = torch.max(ranking_scores, dim=0)
                best_idx = best_idx.item()
                best_score = best_score.item()

                # Initialize prediction strings
                long_pred_str = ""
                short_pred_str = ""

                # Apply Threshold
                if best_score >= self.config.CONFIDENCE_THRESHOLD:
                    # --- Long Answer Prediction ---
                    # Get token span from document
                    l_start = cand_starts[best_idx]
                    l_end = cand_ends[best_idx]
                    long_pred_str = f"{l_start}:{l_end}"

                    # --- Short Answer / Yes-No Prediction ---
                    # Check Yes/No first
                    yn_logits = outputs["yn_logits"][best_idx]  # (3,)
                    yn_idx = torch.argmax(yn_logits).item()
                    yn_label = self.idx_to_yn.get(yn_idx, "NONE")

                    if yn_label in ["YES", "NO"]:
                        short_pred_str = yn_label
                    else:
                        # Extract Span
                        start_logits = outputs["start_logits"][best_idx]  # (C_Len,)
                        end_logits = outputs["end_logits"][best_idx]  # (C_Len,)

                        s_rel = torch.argmax(start_logits).item()
                        e_rel = torch.argmax(end_logits).item()

                        # Validate span
                        if s_rel <= e_rel:
                            # Convert relative to absolute document indices
                            s_abs = l_start + s_rel
                            e_abs = (
                                l_start + e_rel + 1
                            )  # +1 because output format usually expects exclusive end or just consistency
                            # However, NQ evaluation often expects token indices.
                            # Based on sample submission "6:18", it implies tokens 6 to 17.
                            # Dataset logic: s_end_rel was `rel_e - 1`. So `e_rel` is inclusive index.
                            # We need to output `start:end` where end is usually non-inclusive in Python slicing,
                            # but NQ format is `start:end` tokens.
                            # Let's stick to standard `start:end` meaning tokens[start:end].
                            # If e_rel is inclusive index, then end token index is s_abs + e_rel + 1.

                            short_pred_str = f"{s_abs}:{l_start + e_rel + 1}"
                        else:
                            # Invalid span (start > end), fallback to blank or long answer?
                            # Strategy: Leave blank if model is confused about span
                            short_pred_str = ""

                # Append results
                results.append(
                    {
                        "example_id": f"{current_example_id}_long",
                        "PredictionString": long_pred_str,
                    }
                )
                results.append(
                    {
                        "example_id": f"{current_example_id}_short",
                        "PredictionString": short_pred_str,
                    }
                )

        # Restore config
        if debug_sample_size is not None:
            self.config.DEBUG_SAMPLE_SIZE = original_debug_size

        # Create DataFrame
        submission_df = pd.DataFrame(results)
        return submission_df

    def save_submission(self, df):
        """
        Saves the prediction DataFrame to CSV.
        """
        save_path = self.config.SUBMISSION_PATH
        print(f"Saving submission to {save_path}...")
        df.to_csv(save_path, index=False)
        print("Submission saved successfully.")


def main():
    # Setup
    config = Config()
    set_seed(config.SEED)

    # Initialize Manager
    manager = InferenceManager(config)

    # Run Inference
    # We use the config's DEBUG_SAMPLE_SIZE if set, otherwise full inference
    submission_df = manager.generate_predictions()

    # Save
    manager.save_submission(submission_df)


if __name__ == "__main__":
    main()
