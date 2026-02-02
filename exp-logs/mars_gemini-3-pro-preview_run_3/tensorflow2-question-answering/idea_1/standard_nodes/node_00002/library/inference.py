import os
import torch
import pandas as pd
from library.config import Config
from library.models import SiameseDANRanker, ShallowCNNReader
from library.data_loader import get_dataloaders


class NQPipeline:
    """
    Manages the evaluation and prediction pipeline for Natural Questions.
    Loads trained models, processes test data, ranks candidates, extracts answers,
    and generates the final submission file.
    """

    def __init__(self, debug_sample_size=None):
        """
        Initializes the pipeline.

        Args:
            debug_sample_size (int, optional): If provided, limits the test set size for debugging.
        """
        self.device = Config.get_device()
        print(f"Initializing NQPipeline on device: {self.device}")

        # Load DataLoaders and Vocabulary
        # This utilizes the provided library function which handles vocab building and dataset creation.
        print("Loading test data and vocabulary...")
        self.loaders, self.vocab_encoder = get_dataloaders(
            debug_sample_size=debug_sample_size
        )
        self.test_loader = self.loaders["test"]

        # Model parameters based on vocabulary
        vocab_size = len(self.vocab_encoder)
        pad_idx = self.vocab_encoder.pad_idx

        # Initialize Models
        print("Initializing models...")
        self.ranker = SiameseDANRanker(vocab_size, padding_idx=pad_idx).to(self.device)
        self.reader = ShallowCNNReader(vocab_size, padding_idx=pad_idx).to(self.device)

        # Load pre-trained weights
        self._load_models()

    def _load_models(self):
        """
        Loads the best model checkpoints from the cache directory.
        If checkpoints are missing, proceeds with random weights (warns user).
        """
        ranker_path = os.path.join(Config.CACHE_DIR, "ranker_best.pth")
        reader_path = os.path.join(Config.CACHE_DIR, "reader_best.pth")

        if os.path.exists(ranker_path):
            print(f"Loading Ranker weights from {ranker_path}")
            self.ranker.load_state_dict(
                torch.load(ranker_path, map_location=self.device)
            )
        else:
            print(
                f"Warning: Ranker weights not found at {ranker_path}. Using random weights."
            )

        if os.path.exists(reader_path):
            print(f"Loading Reader weights from {reader_path}")
            self.reader.load_state_dict(
                torch.load(reader_path, map_location=self.device)
            )
        else:
            print(
                f"Warning: Reader weights not found at {reader_path}. Using random weights."
            )

        # Set models to evaluation mode
        self.ranker.eval()
        self.reader.eval()

    def generate_predictions(self):
        """
        Runs the inference loop on the test set.
        Ranks candidates, extracts short answers, filters by confidence, and saves results.
        """
        print("\n--- Generating Predictions ---")
        results = []

        # Special tokens for input construction
        sep_token_id = self.vocab_encoder.unk_idx
        pad_token_id = self.vocab_encoder.pad_idx

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                # Unpack batch (batch_size=1, collate_fn returns the single item tuple)
                example_id, q_ids, cand_tensors, cand_meta = batch

                # Add batch dimension: (1, Seq_Len)
                q_ids = q_ids.to(self.device).unsqueeze(0)
                # Add batch dimension: (1, Num_Candidates, Ctx_Len)
                cand_tensors = cand_tensors.to(self.device).unsqueeze(0)

                # --- Step 1: Rank Candidates ---
                # SiameseDANRanker handles 3D inputs for candidates
                scores = self.ranker(q_ids, cand_tensors)  # Output shape: (1, K)
                scores = scores.squeeze(0)  # Shape: (K,)

                # Handle edge case where document has no candidates
                if scores.numel() == 0:
                    results.append(f"{example_id}_long,")
                    results.append(f"{example_id}_short,")
                    continue

                # Identify best candidate
                best_score, best_idx = torch.max(scores, dim=0)
                best_idx = best_idx.item()
                best_score = best_score.item()

                # --- Step 2: Apply Thresholding ---
                if best_score < Config.CONFIDENCE_THRESHOLD:
                    # Score too low, predict NULL
                    results.append(f"{example_id}_long,")
                    results.append(f"{example_id}_short,")
                    continue

                # --- Step 3: Format Long Answer ---
                best_cand_info = cand_meta[best_idx]
                # Format: start_token:end_token
                long_ans_str = f"{best_cand_info['start_token_idx']}:{best_cand_info['end_token_idx']}"
                results.append(f"{example_id}_long,{long_ans_str}")

                # --- Step 4: Extract Short Answer ---
                # Prepare input for Reader: [Q, SEP, Context]
                # We extract valid tokens (non-padding) to concatenate cleanly
                q_raw = q_ids[0]
                ctx_raw = cand_tensors[0, best_idx]

                q_valid = q_raw[q_raw != pad_token_id]
                ctx_valid = ctx_raw[ctx_raw != pad_token_id]

                sep_tensor = torch.tensor([sep_token_id], device=self.device)

                # Concatenate: (1, Seq_Len)
                reader_input = torch.cat([q_valid, sep_tensor, ctx_valid]).unsqueeze(0)

                # Forward pass through Reader
                start_logits, end_logits = self.reader(reader_input)

                # Get predicted local indices
                start_pred = torch.argmax(start_logits, dim=1).item()
                end_pred = torch.argmax(end_logits, dim=1).item()

                # Map local indices back to global document indices
                # The offset is the length of the Question + Separator
                offset = len(q_valid) + 1

                local_start = start_pred - offset
                local_end = end_pred - offset

                # Validate Span Logic
                # 1. Start must be >= 0 (answer is in context, not question)
                # 2. End must be >= Start
                # 3. End must be within the bounds of the candidate context
                if (
                    local_start >= 0
                    and local_end >= local_start
                    and local_end < len(ctx_valid)
                ):

                    global_start = best_cand_info["start_token_idx"] + local_start
                    global_end = best_cand_info["start_token_idx"] + local_end

                    short_ans_str = f"{global_start}:{global_end}"
                    results.append(f"{example_id}_short,{short_ans_str}")
                else:
                    # Prediction invalid or pointed to question area -> Predict NULL
                    results.append(f"{example_id}_short,")

        # Save results to disk
        self._save_results(results)

    def _save_results(self, results):
        """
        Writes the list of result strings to the submission CSV file.
        """
        print(f"Saving {len(results)} predictions to {Config.SUBMISSION_PATH}...")
        try:
            with open(Config.SUBMISSION_PATH, "w") as f:
                f.write("example_id,PredictionString\n")
                f.write("\n".join(results))
            print("Submission saved successfully.")
        except Exception as e:
            print(f"Error saving submission: {e}")
