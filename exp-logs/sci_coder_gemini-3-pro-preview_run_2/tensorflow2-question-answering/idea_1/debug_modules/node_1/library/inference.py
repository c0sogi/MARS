import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.data_utils import build_tokenizer
from library.dataset import NQInferenceDataset, collate_fn
from library.model import BoERanker
from library.embeddings import create_embedding_matrix
from library.short_answer import TFIDFExtractor
from library.trainer import set_seed


def generate_predictions(load_cached_data=True, limit=None):
    """
    Generates predictions for the test set using the trained BoERanker and TFIDFExtractor.
    Saves the submission file to Config.SUBMISSION_SAVE_PATH.

    Args:
        load_cached_data (bool): Whether to use cached artifacts.
        limit (int, optional): Limit the number of test samples for debugging.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference using device: {device}")

    # 2. Load Artifacts needed for Model Initialization
    print("Loading tokenizer and embeddings for model initialization...")
    tokenizer = build_tokenizer(
        load_cached_data=load_cached_data,
        data_path=Config.TRAIN_DATA_PATH,  # Tokenizer built on train data
    )

    # We need the embedding matrix to initialize the model architecture correctly
    embedding_matrix = create_embedding_matrix(
        tokenizer,
        glove_path=None,
        embedding_dim=Config.EMBEDDING_DIM,
        load_cached_data=load_cached_data,
    )

    # 3. Initialize and Load Model
    print("Loading trained model...")
    model = BoERanker(embedding_matrix)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print(f"Model weights loaded from {Config.MODEL_SAVE_PATH}")
    else:
        print(
            f"Warning: Model checkpoint not found at {Config.MODEL_SAVE_PATH}. Using random weights."
        )

    model.to(device)
    model.eval()

    # 4. Initialize Short Answer Extractor
    print("Initializing TF-IDF Short Answer Extractor...")
    short_extractor = TFIDFExtractor(load_cached_data=load_cached_data)

    # 5. Prepare Test Dataset
    print("Preparing test dataset...")
    test_dataset = NQInferenceDataset(
        metadata_path=Config.TEST_META_PATH,
        data_path=Config.TEST_DATA_PATH,
        tokenizer=tokenizer,
        limit=limit,
        load_cached_data=load_cached_data,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one question (with all its candidates) at a time
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # 6. Inference Loop
    results = []
    print(f"Starting inference on {len(test_dataset)} examples...")

    with torch.no_grad():
        for batch in test_loader:
            # Unpack the single example in the batch (batch_size=1)
            example = batch[0]
            ex_id = example["example_id"]
            q_seq = torch.tensor(example["q_seq"], dtype=torch.long).to(device)
            q_text = example["q_text"]
            candidates = example["candidates"]

            long_pred_str = ""
            short_pred_str = ""

            if candidates:
                # Prepare batch of candidates for the model
                # Repeat question sequence for each candidate
                num_candidates = len(candidates)
                q_seqs = q_seq.unsqueeze(0).repeat(num_candidates, 1)

                c_seqs_list = [c["c_seq"] for c in candidates]
                c_seqs = torch.tensor(c_seqs_list, dtype=torch.long).to(device)

                # Forward pass
                scores = model(q_seqs, c_seqs)
                scores = scores.cpu().numpy()

                # Find best candidate
                best_idx = np.argmax(scores)
                max_score = scores[best_idx]
                best_candidate = candidates[best_idx]

                # --- Long Answer Logic ---
                if max_score >= Config.TAU_LONG:
                    long_pred_str = best_candidate["token_indices"]

                    # --- Short Answer Logic ---
                    # Only look for short answer if long answer is confident enough
                    # Get the raw text of the best candidate
                    cand_text = best_candidate["raw_text"]

                    # Parse start token from the token_indices string "start:end"
                    cand_start_token = int(
                        best_candidate["token_indices"].split(":")[0]
                    )

                    # Run sliding window search
                    sa_result = short_extractor.sliding_window_search(
                        q_text, cand_text, cand_start_token
                    )

                    if sa_result["score"] >= Config.TAU_SHORT:
                        # Check for YES/NO
                        yes_no = short_extractor.determine_yes_no(
                            q_text, sa_result["text"]
                        )

                        if yes_no != "NONE":
                            short_pred_str = yes_no
                        else:
                            short_pred_str = (
                                f"{sa_result['start_token']}:{sa_result['end_token']}"
                            )

            # Append results for this example
            # Format: example_id_long, prediction
            results.append(
                {"example_id": f"{ex_id}_long", "PredictionString": long_pred_str}
            )
            # Format: example_id_short, prediction
            results.append(
                {"example_id": f"{ex_id}_short", "PredictionString": short_pred_str}
            )

    # 7. Save Submission
    submission_df = pd.DataFrame(results)

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_SAVE_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_SAVE_PATH}")
    print(f"Total predictions generated: {len(submission_df)}")
