import os
import csv
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.modeling import TweetModel
from library.data_factory import get_loaders
from library.utils import seed_everything


def generate_submission(device=Config.DEVICE, load_cached_data=True):
    """
    Orchestrates the ensemble inference process.
    Loads multiple heterogeneous models, aggregates their predictions at the character level,
    decodes the optimal spans, and generates the submission file.
    """
    seed_everything(Config.SEED)

    print("Initializing Inference Engine...")
    print(f"Device: {device}")
    print(f"Ensemble Architectures: {Config.MODEL_BACKBONES}")

    # 1. Initialize Global Accumulators
    # We need to store character-level probabilities for every sample in the test set.
    # Load raw test metadata to get IDs and Texts
    df_test = pd.read_csv(Config.TEST_META_PATH)

    # Dictionary: textID -> {'start': np.array, 'end': np.array, 'text': str, 'sentiment': str}
    ensemble_results = {}
    for _, row in df_test.iterrows():
        text_content = str(row["text"])
        # Handle potential nan/empty strings gracefully
        if text_content == "nan":
            text_content = ""

        t_len = len(text_content)
        ensemble_results[row["textID"]] = {
            "start": np.zeros(t_len, dtype=np.float32),
            "end": np.zeros(t_len, dtype=np.float32),
            "text": text_content,
            "sentiment": str(row["sentiment"]),
        }

    # 2. Iterate over Architectures (Heterogeneous Ensemble)
    for model_name in Config.MODEL_BACKBONES:
        safe_name = model_name.replace("/", "_")
        print(f"\n--- Processing Architecture: {model_name} ---")

        # Get DataLoader specific to this tokenizer
        # We use batch_size from Config (Validation size is appropriate for inference)
        _, _, test_loader = get_loaders(
            model_name,
            batch_size=Config.VALID_BATCH_SIZE,
            load_cached_data=load_cached_data,
        )

        # 3. Iterate over Folds
        for fold in range(Config.N_FOLDS):
            # Construct model path
            model_filename = f"{safe_name}_fold_{fold}.pth"
            model_path = os.path.join(Config.ARTIFACTS_DIR, model_filename)

            if not os.path.exists(model_path):
                print(f"Warning: Model file {model_filename} not found. Skipping.")
                continue

            print(f"Loading Model: {model_filename}")

            # Load Model
            model = TweetModel(model_name)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            # Inference Loop
            with torch.no_grad():
                for batch in test_loader:
                    ids = batch["ids"].to(device)
                    mask = batch["mask"].to(device)
                    tt_ids = batch["token_type_ids"].to(device)
                    text_ids = batch["textID"]
                    offsets = batch["offsets"].numpy()  # Shape: (B, L, 2)

                    # Forward Pass
                    start_logits, end_logits = model(ids, mask, tt_ids)

                    # Mask padding tokens to -inf before softmax
                    # mask is 1 for tokens, 0 for pads
                    active_mask = mask.bool()
                    start_logits[~active_mask] = -1e9
                    end_logits[~active_mask] = -1e9

                    # Compute Probabilities
                    start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
                    end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

                    # Map Token Probabilities to Character Probabilities
                    for i, tid in enumerate(text_ids):
                        if tid not in ensemble_results:
                            continue

                        res = ensemble_results[tid]
                        txt_len = len(res["text"])

                        # Current sample predictions
                        s_p = start_probs[i]
                        e_p = end_probs[i]
                        offs = offsets[i]

                        # Iterate through tokens and accumulate probs to chars
                        for tok_idx, (start_char, end_char) in enumerate(offs):
                            # Skip special tokens/padding (usually 0,0)
                            # Also ensure we don't go out of bounds
                            if start_char == 0 and end_char == 0:
                                continue

                            # Accumulate Start Probability
                            if start_char < txt_len:
                                res["start"][start_char] += s_p[tok_idx]

                            # Accumulate End Probability
                            # end_char from offset is exclusive upper bound.
                            # The target character index is end_char - 1.
                            target_end_idx = end_char - 1
                            if target_end_idx >= 0 and target_end_idx < txt_len:
                                res["end"][target_end_idx] += e_p[tok_idx]

            # Cleanup to save memory
            del model
            torch.cuda.empty_cache()

    # 4. Decode and Generate Predictions
    print("\nDecoding ensemble predictions...")
    final_predictions = []

    for tid, data in ensemble_results.items():
        text = data["text"]
        sentiment = data["sentiment"]

        # Neutral Heuristic: Always predict full text
        if sentiment == "neutral" or len(text) == 0:
            pred_text = text
        else:
            # Decode using accumulated probabilities
            s_probs = data["start"]
            e_probs = data["end"]

            # We want to find (i, j) such that i <= j and (s_probs[i] + e_probs[j]) is maximized.
            # Create probability matrices
            S = s_probs[:, None]  # Column vector
            E = e_probs[None, :]  # Row vector

            # Sum matrix: Score[i, j] = P(start=i) + P(end=j)
            Score = S + E

            # Mask invalid positions (where start > end)
            # np.triu keeps the upper triangle (including diagonal), setting others to 0
            # We want to keep i <= j, which corresponds to the upper triangle.
            # We set lower triangle (invalid) to a very small number.
            tril_mask = np.triu(np.ones((len(text), len(text))), k=0)

            # Apply mask
            Score = Score * tril_mask
            Score[tril_mask == 0] = -1e9

            # Find indices of maximum score
            flat_idx = np.argmax(Score)
            best_start, best_end = np.unravel_index(flat_idx, Score.shape)

            # Extract substring
            # best_end is the index of the last character, so slice needs +1
            pred_text = text[best_start : best_end + 1]

        final_predictions.append({"textID": tid, "selected_text": pred_text})

    # 5. Save Submission
    df_sub = pd.DataFrame(final_predictions)

    # Ensure submission directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    # Save with quoting to handle special characters/commas in text
    # QUOTE_NONNUMERIC will quote all non-numeric fields (i.e., the text)
    df_sub.to_csv(Config.SUBMISSION_FILE, index=False, quoting=csv.QUOTE_NONNUMERIC)

    print(f"Submission successfully saved to {Config.SUBMISSION_FILE}")
    print(f"Total predictions: {len(df_sub)}")
