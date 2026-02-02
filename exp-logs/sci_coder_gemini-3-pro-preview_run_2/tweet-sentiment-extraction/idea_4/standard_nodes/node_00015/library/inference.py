import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.dataset import get_test_loader
from library.model import TweetModel
from library.utils import seed_everything, get_selected_text


def predict(debug=False):
    """
    Performs ensemble inference on the test set and generates the submission file.

    Args:
        debug (bool): If True, runs on a small subset of the data for debugging.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Ensure submission directory exists
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    print(f"Starting inference on device: {device}")

    # 2. Data Loading
    # Get DataLoader
    test_loader = get_test_loader(load_cached_data=True, debug=debug)

    # Load raw dataframe to get textIDs (order is preserved in loader)
    test_df = pd.read_csv(Config.TEST_FILE)
    if debug or Config.DEBUG:
        test_df = test_df.sample(
            n=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)

    num_samples = len(test_df)
    print(f"Test samples: {num_samples}")

    # 3. Ensemble Inference
    # Arrays to store averaged logits
    # Shape: (num_samples, max_len)
    avg_start_logits = np.zeros((num_samples, Config.MAX_LEN), dtype=np.float32)
    avg_end_logits = np.zeros((num_samples, Config.MAX_LEN), dtype=np.float32)

    for fold in range(Config.N_FOLDS):
        print(f"Processing Fold {fold}/{Config.N_FOLDS - 1}...")

        # Load Model
        model = TweetModel(Config)
        model_path = os.path.join(Config.MODEL_OUTPUT_DIR, f"model_fold_{fold}.pth")

        if not os.path.exists(model_path):
            print(
                f"Warning: Model for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # Inference Loop for this fold
        fold_start_logits = []
        fold_end_logits = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)

                start_logits, end_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )

                fold_start_logits.append(start_logits.cpu().numpy())
                fold_end_logits.append(end_logits.cpu().numpy())

        # Concatenate and accumulate
        fold_start_logits = np.concatenate(fold_start_logits, axis=0)
        fold_end_logits = np.concatenate(fold_end_logits, axis=0)

        avg_start_logits += fold_start_logits
        avg_end_logits += fold_end_logits

        # Cleanup to save memory
        del model, state_dict, fold_start_logits, fold_end_logits
        torch.cuda.empty_cache()

    # Average the logits
    avg_start_logits /= Config.N_FOLDS
    avg_end_logits /= Config.N_FOLDS

    # 4. Decoding
    print("Decoding predictions...")
    predictions = []

    # We iterate the loader again to align metadata with predictions
    # This is memory efficient and ensures we have the exact text/offsets used during tokenization
    sample_idx = 0

    # Pre-calculate upper triangle mask for decoding
    # Mask out the lower triangle (where start > end)
    # shape (max_len, max_len)
    ones = np.ones((Config.MAX_LEN, Config.MAX_LEN))
    triu_mask = np.triu(ones, k=0)

    for batch in test_loader:
        batch_size = batch["input_ids"].size(0)

        # Get corresponding logits slice
        batch_start_logits = avg_start_logits[sample_idx : sample_idx + batch_size]
        batch_end_logits = avg_end_logits[sample_idx : sample_idx + batch_size]

        # Metadata
        texts = batch["text"]
        sentiments = batch["sentiment"]
        offsets = batch["offsets"].numpy()  # Convert tensor to numpy for indexing

        for i in range(batch_size):
            text = texts[i]
            sentiment = sentiments[i]
            offset = offsets[i]

            start_logits = batch_start_logits[i]
            end_logits = batch_end_logits[i]

            # Neutral Heuristic
            if sentiment == "neutral" and Config.NEUTRAL_FULL_TEXT:
                pred_text = text
            else:
                # Score = start_logit + end_logit
                # Shape: (seq_len, seq_len)
                sum_matrix = start_logits[:, None] + end_logits[None, :]

                # Apply mask: set invalid positions (start > end) to very low value
                sum_matrix = sum_matrix * triu_mask + (1 - triu_mask) * -1e9

                # Find best span
                flat_idx = np.argmax(sum_matrix)
                start_idx = flat_idx // Config.MAX_LEN
                end_idx = flat_idx % Config.MAX_LEN

                pred_text = get_selected_text(text, start_idx, end_idx, offset)

            predictions.append(pred_text)

        sample_idx += batch_size

    # 5. Submission
    print(f"Generating submission file at {submission_path}...")

    submission_df = pd.DataFrame(
        {"textID": test_df["textID"], "selected_text": predictions}
    )

    # Ensure correct format
    submission_df.to_csv(submission_path, index=False)

    print("Inference complete.")
