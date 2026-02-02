import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from library.data import get_test_dl
from library.model import TweetModel
from library.utils import seed_everything


def get_best_start_end_idxs(start_logits, end_logits):
    """
    Decodes the start and end indices by maximizing the joint probability.

    Args:
        start_logits (torch.Tensor): Logits for start position (seq_len,).
        end_logits (torch.Tensor): Logits for end position (seq_len,).

    Returns:
        tuple: (best_start_index, best_end_index)
    """
    # Compute probabilities
    start_probs = F.softmax(start_logits, dim=0).cpu().numpy()
    end_probs = F.softmax(end_logits, dim=0).cpu().numpy()

    # Compute joint probability matrix: P(start, end) = P(start) * P(end)
    scores = np.outer(start_probs, end_probs)

    # Mask invalid predictions where end < start (lower triangle)
    # np.triu returns the upper triangle of the matrix, setting lower to 0
    scores = np.triu(scores)

    # Find the indices of the maximum score
    best_idx = np.argmax(scores)
    best_s, best_e = np.unravel_index(best_idx, scores.shape)

    return best_s, best_e


def predict_test(config):
    """
    Runs inference on the test set using an ensemble of trained models.
    Generates the submission file.

    Args:
        config (Config): Configuration object containing paths and parameters.
    """
    # Ensure reproducibility
    seed_everything(config.seed)

    print("--- Starting Inference ---")
    device = config.device

    # Prepare output directory
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Get Test DataLoader
    # This handles data processing/caching internally via library.data
    test_dl = get_test_dl(config)

    # Load Ensemble Models
    models = []
    print(f"Loading {config.n_folds} fold models...")

    for fold in range(config.n_folds):
        model_path = config.get_model_path(fold)

        # Check if model exists (handle cases where training might have been partial)
        if not os.path.exists(model_path):
            print(
                f"  [Warning] Model for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        try:
            model = TweetModel(config.model_name, config.dropout)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            models.append(model)
            print(f"  -> Loaded model fold {fold}")
        except Exception as e:
            print(f"  [Error] Failed to load model fold {fold}: {e}")

    if not models:
        print("No models loaded. Cannot perform inference.")
        return

    # Inference Loop
    all_predictions = []

    print("Running prediction loop...")
    with torch.no_grad():
        for batch in test_dl:
            input_ids = batch["ids"].to(device)
            attention_mask = batch["mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            offsets = batch["offsets"].cpu().numpy()
            orig_texts = batch["orig_text"]
            sentiments = batch["sentiment"]

            batch_size = input_ids.size(0)

            # Accumulate logits from all models
            avg_start_logits = torch.zeros(batch_size, input_ids.size(1), device=device)
            avg_end_logits = torch.zeros(batch_size, input_ids.size(1), device=device)

            for model in models:
                s_logits, e_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                avg_start_logits += s_logits
                avg_end_logits += e_logits

            # Average logits
            avg_start_logits /= len(models)
            avg_end_logits /= len(models)

            # Decode predictions for the batch
            for i in range(batch_size):
                sentiment = sentiments[i]
                offset = offsets[i]

                # Reconstruct the text used during tokenization to ensure offsets align
                # Logic must match library.data.process_data: text = " " + " ".join(texts[i].split())
                text = " " + " ".join(orig_texts[i].split())

                pred_str = ""

                # Post-processing Rule: Neutral tweets -> Full Text
                # Data analysis shows Neutral sentiment has Jaccard ~0.98 with full text
                if sentiment == "neutral":
                    pred_str = text
                else:
                    # Get logits for this sample
                    s_logits = avg_start_logits[i]
                    e_logits = avg_end_logits[i]

                    # Decode best span
                    best_s, best_e = get_best_start_end_idxs(s_logits, e_logits)

                    # Extract string using offsets
                    # Check bounds
                    if best_s < len(offset) and best_e < len(offset):
                        # offset[i] is [start_char, end_char]
                        start_char = offset[best_s][0]
                        end_char = offset[best_e][1]
                        pred_str = text[start_char:end_char]
                    else:
                        pred_str = text

                # Clean up: remove the leading space added during processing
                all_predictions.append(pred_str.strip())

    # Align predictions with TextIDs
    # We need to load the test CSV again to get the IDs.
    print("Generating submission DataFrame...")
    df_test = pd.read_csv(config.test_path)

    # If debug mode was active, the data pipeline sampled the dataset.
    # We must replicate that sampling to ensure IDs match the predictions.
    if config.debug:
        df_test = df_test.sample(
            n=config.debug_sample_size, random_state=config.seed
        ).reset_index(drop=True)

    # Safety check for length mismatch
    if len(df_test) != len(all_predictions):
        print(
            f"Warning: Number of predictions ({len(all_predictions)}) does not match number of test samples ({len(df_test)})."
        )
        # Truncate to minimum length to allow saving (though results may be misaligned if not caused by drop_last)
        min_len = min(len(df_test), len(all_predictions))
        df_test = df_test.iloc[:min_len]
        all_predictions = all_predictions[:min_len]

    df_test["selected_text"] = all_predictions

    # Format for submission
    submission_df = df_test[["textID", "selected_text"]]

    # Save
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved successfully to {submission_path}")

    return submission_df
