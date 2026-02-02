import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from library.config import Config
from library.utils import seed_everything
from library.model import TweetModel
from library.data import get_test_loader
from library.engine import decode_prediction


def predict_test(load_cached_data=True, debug=False):
    """
    Performs inference on the test set using the trained Stage 2 Student models.
    Generates the submission file.

    Args:
        load_cached_data (bool): If False, clears existing cache to force data re-processing.
        debug (bool): If True, runs on a subset (though data loader loads full set,
                      we can limit processing if needed, but standard is full run).
    """
    seed_everything(Config.seed)

    # 1. Cache Management
    if not load_cached_data:
        cache_files = [
            os.path.join(Config.output_dir, f"cached_test_{Config.max_len}.npz"),
            os.path.join(Config.output_dir, f"cached_test_{Config.max_len}.parquet"),
        ]
        for f in cache_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    print(f"Removed cache file: {f}")
                except OSError as e:
                    print(f"Error removing cache file {f}: {e}")

    # 2. Data Loading
    print("Loading test data...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    test_loader, test_df = get_test_loader(tokenizer)

    # Extract offsets for decoding
    # test_loader.dataset is a TweetDataset
    offsets = test_loader.dataset.offsets

    num_samples = len(test_df)
    print(f"Test samples: {num_samples}")

    # 3. Ensemble Inference
    # Initialize accumulators for logits
    final_start_logits = np.zeros((num_samples, Config.max_len))
    final_end_logits = np.zeros((num_samples, Config.max_len))

    device = Config.device

    # Iterate over folds
    models_found = 0
    for fold in range(Config.n_folds):
        model_path = os.path.join(Config.output_dir, f"model_fold_{fold}.pth")

        if not os.path.exists(model_path):
            print(
                f"Warning: Model for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        print(f"Running inference with fold {fold} model...")

        # Load Model
        model = TweetModel()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        fold_start_preds = []
        fold_end_preds = []

        # Batch Prediction
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                input_ids = data["input_ids"].to(device, dtype=torch.long)
                attention_mask = data["attention_mask"].to(device, dtype=torch.long)
                token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)

                start_logits, end_logits = model(
                    input_ids, attention_mask, token_type_ids
                )

                fold_start_preds.append(start_logits.cpu().numpy())
                fold_end_preds.append(end_logits.cpu().numpy())

        # Accumulate logits
        if fold_start_preds:
            final_start_logits += np.concatenate(fold_start_preds, axis=0)
            final_end_logits += np.concatenate(fold_end_preds, axis=0)
            models_found += 1

    if models_found == 0:
        print(
            "Error: No models found. Generating submission with full text (fallback)."
        )
        # In case of total failure, we could just predict the whole text,
        # but the code below will handle zeros by predicting index 0.

    # 4. Decoding
    print("Decoding predictions...")
    predictions = []

    # Get best start/end indices from summed logits
    start_idxs = np.argmax(final_start_logits, axis=1)
    end_idxs = np.argmax(final_end_logits, axis=1)

    for i in range(num_samples):
        text = str(test_df.loc[i, "text"])
        sentiment = str(test_df.loc[i, "sentiment"])
        off = offsets[i]
        s_idx = start_idxs[i]
        e_idx = end_idxs[i]

        # Decode using the engine's logic (handles neutral heuristic and offsets)
        pred_text = decode_prediction(s_idx, e_idx, text, off, sentiment)
        predictions.append(pred_text)

    # 5. Submission Generation
    save_dir = "./submission"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "submission.csv")

    submission = pd.DataFrame(
        {"textID": test_df["textID"], "selected_text": predictions}
    )

    # Save to CSV
    # quoting=2 corresponds to csv.QUOTE_NONNUMERIC, but pandas default is usually fine.
    # We rely on pandas default to quote strings containing delimiters.
    submission.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
    print(submission.head())
