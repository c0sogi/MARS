import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from library.config import Config
from library.dataset import get_data
from library.model import SentimentModel
from library.utils import get_best_start_end_idxs


def run_inference(
    test_meta_path=Config.TEST_META_PATH,
    base_model_dir=Config.WORKING_DIR,
    output_path=Config.SUBMISSION_PATH,
    batch_size=Config.VALID_BATCH_SIZE,
    device=Config.DEVICE,
    n_folds=Config.N_FOLDS,
    sample_size=None,
    debug=False,
):
    """
    Executes the inference pipeline using a 5-fold ensemble.

    Args:
        test_meta_path (str): Path to the test metadata CSV.
        base_model_dir (str): Directory containing trained model weights.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        device (torch.device): Device to run inference on.
        n_folds (int): Number of folds in the ensemble.
        sample_size (int, optional): If set, limits the number of samples for debugging.
        debug (bool): If True, enables debug mode (caching with debug suffix).
    """

    # --- 1. Load and Prepare Data ---
    print(f"Loading test data from {test_meta_path}...")
    if not os.path.exists(test_meta_path):
        raise FileNotFoundError(f"Test metadata not found at {test_meta_path}")

    df_test = pd.read_csv(test_meta_path)

    # Handle Debugging
    if debug and sample_size is None:
        sample_size = Config.DEBUG_SAMPLE_SIZE

    if sample_size is not None:
        print(f"Debugging mode: using first {sample_size} samples.")
        df_test = df_test.head(sample_size)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Create Dataset
    # split_name="test" triggers the caching logic in get_data specific to the test set
    test_dataset = get_data(
        df_test,
        tokenizer,
        split_name="test",
        load_cached_data=True,
        sample_size=sample_size if debug else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Test dataset prepared. Total samples: {len(test_dataset)}")

    # --- 2. Ensemble Inference ---
    num_samples = len(test_dataset)
    max_len = Config.MAX_LEN

    # Buffers to aggregate logits from all folds
    avg_start_logits = np.zeros((num_samples, max_len), dtype=np.float32)
    avg_end_logits = np.zeros((num_samples, max_len), dtype=np.float32)

    models_used = 0

    print(f"Starting inference with {n_folds}-fold ensemble...")

    for fold in range(n_folds):
        model_path = os.path.join(base_model_dir, f"model_fold_{fold}.bin")

        if not os.path.exists(model_path):
            print(
                f"Warning: Model weights for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        print(f"Running inference for Fold {fold}...")

        # Load Model
        model = SentimentModel()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        fold_start_preds = []
        fold_end_preds = []

        # Batch Prediction Loop
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device, dtype=torch.long)
                attention_mask = batch["attention_mask"].to(device, dtype=torch.long)

                with torch.amp.autocast("cuda", enabled=Config.USE_AMP):
                    start_logits, end_logits = model(input_ids, attention_mask)

                # Move to CPU and collect
                fold_start_preds.append(start_logits.float().cpu().numpy())
                fold_end_preds.append(end_logits.float().cpu().numpy())

        # Concatenate fold predictions
        fold_start_preds = np.concatenate(fold_start_preds, axis=0)
        fold_end_preds = np.concatenate(fold_end_preds, axis=0)

        # Accumulate
        avg_start_logits += fold_start_preds
        avg_end_logits += fold_end_preds

        models_used += 1

        # Clean up memory
        del model, state_dict, fold_start_preds, fold_end_preds
        torch.cuda.empty_cache()

    if models_used == 0:
        raise RuntimeError("No models were found for inference!")

    # Compute Average
    avg_start_logits /= models_used
    avg_end_logits /= models_used

    print("Ensemble inference complete. Decoding predictions...")

    # --- 3. Decoding and Reconstruction ---
    predictions = []

    # Retrieve metadata from dataset for reconstruction
    texts = test_dataset.texts
    sentiments = test_dataset.sentiments
    offsets = test_dataset.offsets
    ids = test_dataset.ids

    for i in range(num_samples):
        text = texts[i]
        sentiment = sentiments[i]
        offset = offsets[i]
        text_id = ids[i]

        start_logit = avg_start_logits[i]
        end_logit = avg_end_logits[i]

        if sentiment == "neutral":
            # Deterministic rule: Neutral tweets return the full text
            pred_text = text
        else:
            # Joint Logit Decoding: Find best span (start, end) maximizing sum of logits
            idx_start, idx_end = get_best_start_end_idxs(start_logit, end_logit)

            # Map token indices back to character indices using offsets
            # offset[i] is a tuple (start_char, end_char)
            char_start = offset[idx_start][0]
            char_end = offset[idx_end][1]

            # Extract substring from the NORMALIZED text
            pred_text = text[char_start:char_end]

        predictions.append({"textID": text_id, "selected_text": pred_text})

    # --- 4. Save Submission ---
    submission_df = pd.DataFrame(predictions)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV (Pandas handles quoting automatically for strings containing delimiters)
    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved successfully to {output_path}")
    print("Head of submission:")
    print(submission_df.head())

    return submission_df
