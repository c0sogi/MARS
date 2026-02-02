import os
import torch
import pandas as pd
from library.config import Config
from library.dataset import get_dataloaders
from library.model import load_model
from library.utils import find_best_substring


def generate_predictions(load_cached_data=True, device=None):
    """
    Generates predictions for the test set using the trained model and saves them to a CSV file.

    Args:
        load_cached_data (bool): Whether to attempt loading data from the parquet cache.
        device (torch.device, optional): The device to run inference on. Defaults to Config.DEVICE.

    Returns:
        str: The path to the saved submission file.
    """
    # 1. Setup Configuration
    if device is None:
        device = Config.DEVICE

    print(f"Initializing inference on device: {device}")

    # 2. Load Data and Tokenizer
    # get_dataloaders returns (train_loader, val_loader, test_loader, tokenizer)
    # We only need the test_loader and tokenizer for inference.
    print("Loading test data...")
    _, _, test_loader, tokenizer = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Load Model
    # Prioritize loading the best model saved during training.
    # If not found, fall back to the base model (useful for debugging/testing without training).
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading best model from checkpoint: {Config.MODEL_SAVE_PATH}")
        model_path = Config.MODEL_SAVE_PATH
    else:
        print(
            f"Warning: Checkpoint not found at {Config.MODEL_SAVE_PATH}. Loading base model: {Config.MODEL_NAME}"
        )
        model_path = Config.MODEL_NAME

    model = load_model(model_path=model_path, device=device)
    model.eval()

    # 4. Inference Loop
    predictions_map = {}  # id -> list of predictions
    print(f"Starting prediction generation for {len(test_loader.dataset)} windows...")

    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to the appropriate device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Retrieve metadata required for submission and post-processing
            ids = batch["ids"]
            contexts = batch["context"]

            # Generate answer token IDs
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=Config.MAX_TARGET_LENGTH,
            )

            # Decode token IDs to text strings
            decoded_preds = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            # Post-process predictions
            for i, pred_text in enumerate(decoded_preds):
                context = contexts[i]
                sample_id = ids[i]

                # The task requires the answer to be a quoted substring of the context.
                # The generative model might produce text that is slightly different (e.g., normalization).
                # find_best_substring locates the best matching span in the original context.
                final_pred = find_best_substring(context, pred_text)

                if sample_id not in predictions_map:
                    predictions_map[sample_id] = []
                predictions_map[sample_id].append(final_pred)

    # 5. Aggregate and Save Submission
    results = []
    for sample_id, preds in predictions_map.items():
        # Heuristic: Select the longest prediction across all windows
        # Cite solution_lesson_node_00004
        best_pred = max(preds, key=len) if preds else ""
        results.append({"id": sample_id, "PredictionString": best_pred})

    # Create DataFrame conforming to the submission format
    submission_df = pd.DataFrame(results)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)

    print(f"Successfully generated predictions for {len(submission_df)} samples.")
    print(f"Submission file saved to: {Config.SUBMISSION_FILE}")

    return Config.SUBMISSION_FILE
