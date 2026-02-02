import os
import torch
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer
from library.config import Config
from library.utils import set_seed
from library.data_loader import get_dataloaders
from library.model import get_model, predict as lib_predict, extract_answer


def decode_span(input_ids, predictions, tokenizer):
    """
    Extracts the answer text from token IDs and predicted labels.
    Wraps the library function to utilize the existing logic for finding
    the valid B-ANS/I-ANS span within the context.

    Args:
        input_ids (torch.Tensor or np.array): Token IDs.
        predictions (torch.Tensor or np.array): Predicted labels.
        tokenizer: Transformers tokenizer.

    Returns:
        str: Decoded answer string.
    """
    return extract_answer(input_ids, predictions, tokenizer)


def predict_with_model(model, dataloader, tokenizer, device):
    """
    Generates predictions for the test set using a single model instance.
    Wraps the library function which implements the 'Greedy First-Match'
    strategy (scanning windows sequentially and stopping at the first valid span).

    Args:
        model: The trained PyTorch model.
        dataloader: DataLoader for the test set.
        tokenizer: Transformers tokenizer.
        device: Torch device.

    Returns:
        dict: A dictionary mapping example_ids to prediction strings.
    """
    return lib_predict(model, dataloader, tokenizer, device)


def majority_vote(prediction_dicts):
    """
    Aggregates predictions from multiple models using majority voting.

    Args:
        prediction_dicts (list): List of dictionaries {example_id: prediction_string}
                                 from different models.

    Returns:
        dict: {example_id: final_prediction_string}
    """
    if not prediction_dicts:
        return {}

    # Get all example IDs (assuming all models predicted on the same set)
    example_ids = list(prediction_dicts[0].keys())
    final_preds = {}

    for ex_id in example_ids:
        votes = []
        for preds in prediction_dicts:
            # Retrieve prediction for this ID from each model
            votes.append(preds.get(ex_id, ""))

        # Find the most common prediction
        # In case of a tie (e.g., 3 different answers), Counter.most_common
        # typically returns the first one encountered (prioritizing the first seed).
        most_common = Counter(votes).most_common(1)[0][0]
        final_preds[ex_id] = most_common

    return final_preds


def run_inference_pipeline():
    """
    Main orchestration function:
    1. Loads test data.
    2. Runs inference for each seeded model.
    3. Aggregates results via majority vote.
    4. Saves the submission file.
    """
    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Load Test DataLoader (using cache if available)
    # We ignore train/val loaders here
    _, _, test_loader = get_dataloaders(debug=False, load_cached_data=True)

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_CHECKPOINT)
    device = Config.DEVICE

    all_predictions = []

    # Iterate over the ensemble seeds
    for seed in Config.SEEDS:
        model_path = os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pt")

        if not os.path.exists(model_path):
            print(
                f"Warning: Model for seed {seed} not found at {model_path}. Skipping."
            )
            continue

        print(f"Running inference with model seed {seed}...")
        set_seed(seed)  # Ensure deterministic behavior

        # Initialize and load model
        model = get_model()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        # Generate predictions
        preds = predict_with_model(model, test_loader, tokenizer, device)
        all_predictions.append(preds)

        # Cleanup to save memory
        del model
        torch.cuda.empty_cache()

    if not all_predictions:
        print("No predictions generated.")
        return

    # Aggregate predictions
    print("Aggregating predictions with Majority Voting...")
    final_preds_dict = majority_vote(all_predictions)

    # Prepare Submission DataFrame
    submission_df = pd.DataFrame(
        {
            "id": list(final_preds_dict.keys()),
            "PredictionString": list(final_preds_dict.values()),
        }
    )

    # Save to CSV
    save_path = Config.SUBMISSION_FILE
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
