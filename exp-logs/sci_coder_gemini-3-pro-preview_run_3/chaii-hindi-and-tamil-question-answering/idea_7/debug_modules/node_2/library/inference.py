import os
import torch
import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed
from library.data import QADataset, qa_collate_fn
from library.model import WeightedTokenClassifier
from library.qa_engine import get_predictions


def predict_engine(model, dataloader, device):
    """
    Performs Global Confidence Aggregation: processes all sliding windows for a document,
    decodes candidate spans, and selects the single span with the highest confidence score
    across all windows.

    Args:
        model: The trained PyTorch model.
        dataloader: DataLoader containing the dataset.
        device: The torch device.

    Returns:
        dict: A dictionary mapping example_id (str) to predicted answer text (str).
    """
    # Relies on the library implementation which performs the sliding window aggregation
    return get_predictions(model, dataloader, device)


def ensemble_vote(prediction_dicts, test_ids):
    """
    Aggregates string predictions from multiple seeded models using a majority voting mechanism.

    Args:
        prediction_dicts (list[dict]): List of dictionaries containing predictions from different models.
        test_ids (list): List of all test example IDs to ensure completeness.

    Returns:
        dict: A dictionary mapping example_id to the final voted prediction string.
    """
    final_predictions = {}

    for eid in test_ids:
        votes = []
        for p_dict in prediction_dicts:
            # Default to empty string if ID missing in a specific model's preds
            # This handles cases where a model might fail to predict for a specific ID
            votes.append(p_dict.get(eid, ""))

        # Majority Vote
        counter = Counter(votes)
        # Get the most common prediction.
        # most_common(1) returns a list of tuples [(element, count)].
        # We take the element of the first tuple.
        if votes:
            best_pred = counter.most_common(1)[0][0]
        else:
            best_pred = ""

        final_predictions[eid] = best_pred

    return final_predictions


def run_inference(
    test_csv_path=Config.TEST_CSV,
    submission_path=Config.SUBMISSION_FILE,
    model_dir=Config.MODEL_OUTPUT_DIR,
    seeds=Config.SEEDS,
    batch_size=Config.BATCH_SIZE,
    device=Config.DEVICE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Orchestrates the inference pipeline: loads data, runs predictions for each seed,
    performs ensemble voting, and saves the submission file.

    Args:
        test_csv_path (str): Path to the test CSV file.
        submission_path (str): Path where the submission CSV will be saved.
        model_dir (str): Directory containing trained model checkpoints.
        seeds (list): List of seeds/folds to use for ensembling.
        batch_size (int): Batch size for inference.
        device (str): Device to run inference on ('cuda' or 'cpu').
        num_workers (int): Number of dataloader workers.
    """
    set_seed(Config.SEED)
    print(f"Starting inference pipeline on device: {device}")

    # 1. Load Test IDs
    # We read the CSV directly to get the list of IDs for the ensemble step ordering
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV not found at {test_csv_path}")

    df_test = pd.read_csv(test_csv_path)
    test_ids = df_test["id"].tolist()
    print(f"Loaded {len(test_ids)} test samples.")

    # 2. Prepare DataLoader
    # QADataset uses Config paths internally for the 'test' mode.
    test_dataset = QADataset(mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=qa_collate_fn,
        num_workers=num_workers,
    )

    # 3. Generate Predictions for each Seed
    all_seed_predictions = []

    for seed in seeds:
        model_filename = f"model_seed_{seed}.pt"
        model_path = os.path.join(model_dir, model_filename)

        if not os.path.exists(model_path):
            print(
                f"Warning: Model checkpoint not found at {model_path}. Skipping seed {seed}."
            )
            continue

        print(f"Running inference for seed {seed}...")

        # Initialize model structure
        # class_weights=None because we are only doing inference
        model = WeightedTokenClassifier(class_weights=None)

        # Load weights
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            continue

        model.to(device)
        model.eval()

        # Get predictions using the predict_engine
        preds = predict_engine(model, test_loader, device)
        all_seed_predictions.append(preds)

        # Cleanup to save memory
        del model
        del state_dict
        torch.cuda.empty_cache()

    if not all_seed_predictions:
        print("Error: No valid predictions generated from any seed.")
        return

    # 4. Ensemble Voting
    print("Aggregating predictions with Majority Voting...")
    final_preds_map = ensemble_vote(all_seed_predictions, test_ids)

    # 5. Create Submission File
    # Ensure we output in the format: id,PredictionString
    submission_data = [
        {"id": eid, "PredictionString": final_preds_map[eid]} for eid in test_ids
    ]

    submission_df = pd.DataFrame(submission_data)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
