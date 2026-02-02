import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from library.config import Config
from library.data_loader import get_tokenizer, prepare_test_features, QADataset
from library.model_arch import get_model
from library.utils import cleanup


def get_test_features_cached(test_df, tokenizer, load_cached_data=True):
    """
    Retrieves test features, utilizing Parquet caching to ensure efficiency and reproducibility.

    Args:
        test_df (pd.DataFrame): The raw test dataframe.
        tokenizer: The tokenizer instance.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed features dataframe.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, "test_features_inference.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached test features from {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print("Processing test features...")
    features_df = prepare_test_features(test_df, tokenizer)
    features_df.to_parquet(cache_path, index=False)
    return features_df


def get_fold_logits(fold_idx, features_df, device):
    """
    Loads the model for a specific fold and computes logits for the entire feature set.

    Args:
        fold_idx (int): The fold index (0 to N_FOLDS-1).
        features_df (pd.DataFrame): The processed test features.
        device (torch.device): The computation device.

    Returns:
        tuple: (start_logits, end_logits) as numpy arrays, or (None, None) if checkpoint missing.
    """
    checkpoint_path = os.path.join(
        Config.WORKING_DIR, f"fold_{fold_idx}_best_model.pth"
    )

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found for fold {fold_idx}: {checkpoint_path}")
        return None, None

    print(f"Loading model for fold {fold_idx} from {checkpoint_path}")
    model = get_model(checkpoint_path)
    model.eval()

    # Create Dataset and DataLoader
    # QADataset handles the conversion of dataframe rows to tensors
    dataset = QADataset(features_df, mode="test")

    # Inference can handle larger batch sizes than training
    batch_size = Config.BATCH_SIZE * 4
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    start_logits_list = []
    end_logits_list = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            start_logits_list.append(outputs.start_logits.cpu().numpy())
            end_logits_list.append(outputs.end_logits.cpu().numpy())

    # Concatenate results from all batches
    start_logits = np.concatenate(start_logits_list, axis=0)
    end_logits = np.concatenate(end_logits_list, axis=0)

    # Cleanup to free memory for the next fold
    del model
    cleanup()

    return start_logits, end_logits


def ensemble_and_postprocess(test_df, features_df, fold_indices):
    """
    Aggregates logits from multiple folds, selects the best answer spans,
    and reconstructs the answer text.

    Args:
        test_df (pd.DataFrame): Raw test data (needed for context text).
        features_df (pd.DataFrame): Processed features (needed for mappings).
        fold_indices (list): List of fold indices to ensemble.

    Returns:
        pd.DataFrame: Final predictions in submission format.
    """
    device = Config.DEVICE
    num_samples = len(features_df)

    # Initialize accumulators for logits
    # We rely on numpy broadcasting for shape inference or initialize based on first valid fold
    avg_start_logits = None
    avg_end_logits = None
    valid_folds = 0

    for fold_idx in fold_indices:
        s_logits, e_logits = get_fold_logits(fold_idx, features_df, device)
        if s_logits is not None:
            if avg_start_logits is None:
                avg_start_logits = s_logits
                avg_end_logits = e_logits
            else:
                avg_start_logits += s_logits
                avg_end_logits += e_logits
            valid_folds += 1

    if valid_folds == 0:
        print("No valid models found for ensemble. Generating dummy predictions.")
        return pd.DataFrame(
            {"id": test_df["id"], "PredictionString": ['"dummy text"'] * len(test_df)}
        )

    # Compute average
    avg_start_logits /= valid_folds
    avg_end_logits /= valid_folds

    print("Post-processing predictions...")

    # Map example_id to context text for fast lookup
    id_to_context = dict(zip(test_df["id"], test_df["context"]))

    # Dictionary to store the best result per example_id
    # format: example_id -> (best_score, prediction_string)
    best_results = {}

    # Extract columns to numpy/lists for faster iteration
    example_ids = features_df["example_id"].values
    offset_mappings = features_df["offset_mapping"].values
    sequence_ids_list = features_df["sequence_ids"].values

    # Hyperparameters for span selection
    n_best_size = 20
    max_answer_len = 100

    for i in range(num_samples):
        ex_id = example_ids[i]
        offsets = offset_mappings[i]
        seq_ids = sequence_ids_list[i]
        start_log = avg_start_logits[i]
        end_log = avg_end_logits[i]

        # Identify context tokens (where sequence_id == 1)
        # Note: sequence_ids might contain -1 for special tokens
        context_start_idx = -1
        context_end_idx = -1

        for idx, sid in enumerate(seq_ids):
            if sid == 1:
                if context_start_idx == -1:
                    context_start_idx = idx
                context_end_idx = idx

        # Skip if no context found in this window
        if context_start_idx == -1:
            continue

        # Get top-k start and end indices
        start_indexes = np.argsort(start_log)[-n_best_size:]
        end_indexes = np.argsort(end_log)[-n_best_size:]

        window_best_score = -float("inf")
        window_best_span = None

        for start_index in start_indexes:
            # Ensure start is within context
            if start_index < context_start_idx or start_index > context_end_idx:
                continue

            for end_index in end_indexes:
                # Ensure end is within context
                if end_index < context_start_idx or end_index > context_end_idx:
                    continue

                # Valid span constraints
                if end_index < start_index:
                    continue

                length = end_index - start_index + 1
                if length > max_answer_len:
                    continue

                score = start_log[start_index] + end_log[end_index]

                if score > window_best_score:
                    window_best_score = score
                    window_best_span = (start_index, end_index)

        # Update global best for this example_id if this window has a better score
        if ex_id not in best_results or window_best_score > best_results[ex_id][0]:
            if window_best_span:
                s_idx, e_idx = window_best_span

                # Retrieve character offsets
                start_char = offsets[s_idx][0]
                end_char = offsets[e_idx][1]

                context_text = id_to_context.get(ex_id, "")

                # Clamp indices just in case
                start_char = min(max(0, start_char), len(context_text))
                end_char = min(max(0, end_char), len(context_text))

                pred_text = context_text[start_char:end_char]

                # Format as quoted string
                pred_string = f'"{pred_text}"'
                best_results[ex_id] = (window_best_score, pred_string)
            else:
                best_results[ex_id] = (window_best_score, '"dummy text"')

    # Construct final dataframe ensuring all IDs are present
    final_data = []
    unique_ids = test_df["id"].unique()

    for uid in unique_ids:
        if uid in best_results:
            final_data.append({"id": uid, "PredictionString": best_results[uid][1]})
        else:
            final_data.append({"id": uid, "PredictionString": '"dummy text"'})

    return pd.DataFrame(final_data)


def generate_submission(load_cached_data=True):
    """
    Main orchestration function to generate the submission file.
    """
    print("Starting submission generation...")

    # Load Test Data
    test_df = pd.read_csv(Config.TEST_CSV)

    # Initialize Tokenizer
    tokenizer = get_tokenizer()

    # Prepare Features (with caching)
    features_df = get_test_features_cached(
        test_df, tokenizer, load_cached_data=load_cached_data
    )

    # Define folds to use (0 to N_FOLDS-1)
    fold_indices = list(range(Config.N_FOLDS))

    # Run Ensemble and Post-processing
    submission_df = ensemble_and_postprocess(test_df, features_df, fold_indices)

    # Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
