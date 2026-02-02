import os
import json
import csv
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import set_seed


def load_tag_vocab():
    """
    Loads the tag vocabulary from the JSON file.
    """
    path = Config.tags_path
    # Fallback for debug mode if the main file doesn't exist yet
    if not os.path.exists(path) and Config.debug:
        path = os.path.join(Config.working_dir, "debug_tags.json")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Tag vocabulary not found at {path}. Ensure training has run."
        )

    with open(path, "r") as f:
        tags = json.load(f)
    return tags


def predict(model, loader, device):
    """
    Performs inference on the provided loader using the model.
    Returns probabilities, labels (if available), and IDs.
    """
    model.eval()
    all_probs = []
    all_ids = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            ids = batch["id"]

            # Forward pass
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_ids.extend(ids.numpy())

            if "labels" in batch:
                all_labels.append(batch["labels"].cpu().numpy())

    # Concatenate results
    all_probs = np.concatenate(all_probs, axis=0)
    all_ids = np.array(all_ids)

    if all_labels:
        all_labels = np.concatenate(all_labels, axis=0)
        return all_probs, all_labels, all_ids
    else:
        return all_probs, None, all_ids


def get_predictions(model, loader, cache_name, load_cached_data=True):
    """
    Wrapper around predict that handles caching of predictions to disk.
    Strictly follows the requirement to check cache first, then compute if missing.
    """
    cache_dir = Config.working_dir
    os.makedirs(cache_dir, exist_ok=True)

    probs_path = os.path.join(cache_dir, f"{cache_name}_probs.npy")
    ids_path = os.path.join(cache_dir, f"{cache_name}_ids.npy")
    labels_path = os.path.join(cache_dir, f"{cache_name}_labels.npy")

    has_labels = loader.dataset.labels is not None

    # 1. Try to load cached data
    if load_cached_data:
        # Check if essential files exist
        if os.path.exists(probs_path) and os.path.exists(ids_path):
            # If labels are expected, check for them too
            if has_labels and not os.path.exists(labels_path):
                pass  # Cache incomplete, proceed to compute
            else:
                print(f"Loading cached predictions for {cache_name}...")
                probs = np.load(probs_path)
                ids = np.load(ids_path)
                labels = np.load(labels_path) if has_labels else None
                return probs, labels, ids

    # 2. Compute from scratch
    print(f"Computing predictions for {cache_name}...")
    probs, labels, ids = predict(model, loader, Config.device)

    # 3. Save to cache
    print(f"Saving predictions for {cache_name} to cache...")
    np.save(probs_path, probs)
    np.save(ids_path, ids)
    if labels is not None:
        np.save(labels_path, labels)

    return probs, labels, ids


def optimize_threshold(val_probs, val_labels):
    """
    Finds the optimal threshold that maximizes the Mean F1-Score (Samples) on the validation set.
    The search range is dynamically determined based on the percentiles of the predicted probabilities.
    """
    print("Optimizing threshold...")

    # Sample probabilities to estimate distribution percentiles efficiently
    sample_size = min(1000000, val_probs.size)
    sample_probs = np.random.choice(val_probs.ravel(), size=sample_size, replace=False)

    start_p = Config.threshold_search_start_percentile
    end_p = Config.threshold_search_end_percentile

    low = np.percentile(sample_probs, start_p)
    high = np.percentile(sample_probs, end_p)

    # Safety check
    if low >= high:
        low, high = 0.1, 0.9

    print(f"Search range (Percentiles {start_p}-{end_p}): {low:.6f} to {high:.6f}")

    thresholds = np.linspace(low, high, Config.threshold_search_steps)
    best_threshold = 0.5
    best_f1 = -1.0

    # Ensure labels are binary integers for calculation
    val_labels_int = val_labels.astype(np.uint8)
    true_sum = val_labels_int.sum(axis=1)

    for th in thresholds:
        # Binarize predictions
        preds = (val_probs > th).astype(np.uint8)

        # Calculate F1 (Samples) manually for efficiency
        # F1 = 2 * |P intersect T| / (|P| + |T|)
        intersection = (preds * val_labels_int).sum(axis=1)
        pred_sum = preds.sum(axis=1)

        denominator = pred_sum + true_sum

        # Safe divide: if denom is 0, set to 1 to avoid error.
        # If both pred and true are empty, intersection is 0, result 0.
        safe_denom = np.maximum(denominator, 1e-9)
        f1_scores = 2 * intersection / safe_denom

        mean_f1 = np.mean(f1_scores)

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_threshold = th

    print(f"Best Threshold: {best_threshold} with F1-Score: {best_f1}")
    return best_threshold


def generate_submission(test_probs, test_ids, threshold, tag_vocab, output_path):
    """
    Applies the threshold to test probabilities, decodes tags, and saves the submission file.
    """
    print(f"Generating submission with threshold {threshold:.6f}...")

    # Binarize
    preds_binary = test_probs > threshold

    predictions = []

    # Decode tags
    # tag_vocab is a list where index corresponds to the class index
    for i in range(len(test_ids)):
        row_indices = np.where(preds_binary[i])[0]

        if len(row_indices) == 0:
            # Fallback: predict the single tag with highest probability
            top_idx = np.argmax(test_probs[i])
            row_tags = tag_vocab[top_idx]
        else:
            row_tags = " ".join([tag_vocab[idx] for idx in row_indices])

        predictions.append(row_tags)

    # Create DataFrame
    df_sub = pd.DataFrame({"Id": test_ids, "Tags": predictions})

    print(f"Saving submission to {output_path}...")
    # quoting=2 corresponds to csv.QUOTE_NONNUMERIC (quote non-numeric fields)
    df_sub.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print("Submission saved.")


def run_inference(model, val_loader, test_loader, load_cached_data=True):
    """
    Main inference pipeline:
    1. Predict on Validation Set (with caching)
    2. Optimize Threshold based on Validation F1
    3. Predict on Test Set (with caching)
    4. Generate Submission CSV
    """
    set_seed()
    device = Config.device
    model.to(device)

    # 1. Validation Inference
    val_probs, val_labels, _ = get_predictions(
        model, val_loader, "val", load_cached_data=load_cached_data
    )

    # 2. Optimize Threshold
    best_threshold = optimize_threshold(val_probs, val_labels)

    # Cleanup memory
    del val_probs, val_labels
    import gc

    gc.collect()

    # 3. Test Inference
    test_probs, _, test_ids = get_predictions(
        model, test_loader, "test", load_cached_data=load_cached_data
    )

    # 4. Generate Submission
    tag_vocab = load_tag_vocab()
    generate_submission(
        test_probs, test_ids, best_threshold, tag_vocab, Config.submission_path
    )
