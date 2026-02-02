import os
import json
import pandas as pd
import numpy as np
from library.config import Config


def load_jsonl(file_path, sample_size=0):
    """
    Generator that yields entries from a JSONL file.

    Args:
        file_path (str): Path to the JSONL file.
        sample_size (int): Number of samples to read. If 0, read all.

    Yields:
        dict: The parsed JSON object.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if sample_size > 0 and i >= sample_size:
                break
            yield json.loads(line)


def format_prediction_string(start_index, end_index, yes_no_answer="NONE"):
    """
    Formats the prediction into the required submission string.

    Args:
        start_index (int): Start token index.
        end_index (int): End token index.
        yes_no_answer (str): 'YES', 'NO', or 'NONE'.

    Returns:
        str: Formatted prediction string (e.g., "1:10", "YES", or "").
    """
    if yes_no_answer in ["YES", "NO"]:
        return yes_no_answer

    # Check for valid span (start < end)
    if start_index != -1 and end_index != -1 and start_index < end_index:
        return f"{start_index}:{end_index}"

    return ""


def _extract_valid_answers(annotations):
    """
    Helper to extract set of valid answer strings from annotations list.
    Returns tuple of lists: (valid_long_strings, valid_short_strings)
    """
    valid_long = set()
    valid_short = set()

    for ann in annotations:
        # Long answer extraction
        la = ann.get("long_answer", {})
        if la.get("candidate_index", -1) != -1:
            s = la.get("start_token", -1)
            e = la.get("end_token", -1)
            if s != -1 and e != -1:
                valid_long.add(f"{s}:{e}")

        # Short answer extraction (Yes/No or Span)
        yn = ann.get("yes_no_answer", "NONE")
        if yn in ["YES", "NO"]:
            valid_short.add(yn)
        else:
            sas = ann.get("short_answers", [])
            for sa in sas:
                s = sa.get("start_token", -1)
                e = sa.get("end_token", -1)
                if s != -1 and e != -1:
                    valid_short.add(f"{s}:{e}")

    return list(valid_long), list(valid_short)


def load_ground_truth_data(file_path, load_cached_data=True, sample_size=0):
    """
    Loads ground truth data (valid answer strings) for all examples in the file.
    Uses caching to parquet to avoid re-parsing the large JSONL file.

    Args:
        file_path (str): Path to the JSONL file containing annotations.
        load_cached_data (bool): Whether to try loading from cache.
        sample_size (int): Limit number of examples processed/loaded.

    Returns:
        pd.DataFrame: DataFrame with columns ['example_id', 'valid_long', 'valid_short']
    """
    # Determine cache filename based on sample_size to ensure consistency
    if sample_size > 0:
        cache_filename = f"ground_truth_sample_{sample_size}.parquet"
    else:
        cache_filename = "ground_truth_full.parquet"

    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If load fails (e.g. corrupt file), proceed to recompute
            pass

    # 2. Compute from scratch
    data = []

    # Use the generator to iterate efficiently
    for entry in load_jsonl(file_path, sample_size):
        eid = str(entry["example_id"])
        annotations = entry.get("annotations", [])
        v_long, v_short = _extract_valid_answers(annotations)

        data.append({"example_id": eid, "valid_long": v_long, "valid_short": v_short})

    df = pd.DataFrame(data)

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def compute_f1_score(predictions_df, ground_truth_df):
    """
    Computes Micro F1 score based on exact string matching.

    Args:
        predictions_df (pd.DataFrame): Must contain 'example_id' (with _long/_short suffix)
                                       and 'PredictionString'.
        ground_truth_df (pd.DataFrame): Must contain 'example_id' (base ID), 'valid_long',
                                        and 'valid_short'.

    Returns:
        tuple: (precision, recall, f1)
    """
    # Create lookup for predictions: { 'id_type': 'pred_string' }
    pred_lookup = dict(
        zip(predictions_df["example_id"], predictions_df["PredictionString"])
    )

    tp = 0
    fp = 0
    fn = 0

    # Iterate over ground truth examples to ensure all targets are accounted for
    for _, row in ground_truth_df.iterrows():
        base_id = str(row["example_id"])
        valid_longs = set(row["valid_long"])
        valid_shorts = set(row["valid_short"])

        # --- Evaluate Long Answer ---
        long_id = f"{base_id}_long"
        val_long = pred_lookup.get(long_id)

        # Handle NaN/None/Float('nan') safely
        if val_long is None or (isinstance(val_long, float) and np.isnan(val_long)):
            pred_long = ""
        else:
            pred_long = str(val_long).strip()

        if pred_long:
            # Prediction exists
            if pred_long in valid_longs:
                tp += 1
            else:
                fp += 1
        else:
            # No prediction made
            if valid_longs:
                # But a valid answer existed
                fn += 1

        # --- Evaluate Short Answer ---
        short_id = f"{base_id}_short"
        val_short = pred_lookup.get(short_id)

        if val_short is None or (isinstance(val_short, float) and np.isnan(val_short)):
            pred_short = ""
        else:
            pred_short = str(val_short).strip()

        if pred_short:
            if pred_short in valid_shorts:
                tp += 1
            else:
                fp += 1
        else:
            if valid_shorts:
                fn += 1

    # Calculate Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1
