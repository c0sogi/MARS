import os
import json
import numpy as np
import pandas as pd
from library.config import Config


def tokenize(text):
    """
    Performs simple whitespace tokenization.

    Args:
        text (str): Input text.

    Returns:
        list: List of tokens.
    """
    if not text or not isinstance(text, str):
        return []
    return text.split()


def load_glove_embeddings(vocab, embedding_dim=100, load_cached_data=True):
    """
    Loads pre-trained GloVe embeddings and maps them to the vocabulary.
    Implements caching mechanism strictly following the requirements.

    Args:
        vocab (dict): Dictionary mapping tokens to indices.
        embedding_dim (int): Dimension of embeddings.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        numpy.ndarray: Embedding matrix of shape (vocab_size, embedding_dim).
    """
    cache_path = Config.EMBEDDING_MATRIX_CACHE_FILE

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            embedding_matrix = np.load(cache_path)
            # Verify shape matches current vocab/dim requirements
            if embedding_matrix.shape == (len(vocab), embedding_dim):
                print(f"Loaded embedding matrix from {cache_path}")
                return embedding_matrix
            else:
                print("Cached embedding matrix shape mismatch. Recomputing...")
        except Exception as e:
            print(f"Failed to load cached embeddings: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Initializing embedding matrix...")

    # Initialize with random normal distribution
    # Scale by 1/sqrt(dim) for better convergence
    embedding_matrix = np.random.normal(scale=0.6, size=(len(vocab), embedding_dim))

    # Set PAD and UNK if they exist in vocab
    if Config.PAD_TOKEN in vocab:
        embedding_matrix[vocab[Config.PAD_TOKEN]] = np.zeros(embedding_dim)

    # Attempt to load GloVe file if it exists in input
    # We check for a standard filename, though it may not be present in this environment
    glove_file = os.path.join(Config.INPUT_DIR, "glove.6B.100d.txt")

    if os.path.exists(glove_file):
        print(f"Loading GloVe vectors from {glove_file}...")
        hits = 0
        with open(glove_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                word = parts[0]
                if word in vocab:
                    try:
                        vector = np.array(parts[1:], dtype=np.float32)
                        if vector.shape[0] == embedding_dim:
                            embedding_matrix[vocab[word]] = vector
                            hits += 1
                    except ValueError:
                        continue
        print(f"Loaded {hits} vectors from GloVe.")
    else:
        print("GloVe file not found. Using random initialization for embeddings.")

    # 3. Save to cache
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embedding_matrix)
        print(f"Saved embedding matrix to {cache_path}")
    except Exception as e:
        print(f"Failed to save embedding matrix cache: {e}")

    return embedding_matrix


def parse_annotation_record(annotations_json):
    """
    Parses the annotation JSON string into sets of valid answer strings.

    Args:
        annotations_json (str): JSON string from metadata.

    Returns:
        dict: {'long': set of strings, 'short': set of strings}
        Strings are "start:end" or "YES"/"NO".
    """
    if not annotations_json:
        return {"long": set(), "short": set()}

    try:
        anns = json.loads(annotations_json)
    except:
        return {"long": set(), "short": set()}

    long_answers = set()
    short_answers = set()

    for ann in anns:
        # Long Answer
        la = ann.get("long_answer", {})
        start = la.get("start_token", -1)
        end = la.get("end_token", -1)
        if start != -1 and end != -1:
            long_answers.add(f"{start}:{end}")

        # Short Answers
        sas = ann.get("short_answers", [])
        for sa in sas:
            s_start = sa.get("start_token", -1)
            s_end = sa.get("end_token", -1)
            if s_start != -1 and s_end != -1:
                short_answers.add(f"{s_start}:{s_end}")

        # Yes/No Answer
        yn = ann.get("yes_no_answer", "NONE")
        if yn in ["YES", "NO"]:
            short_answers.add(yn)

    return {"long": long_answers, "short": short_answers}


def compute_micro_f1(predictions, ground_truths_metadata):
    """
    Computes Micro F1 score for Long and Short answers.

    Args:
        predictions (dict): {example_id: {'long': "str", 'short': "str"}}
                            "str" can be "start:end", "YES", "NO", or ""/None.
        ground_truths_metadata (pd.DataFrame): DataFrame containing 'example_id' and 'annotations'.

    Returns:
        dict: {'long_f1': float, 'short_f1': float, 'overall_f1': float}
    """
    tp_long, fp_long, fn_long = 0, 0, 0
    tp_short, fp_short, fn_short = 0, 0, 0

    # Convert metadata to dict for fast lookup
    gt_map = {}
    for _, row in ground_truths_metadata.iterrows():
        gt_map[row["example_id"]] = parse_annotation_record(row["annotations"])

    for eid, pred in predictions.items():
        if eid not in gt_map:
            continue

        truth = gt_map[eid]

        # --- Long Answer Evaluation ---
        p_long = str(pred.get("long", "")).strip()
        t_long_set = truth["long"]

        if not p_long and not t_long_set:
            # TN (both empty) - doesn't affect F1
            pass
        elif not p_long and t_long_set:
            fn_long += 1
        elif p_long and not t_long_set:
            fp_long += 1
        elif p_long and t_long_set:
            if p_long in t_long_set:
                tp_long += 1
            else:
                fp_long += 1
                fn_long += 1

        # --- Short Answer Evaluation ---
        p_short = str(pred.get("short", "")).strip()
        t_short_set = truth["short"]

        if not p_short and not t_short_set:
            pass
        elif not p_short and t_short_set:
            fn_short += 1
        elif p_short and not t_short_set:
            fp_short += 1
        elif p_short and t_short_set:
            if p_short in t_short_set:
                tp_short += 1
            else:
                fp_short += 1
                fn_short += 1

    def calc_f1(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    f1_long = calc_f1(tp_long, fp_long, fn_long)
    f1_short = calc_f1(tp_short, fp_short, fn_short)

    return {
        "long_f1": f1_long,
        "short_f1": f1_short,
        "overall_f1": (f1_long + f1_short) / 2,
    }


def format_submission(predictions, output_path=Config.SUBMISSION_FILE):
    """
    Formats predictions into the required CSV format and saves it.

    Args:
        predictions (dict): {example_id: {'long': "str", 'short': "str"}}
        output_path (str): Path to save the CSV.
    """
    rows = []
    for eid, pred in predictions.items():
        # Long answer row
        long_ans = pred.get("long", "")
        if long_ans is None:
            long_ans = ""
        rows.append({"example_id": f"{eid}_long", "PredictionString": str(long_ans)})

        # Short answer row
        short_ans = pred.get("short", "")
        if short_ans is None:
            short_ans = ""
        rows.append({"example_id": f"{eid}_short", "PredictionString": str(short_ans)})

    df = pd.DataFrame(rows)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(df)} rows.")
