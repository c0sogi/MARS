import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_metadata(csv_path, load_cached_data=True):
    """
    Loads metadata from a CSV file. Implements caching using Parquet.

    Args:
        csv_path (str): Path to the source CSV file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Construct cache path
    filename = os.path.basename(csv_path)
    cache_filename = filename.replace(".csv", ".parquet")
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If load fails, proceed to process from scratch
            pass

    # 2. Compute/Process data from scratch
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Basic preprocessing: Ensure labels are strings and handle NaNs
    if "labels" in df.columns:
        df["labels"] = df["labels"].fillna("")

    # 3. Save result to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return df


def parse_ground_truth(label_str):
    """
    Parses a ground truth label string into a list of dictionaries.
    Format: "Unicode X Y Width Height ..."
    """
    if pd.isna(label_str) or label_str == "":
        return []

    parts = label_str.strip().split(" ")
    # Each annotation consists of 5 values
    if len(parts) % 5 != 0:
        return []

    annotations = []
    for i in range(0, len(parts), 5):
        try:
            ann = {
                "label": parts[i],
                "x": int(parts[i + 1]),
                "y": int(parts[i + 2]),
                "w": int(parts[i + 3]),
                "h": int(parts[i + 4]),
            }
            annotations.append(ann)
        except ValueError:
            continue
    return annotations


def parse_prediction(label_str):
    """
    Parses a prediction label string into a list of dictionaries.
    Format: "Unicode X Y ..."
    """
    if pd.isna(label_str) or label_str == "":
        return []

    parts = label_str.strip().split(" ")
    # Each prediction consists of 3 values
    if len(parts) % 3 != 0:
        return []

    predictions = []
    for i in range(0, len(parts), 3):
        try:
            pred = {"label": parts[i], "x": int(parts[i + 1]), "y": int(parts[i + 2])}
            predictions.append(pred)
        except ValueError:
            continue
    return predictions


def calc_f1_score(true_df, pred_df):
    """
    Calculates the modified F1 Score.
    A true positive is a prediction with the correct label and a center point
    inside the ground truth bounding box.

    Args:
        true_df (pd.DataFrame): DataFrame containing 'image_id' and 'labels' (Ground Truth).
        pred_df (pd.DataFrame): DataFrame containing 'image_id' and 'labels' (Predictions).

    Returns:
        float: The calculated F1 score.
    """
    # Create mappings for fast lookup
    true_map = dict(zip(true_df["image_id"].astype(str), true_df["labels"]))
    pred_map = dict(zip(pred_df["image_id"].astype(str), pred_df["labels"]))

    tp = 0
    total_gt = 0
    total_pred = 0

    # Iterate over all images present in the ground truth
    for img_id, gt_str in true_map.items():
        gts = parse_ground_truth(gt_str)
        total_gt += len(gts)

        pred_str = pred_map.get(img_id, "")
        preds = parse_prediction(pred_str)
        total_pred += len(preds)

        matched_gt_indices = set()

        # Match predictions to ground truth
        for p in preds:
            p_label = p["label"]
            p_x = p["x"]
            p_y = p["y"]

            # Find a matching ground truth box
            for i, g in enumerate(gts):
                if i in matched_gt_indices:
                    continue

                # Check Label Match
                if p_label != g["label"]:
                    continue

                # Check Spatial Match (Point inside Box)
                # Box: x, y, w, h -> [x, x+w), [y, y+h)
                if (g["x"] <= p_x < g["x"] + g["w"]) and (
                    g["y"] <= p_y < g["y"] + g["h"]
                ):

                    matched_gt_indices.add(i)
                    tp += 1
                    break  # Prediction matched, move to next prediction

    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return f1


class Logger:
    """
    Logs training metrics to console and a CSV file.
    """

    def __init__(self, filename="training_log.csv"):
        self.log_path = os.path.join(Config.WORKING_DIR, filename)
        self.data = []

    def log(self, metrics):
        """
        Args:
            metrics (dict): Dictionary of metric names and values.
        """
        self.data.append(metrics)

        # Print full precision without rounding
        log_str = " | ".join([f"{k}: {v}" for k, v in metrics.items()])
        print(log_str)

        self.save()

    def save(self):
        df = pd.DataFrame(self.data)
        df.to_csv(self.log_path, index=False)


class EarlyStopping:
    """
    Implements early stopping to terminate training when validation loss stops improving.
    """

    def __init__(self, patience=5, delta=0, mode="min", save_path="best_model.pth"):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the
                        quantity monitored has stopped decreasing.
            save_path (str): Filename to save the best model.
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.save_path = os.path.join(Config.WORKING_DIR, save_path)
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score = np.inf if mode == "min" else -np.inf

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        else:
            if self.mode == "min":
                improved = score < (self.best_score - self.delta)
            else:
                improved = score > (self.best_score + self.delta)

            if improved:
                self.best_score = score
                self.save_checkpoint(model)
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, model):
        """Saves model when validation score decreases."""
        torch.save(model.state_dict(), self.save_path)


def create_submission_file(predictions, output_path="./submission/submission.csv"):
    """
    Generates the submission CSV file from predictions.

    Args:
        predictions (list of dict): List containing dicts with 'image_id' and 'labels'.
        output_path (str): Path to save the submission file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame(predictions)

    # Ensure columns are in correct order
    if "image_id" in df.columns and "labels" in df.columns:
        df = df[["image_id", "labels"]]

    df.to_csv(output_path, index=False)
