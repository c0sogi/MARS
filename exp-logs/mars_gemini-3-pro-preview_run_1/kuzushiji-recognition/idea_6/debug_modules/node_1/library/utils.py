import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import SEED, WORKING_DIR, IMG_SIZE, DOWN_RATIO, NUM_CLASSES


def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger:
    """
    Simple logger to track training progress in a CSV file.
    """

    def __init__(self, log_path):
        self.log_path = log_path
        self.data = []

    def log(self, epoch, train_loss, val_loss, val_f1, time_elapsed):
        # Print metrics with full precision
        print(
            f"Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}, Val F1: {val_f1}, Time: {time_elapsed}"
        )

        self.data.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_f1,
                "time": time_elapsed,
            }
        )

        # Save to CSV
        df = pd.DataFrame(self.data)
        df.to_csv(self.log_path, index=False)


def load_and_parse_metadata(csv_path, load_cached_data=True):
    """
    Parses the metadata CSV file.
    Extracts bounding boxes and labels from the 'labels' column.
    Caches the result as an .npy file in the working directory.

    Args:
        csv_path (str): Path to the metadata CSV file.
        load_cached_data (bool): Whether to try loading from cache first.

    Returns:
        list: A list of dictionaries containing image metadata and parsed annotations.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Construct cache path
    filename = os.path.basename(csv_path).replace(".csv", ".npy")
    cache_path = os.path.join(WORKING_DIR, f"parsed_{filename}")

    # Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            return data.tolist()
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute from scratch
    df = pd.read_csv(csv_path)
    data = []

    for _, row in df.iterrows():
        item = {
            "image_id": row["image_id"],
            "file_path": row["file_path"],
            "group_id": row.get("group_id", ""),
            "annotations": [],
        }

        labels_str = row["labels"]
        # Check if labels_str is valid and not NaN
        if isinstance(labels_str, str) and len(labels_str.strip()) > 0:
            parts = labels_str.strip().split(" ")
            # Format: Unicode X Y W H
            # We expect groups of 5
            if len(parts) % 5 == 0:
                for i in range(0, len(parts), 5):
                    label = parts[i]
                    try:
                        x = int(parts[i + 1])
                        y = int(parts[i + 2])
                        w = int(parts[i + 3])
                        h = int(parts[i + 4])
                        item["annotations"].append(
                            {"label": label, "bbox": [x, y, w, h]}
                        )
                    except ValueError:
                        continue

        data.append(item)

    # Save to cache
    np.save(cache_path, np.array(data, dtype=object))

    return data


def collate_fn(batch):
    """
    Custom collate function for object detection.
    Handles stacking of images and dense targets, while keeping variable-length
    metadata (like image_ids) in lists.

    Args:
        batch (list): List of samples from Dataset.__getitem__.
                      Expected format: {'image': tensor, 'target': dict, 'image_id': str, 'original_shape': tuple}

    Returns:
        dict: Collated batch.
    """
    images = []
    targets = []
    image_ids = []
    original_shapes = []

    for sample in batch:
        images.append(sample["image"])
        targets.append(sample["target"])
        image_ids.append(sample["image_id"])
        original_shapes.append(sample["original_shape"])

    # Stack images
    images = torch.stack(images, dim=0)

    # Stack targets if they are dense tensors (CenterNet style)
    collated_targets = {}
    if len(targets) > 0 and isinstance(targets[0], dict):
        keys = targets[0].keys()
        for k in keys:
            if isinstance(targets[0][k], torch.Tensor):
                collated_targets[k] = torch.stack([t[k] for t in targets], dim=0)
            else:
                # Fallback for non-tensor targets
                collated_targets[k] = [t[k] for t in targets]

    return {
        "image": images,
        "target": collated_targets,
        "image_id": image_ids,
        "original_shape": original_shapes,
    }


def post_process_coords(x, y, original_shape, input_size=IMG_SIZE):
    """
    Maps coordinate (x, y) from the resized/padded input space back to the original image space.
    Assumes the preprocessing involved resizing the longest edge to input_size and padding the shorter edge.

    Args:
        x (float or np.array): X coordinate in model input space (0 to input_size).
        y (float or np.array): Y coordinate in model input space (0 to input_size).
        original_shape (tuple): (height, width) of the original image.
        input_size (int): The size of the square input image.

    Returns:
        tuple: (orig_x, orig_y) in original image coordinates.
    """
    orig_h, orig_w = original_shape

    # Calculate scaling factor
    scale = min(input_size / orig_h, input_size / orig_w)

    # Calculate new dimensions after resizing
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)

    # Calculate padding
    pad_y = (input_size - new_h) / 2
    pad_x = (input_size - new_w) / 2

    # Reverse transformation: (x_input - pad) / scale
    orig_x = (x - pad_x) / scale
    orig_y = (y - pad_y) / scale

    return orig_x, orig_y
