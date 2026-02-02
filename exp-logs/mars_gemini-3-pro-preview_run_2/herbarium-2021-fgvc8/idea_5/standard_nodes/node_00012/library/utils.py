import os
import json
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_taxonomy_mapping(load_cached_data=True):
    """
    Loads or creates the mapping from Species ID -> Family ID and Order ID.
    Caches the result as a parquet file.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(Config.TAXONOMY_MAP_PATH):
        try:
            df = pd.read_parquet(Config.TAXONOMY_MAP_PATH)
            return df
        except Exception:
            pass  # Fallback to recomputing if load fails

    # Load raw metadata
    # Note: This file is large, so we only want to do this once
    with open(Config.TRAIN_METADATA_JSON, "r") as f:
        data = json.load(f)

    categories = data["categories"]
    df = pd.DataFrame(categories)

    # We expect columns: 'id', 'name', 'family', 'order'
    # Rename 'id' to 'category_id' for consistency with other dataframes
    if "id" in df.columns:
        df.rename(columns={"id": "category_id"}, inplace=True)

    # Map strings to integers for family and order
    # Sort unique values to ensure deterministic mapping
    unique_families = sorted(df["family"].astype(str).unique())
    unique_orders = sorted(df["order"].astype(str).unique())

    family_map = {name: i for i, name in enumerate(unique_families)}
    order_map = {name: i for i, name in enumerate(unique_orders)}

    df["family_id"] = df["family"].map(family_map)
    df["order_id"] = df["order"].map(order_map)

    # Select relevant columns
    df_out = df[["category_id", "family_id", "order_id"]]

    # Save to cache
    df_out.to_parquet(Config.TAXONOMY_MAP_PATH, index=False)

    return df_out


def compute_class_priors(load_cached_data=True):
    """
    Computes the prior probability (frequency) of each class in the training set.
    Used for Post-Hoc Logit Adjustment.
    """
    cache_path = os.path.join(Config.WORK_DIR, "class_priors.parquet")
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        try:
            df_priors = pd.read_parquet(cache_path)
            # Ensure alignment by sorting
            df_priors = df_priors.sort_values("category_id")
            return df_priors["frequency"].values
        except Exception:
            pass

    # Load train csv
    df_train = pd.read_csv(Config.TRAIN_CSV)

    # Count frequencies
    counts = df_train["category_id"].value_counts().sort_index()

    # Initialize a full array to handle potential missing classes in the csv (though unlikely)
    total_classes = Config.NUM_CLASSES
    full_counts = np.zeros(total_classes, dtype=np.float64)

    # Fill with actual counts
    # counts.index contains the category_ids
    valid_indices = counts.index[counts.index < total_classes]
    full_counts[valid_indices] = counts[valid_indices].values

    # Calculate frequencies
    total_samples = len(df_train)
    frequencies = full_counts / total_samples

    # Create DataFrame for caching
    df_out = pd.DataFrame(
        {"category_id": np.arange(total_classes), "frequency": frequencies}
    )

    df_out.to_parquet(cache_path, index=False)

    return frequencies


def calculate_macro_f1(y_true, y_pred):
    """
    Calculates Macro F1 Score.
    """
    return f1_score(y_true, y_pred, average="macro")


def generate_submission(
    model, data_loader, device, class_priors=None, output_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set and saves to CSV.
    Applies Post-Hoc Logit Adjustment if class_priors are provided.
    """
    model.eval()
    ids = []
    preds = []

    # Pre-compute adjustment if priors are provided
    # Adjustment: logits = logits - log(priors)
    logit_adjustment = None
    if class_priors is not None:
        # Add epsilon to avoid log(0)
        priors_tensor = torch.tensor(class_priors, device=device, dtype=torch.float32)
        priors_tensor = priors_tensor + 1e-12
        logit_adjustment = torch.log(priors_tensor)

    with torch.no_grad():
        for batch in data_loader:
            # Unpack batch (assuming tuple of images, image_ids)
            images = batch[0].to(device)
            image_ids = batch[1]

            outputs = model(images)

            # Handle different model output formats (Tuple, Dict, Tensor)
            # We assume the first output or 'species' key is the species logits
            if isinstance(outputs, dict):
                species_logits = outputs.get("species", list(outputs.values())[0])
            elif isinstance(outputs, (tuple, list)):
                species_logits = outputs[0]
            else:
                species_logits = outputs

            # Apply Post-Hoc Logit Adjustment
            if logit_adjustment is not None:
                species_logits = species_logits - logit_adjustment

            # Get predictions
            predicted_labels = torch.argmax(species_logits, dim=1).cpu().numpy()

            ids.extend(image_ids.numpy())
            preds.extend(predicted_labels)

    # Create submission DataFrame
    df = pd.DataFrame({"Id": ids, "Predicted": preds})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    df.to_csv(output_path, index=False)
