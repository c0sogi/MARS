import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from library.config import Config
from library.models import WhaleModel
from library.dataset import get_test_loader, get_dataloaders
from library.utils import seed_everything


def predict_on_loader(model, loader, device):
    """
    Runs inference on a DataLoader and returns a flat array of probabilities.
    Handles both (image, target) and (image, clip_name) batches.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            # batch[0] is always images
            images = batch[0].to(device, dtype=torch.float)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().detach().numpy().flatten())

    return np.array(all_probs)


def load_model_checkpoint(model_name, checkpoint_path, device):
    """
    Loads a model architecture and its weights from a checkpoint.
    """
    # Initialize model (pretrained=False because we are loading specific weights)
    model = WhaleModel(model_name=model_name, pretrained=False)
    model.to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support both full checkpoint dicts and direct state dicts
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_checkpoint_paths(checkpoint_dir):
    """
    Scans the directory for checkpoints matching the ensemble pattern.
    Pattern: {model_name}_fold_{fold}_best_{metric}.pth
    """
    paths = []
    model_names = Config.model_names
    metrics = ["auc", "loss"]
    folds = range(Config.n_folds)

    for model_name in model_names:
        for fold in folds:
            for metric in metrics:
                filename = f"{model_name}_fold_{fold}_best_{metric}.pth"
                full_path = os.path.join(checkpoint_dir, filename)
                if os.path.exists(full_path):
                    paths.append(
                        {
                            "path": full_path,
                            "model_name": model_name,
                            "fold": fold,
                            "metric": metric,
                        }
                    )
    return paths


def generate_pseudo_labels(round1_checkpoint_dir, load_cached_data=True):
    """
    Generates pseudo-labels using Round 1 models via Consensus-Based Self-Distillation.
    Creates an expanded training CSV containing original train data + high-confidence test samples.
    """
    output_csv = os.path.join(Config.working_dir, "pseudo_train.csv")

    # Return cached result if available
    if load_cached_data and os.path.exists(output_csv):
        print(f"Loading cached pseudo-label dataset from {output_csv}")
        return output_csv

    print("Generating pseudo-labels from Round 1 models...")

    # 1. Get Test Data
    test_loader = get_test_loader(load_cached_data=load_cached_data)
    # Access clip names from the dataset to map back to files
    test_clips = test_loader.dataset.clip_names

    # 2. Gather Checkpoints
    ckpt_infos = get_checkpoint_paths(round1_checkpoint_dir)
    if not ckpt_infos:
        raise RuntimeError(f"No valid checkpoints found in {round1_checkpoint_dir}")

    # 3. Generate Predictions (Ensemble)
    all_preds = []
    device = Config.device

    for info in ckpt_infos:
        print(f"  Inference: {os.path.basename(info['path'])}")
        model = load_model_checkpoint(info["model_name"], info["path"], device)
        preds = predict_on_loader(model, test_loader, device)
        all_preds.append(preds)

    # Shape: (N_test_samples, N_models)
    all_preds = np.array(all_preds).T

    # 4. Uncertainty Filtering
    # Calculate consensus statistics
    mean_preds = np.mean(all_preds, axis=1)
    std_preds = np.std(all_preds, axis=1)

    # Filter Logic: High Confidence AND Low Variance
    # Confidence: > 0.95 (Call) or < 0.05 (Noise)
    high_conf = (mean_preds > Config.pseudo_label_confidence_threshold) | (
        mean_preds < (1.0 - Config.pseudo_label_confidence_threshold)
    )

    # Variance: < 0.05
    low_var = std_preds < Config.pseudo_label_uncertainty_threshold

    mask = high_conf & low_var
    selected_indices = np.where(mask)[0]

    print(
        f"  Consensus Reached: {len(selected_indices)} samples selected out of {len(test_clips)}."
    )

    # 5. Construct Expanded Dataset
    # Load original training metadata
    train_df = pd.read_csv(Config.train_csv)

    # Load test metadata to get file paths
    test_metadata = pd.read_csv(Config.test_csv)

    # Extract selected test samples
    pseudo_df = test_metadata.iloc[selected_indices].copy()

    # Assign labels based on mean prediction (rounded to 0 or 1)
    pseudo_labels = (mean_preds[selected_indices] > 0.5).astype(int)
    pseudo_df["label"] = pseudo_labels

    # Ensure columns match train_df
    pseudo_df = pseudo_df[["file_path", "label"]]

    # Concatenate
    expanded_df = pd.concat([train_df, pseudo_df], axis=0, ignore_index=True)

    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    expanded_df.to_csv(output_csv, index=False)
    print(f"Expanded dataset saved to {output_csv}")

    return output_csv


def generate_oof_features(round2_checkpoint_dir, load_cached_data=True):
    """
    Generates Out-Of-Fold (OOF) features for the Meta-Learner.
    Since the validation split is fixed, this aggregates predictions on the validation set
    from all Round 2 models.

    Returns:
        X (np.ndarray): Feature matrix (N_val, 4).
        y (np.ndarray): Target vector (N_val,).
        sorted_keys (list): List of (model_name, metric) tuples defining column order.
    """
    print("Generating OOF features for Meta-Learner...")

    # 1. Get Validation Data
    _, val_loader = get_dataloaders(load_cached_data=load_cached_data)
    val_targets = val_loader.dataset.targets

    # 2. Gather Checkpoints
    ckpt_infos = get_checkpoint_paths(round2_checkpoint_dir)

    # 3. Aggregate Predictions by Type (Bagging)
    # We group predictions by (Model, Metric) and average across folds/seeds.
    feature_map = {}  # Key: (model_name, metric), Value: list of prediction arrays
    device = Config.device

    for info in ckpt_infos:
        key = (info["model_name"], info["metric"])
        if key not in feature_map:
            feature_map[key] = []

        # print(f"  Processing OOF: {os.path.basename(info['path'])}")
        model = load_model_checkpoint(info["model_name"], info["path"], device)
        preds = predict_on_loader(model, val_loader, device)
        feature_map[key].append(preds)

    # 4. Stack Features
    # Ensure deterministic order of columns
    sorted_keys = sorted(feature_map.keys())

    final_features = []
    for key in sorted_keys:
        preds_list = feature_map[key]
        if not preds_list:
            continue
        # Average across the 5 folds/seeds
        avg_preds = np.mean(np.array(preds_list), axis=0)
        final_features.append(avg_preds)

    if not final_features:
        raise RuntimeError("No features generated. Check round 2 checkpoints.")

    X = np.column_stack(final_features)  # Shape: (N_val, 4)
    y = np.array(val_targets)

    return X, y, sorted_keys


def train_meta_learner(X, y, save_path):
    """
    Trains the Logistic Regression Meta-Learner on OOF features.
    """
    print(f"Training Meta-Learner (Logistic Regression) on {len(y)} samples...")

    # Use fixed seed for reproducibility
    clf = LogisticRegression(random_state=Config.seed, solver="liblinear")
    clf.fit(X, y)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(clf, f)

    print(f"Meta-Learner saved to {save_path}")
    return clf


def predict_submission(
    round2_checkpoint_dir, meta_learner_path, output_path, load_cached_data=True
):
    """
    Generates the final submission using the Bagged-Stacking Ensemble.
    1. Generates predictions on Test set using Round 2 models.
    2. Bags predictions by model type.
    3. Stacks using the Meta-Learner.
    """
    print("Generating Final Submission...")

    # 1. Get Test Data
    test_loader = get_test_loader(load_cached_data=load_cached_data)
    clips = test_loader.dataset.clip_names

    # 2. Load Meta-Learner
    if not os.path.exists(meta_learner_path):
        raise FileNotFoundError(f"Meta-learner not found at {meta_learner_path}")

    with open(meta_learner_path, "rb") as f:
        clf = pickle.load(f)

    # 3. Generate Bagged Features (Same logic as OOF)
    ckpt_infos = get_checkpoint_paths(round2_checkpoint_dir)
    feature_map = {}
    device = Config.device

    for info in ckpt_infos:
        key = (info["model_name"], info["metric"])
        if key not in feature_map:
            feature_map[key] = []

        # print(f"  Predicting Test: {os.path.basename(info['path'])}")
        model = load_model_checkpoint(info["model_name"], info["path"], device)
        preds = predict_on_loader(model, test_loader, device)
        feature_map[key].append(preds)

    # Ensure same column order as training
    sorted_keys = sorted(feature_map.keys())

    final_features = []
    for key in sorted_keys:
        preds_list = feature_map[key]
        if not preds_list:
            continue
        # Average across folds/seeds
        avg_preds = np.mean(np.array(preds_list), axis=0)
        final_features.append(avg_preds)

    X_test = np.column_stack(final_features)

    # 4. Meta-Inference
    # Predict probability of class 1
    final_probs = clf.predict_proba(X_test)[:, 1]

    # 5. Save Submission
    df = pd.DataFrame({"clip": clips, "probability": final_probs})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
