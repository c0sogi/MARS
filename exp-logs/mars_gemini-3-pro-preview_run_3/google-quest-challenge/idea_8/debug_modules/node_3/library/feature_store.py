import os
import numpy as np
import torch
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_dataloaders, get_tokenizer
from library.model_core import BackboneWrapper


def _run_inference(model, loader, device):
    """
    Runs inference on a dataloader and extracts raw embeddings and targets.

    Args:
        model: The PyTorch model (BackboneWrapper).
        loader: The DataLoader.
        device: The torch device.

    Returns:
        dict: Dictionary containing concatenated numpy arrays for h_cls, h_q, h_a, targets, and qa_ids.
    """
    model.eval()

    # Storage lists
    h_cls_list = []
    h_q_list = []
    h_a_list = []
    targets_list = []
    qa_ids_list = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            q_mask = batch["q_mask"].to(device)
            a_mask = batch["a_mask"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                q_mask=q_mask,
                a_mask=a_mask,
            )

            features = outputs["features"]

            # Move to CPU and store
            # h_cls is always present
            h_cls_list.append(features["h_cls"].cpu().numpy())

            # h_q and h_a should be present based on our data loader logic
            if "h_q" in features:
                h_q_list.append(features["h_q"].cpu().numpy())
            if "h_a" in features:
                h_a_list.append(features["h_a"].cpu().numpy())

            # Store labels if present (Validation set)
            if "labels" in batch:
                targets_list.append(batch["labels"].cpu().numpy())

            # Store QA IDs for tracking
            if "qa_id" in batch:
                qa_ids_list.append(batch["qa_id"].numpy())

    # Concatenate results
    results = {
        "h_cls": np.concatenate(h_cls_list, axis=0) if h_cls_list else None,
        "h_q": np.concatenate(h_q_list, axis=0) if h_q_list else None,
        "h_a": np.concatenate(h_a_list, axis=0) if h_a_list else None,
        "qa_ids": np.concatenate(qa_ids_list, axis=0) if qa_ids_list else None,
    }

    if targets_list:
        results["targets"] = np.concatenate(targets_list, axis=0)
    else:
        results["targets"] = None

    return results


def extract_and_save_features(
    base_model_name,
    checkpoint_path,
    fold_idx,
    model_tag,
    debug=False,
    load_cached_data=True,
):
    """
    Loads a fine-tuned model, runs inference on Validation (Holdout) and Test sets,
    computes topology-aware features, and saves them to disk as .npy files.

    Args:
        base_model_name (str): HF model name (e.g., 'microsoft/deberta-v3-large') used to init config.
        checkpoint_path (str): Path to the .pth file containing fine-tuned model weights.
        fold_idx (int): Fold index (used for file naming).
        model_tag (str): Tag for the model type (e.g., 'deberta', 'mpnet').
        debug (bool): If True, runs on a subset of data.
        load_cached_data (bool): If True, checks for existing files before running.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. Define Output Paths
    # --------------------------------------------------------------------------
    # Validation (OOF) Paths
    path_val_q = os.path.join(
        Config.WORKING_DIR, f"{model_tag}_fold{fold_idx}_val_features_Q.npy"
    )
    path_val_a = os.path.join(
        Config.WORKING_DIR, f"{model_tag}_fold{fold_idx}_val_features_A.npy"
    )
    path_val_t = os.path.join(
        Config.WORKING_DIR, f"{model_tag}_fold{fold_idx}_val_targets.npy"
    )
    path_val_ids = os.path.join(
        Config.WORKING_DIR, f"{model_tag}_fold{fold_idx}_val_ids.npy"
    )

    # Test Paths
    path_test_q = os.path.join(
        Config.WORKING_DIR, f"{model_tag}_fold{fold_idx}_test_features_Q.npy"
    )
    path_test_a = os.path.join(
        Config.WORKING_DIR, f"{model_tag}_fold{fold_idx}_test_features_A.npy"
    )
    path_test_ids = os.path.join(
        Config.WORKING_DIR, f"{model_tag}_fold{fold_idx}_test_ids.npy"
    )

    # --------------------------------------------------------------------------
    # 2. Check Cache
    # --------------------------------------------------------------------------
    required_files = [path_val_q, path_val_a, path_val_t, path_test_q, path_test_a]
    if load_cached_data and all(os.path.exists(p) for p in required_files):
        print(
            f"[FeatureStore] Found cached features for {model_tag} Fold {fold_idx}. Skipping extraction."
        )
        return

    print(f"[FeatureStore] Extracting features for {model_tag} Fold {fold_idx}...")

    # --------------------------------------------------------------------------
    # 3. Initialize Model & Load Weights
    # --------------------------------------------------------------------------
    # We need the tokenizer to setup data loaders
    tokenizer = get_tokenizer(base_model_name)

    # Initialize architecture
    model = BackboneWrapper(base_model_name, num_labels=len(Config.TARGET_COLS))

    # Load state dict
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[FeatureStore] Loading weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)

    # --------------------------------------------------------------------------
    # 4. Get DataLoaders
    # --------------------------------------------------------------------------
    # val_loader serves as our OOF/Holdout set for stacking
    _, val_loader, test_loader = get_dataloaders(
        tokenizer=tokenizer,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    # --------------------------------------------------------------------------
    # 5. Process Validation Set (OOF)
    # --------------------------------------------------------------------------
    print("[FeatureStore] Processing Validation Set...")
    val_res = _run_inference(model, val_loader, device)

    # Construct Topology-Aware Features
    # Q-Feature: Just h_q
    val_feat_q = val_res["h_q"]

    # A-Feature: Concatenation of [h_cls, h_q, h_a, |h_q - h_a|]
    # Calculate interaction term
    h_diff_val = np.abs(val_res["h_q"] - val_res["h_a"])
    val_feat_a = np.concatenate(
        [val_res["h_cls"], val_res["h_q"], val_res["h_a"], h_diff_val], axis=1
    )

    # Save Validation Data
    np.save(path_val_q, val_feat_q)
    np.save(path_val_a, val_feat_a)
    np.save(path_val_t, val_res["targets"])
    np.save(path_val_ids, val_res["qa_ids"])

    # --------------------------------------------------------------------------
    # 6. Process Test Set
    # --------------------------------------------------------------------------
    print("[FeatureStore] Processing Test Set...")
    test_res = _run_inference(model, test_loader, device)

    # Construct Topology-Aware Features for Test
    test_feat_q = test_res["h_q"]

    h_diff_test = np.abs(test_res["h_q"] - test_res["h_a"])
    test_feat_a = np.concatenate(
        [test_res["h_cls"], test_res["h_q"], test_res["h_a"], h_diff_test], axis=1
    )

    # Save Test Data
    np.save(path_test_q, test_feat_q)
    np.save(path_test_a, test_feat_a)
    np.save(path_test_ids, test_res["qa_ids"])

    print(f"[FeatureStore] Successfully saved features to {Config.WORKING_DIR}")

    # Cleanup memory
    del model, val_res, test_res, val_feat_q, val_feat_a, test_feat_q, test_feat_a
    torch.cuda.empty_cache()
