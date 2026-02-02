import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
from scipy.optimize import minimize_scalar

from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    STREAMS,
    SEED,
    NUM_CLASSES,
    get_model_head_path,
)
from library.feature_engine import extract_features
from library.dataset import get_class_mapping

# Ensure reproducibility
np.random.seed(SEED)


def get_fused_path(stream_name, split, kind):
    """
    Generates path for fused artifacts.
    kind: 'X' (features), 'y' (labels), 'ids' (identifiers)
    """
    return os.path.join(WORKING_DIR, f"{stream_name}_{split}_fused_{kind}.npy")


def load_and_fuse_data(stream_config, split, load_cached_data=True):
    """
    Loads embeddings for all views, concatenates them (Early Fusion), and returns fused data.
    Implements caching for the fused result.
    """
    stream_name = stream_config["name"]

    # Paths for fused data
    path_X = get_fused_path(stream_name, split, "X")
    path_y = get_fused_path(stream_name, split, "y")
    path_ids = get_fused_path(stream_name, split, "ids")

    # Check cache
    if (
        load_cached_data
        and os.path.exists(path_X)
        and os.path.exists(path_y)
        and os.path.exists(path_ids)
    ):
        print(f"[{stream_name} | {split}] Loading fused features from cache...")
        X = np.load(path_X)
        y = np.load(path_y)
        ids = np.load(path_ids)
        return X, y, ids

    print(f"[{stream_name} | {split}] Generating fused features...")

    # Ensure raw embeddings exist
    # extract_features handles its own caching logic
    view_paths = extract_features(
        stream_config, split, load_cached_data=load_cached_data
    )

    # Load data for all views
    views_data = {}
    views_order = ["global", "standard", "local"]  # Fixed order for concatenation

    for view in views_order:
        if view not in view_paths:
            raise ValueError(f"Missing view configuration: {view}")

        p = view_paths[view]
        views_data[view] = {
            "emb": np.load(p["embeddings"]),
            "ids": np.load(p["ids"]),
            "lbl": np.load(p["labels"]),
        }

    # Validation: Ensure IDs align across views
    # Since feature_engine extracts all views in one pass per image, they should align.
    # We check strictly to avoid data corruption.
    ref_ids = views_data["global"]["ids"]
    ref_lbl = views_data["global"]["lbl"]

    for view in views_order[1:]:
        if not np.array_equal(ref_ids, views_data[view]["ids"]):
            raise ValueError(
                f"ID mismatch between global and {view} views in {split} split."
            )
        if not np.array_equal(ref_lbl, views_data[view]["lbl"]):
            # Labels should definitely match
            raise ValueError(
                f"Label mismatch between global and {view} views in {split} split."
            )

    # Concatenate Features (Early Fusion)
    # Shape: (N, D_global + D_standard + D_local)
    feature_list = [views_data[view]["emb"] for view in views_order]
    X_fused = np.concatenate(feature_list, axis=1)

    y_fused = ref_lbl
    ids_fused = ref_ids

    # Save to cache
    print(f"[{stream_name} | {split}] Saving fused features to {WORKING_DIR}...")
    np.save(path_X, X_fused)
    np.save(path_y, y_fused)
    np.save(path_ids, ids_fused)

    return X_fused, y_fused, ids_fused


def train_stream(stream_config, load_cached_model=True):
    """
    Trains the LogisticRegressionCV head for a specific stream.
    Returns validation predictions, test predictions, and targets.
    """
    stream_name = stream_config["name"]
    model_path = get_model_head_path(stream_name)

    # 1. Prepare Data
    print(f"\n=== Processing {stream_name} ===")
    X_train, y_train, _ = load_and_fuse_data(stream_config, "train")
    X_val, y_val, val_ids = load_and_fuse_data(stream_config, "val")
    X_test, _, test_ids = load_and_fuse_data(stream_config, "test")

    # 2. Train or Load Model
    if load_cached_model and os.path.exists(model_path):
        print(f"[{stream_name}] Loading trained head from {model_path}...")
        clf = joblib.load(model_path)
    else:
        print(f"[{stream_name}] Training LogisticRegressionCV head...")
        # Solver 'saga' is faster for large datasets; 'multinomial' for multi-class log loss
        # Cs=10 attempts 10 values on log scale. cv=5 for robust tuning.
        clf = LogisticRegressionCV(
            Cs=10,
            cv=5,
            solver="saga",
            multi_class="multinomial",
            max_iter=2000,  # Increased to ensure convergence
            random_state=SEED,
            n_jobs=-1,
            verbose=0,
        )
        clf.fit(X_train, y_train)

        print(f"[{stream_name}] Saving model...")
        joblib.dump(clf, model_path)

    # 3. Evaluate
    print(f"[{stream_name}] Generating predictions...")
    val_probs = clf.predict_proba(X_val)
    test_probs = clf.predict_proba(X_test)

    # Metric Check
    loss = log_loss(y_val, val_probs)
    print(f"[{stream_name}] Validation Log Loss: {loss:.8f}")

    return {
        "val_probs": val_probs,
        "test_probs": test_probs,
        "val_y": y_val,
        "val_ids": val_ids,
        "test_ids": test_ids,
    }


def optimize_ensemble(res_a, res_b):
    """
    Finds optimal weight w for: P = w * P_a + (1-w) * P_b
    Minimizes Log Loss on Validation set.
    """
    print("\n=== Optimizing Ensemble Weights ===")

    y_true = res_a["val_y"]
    p_a = res_a["val_probs"]
    p_b = res_b["val_probs"]

    # Sanity check
    assert np.array_equal(
        y_true, res_b["val_y"]
    ), "Validation labels mismatch between streams"

    def objective(w):
        # Constrain w to [0, 1] implicitly or explicitly
        # Here we use bounds in minimize_scalar
        p_blend = w * p_a + (1 - w) * p_b
        # Clip to avoid log(0)
        p_blend = np.clip(p_blend, 1e-15, 1 - 1e-15)
        return log_loss(y_true, p_blend)

    # Minimize
    res = minimize_scalar(objective, bounds=(0, 1), method="bounded")
    best_w = res.x
    best_loss = res.fun

    print(f"Optimal Weight for Stream A (ConvNeXt): {best_w:.4f}")
    print(f"Optimal Weight for Stream B (MaxViT):  {1 - best_w:.4f}")
    print(f"Combined Validation Log Loss: {best_loss:.8f}")

    return best_w


def run_pipeline(load_cached_data=True, load_cached_model=True):
    """
    Main entry point. Runs both streams, optimizes ensemble, generates submission.
    """
    # 1. Train Stream A (ConvNeXt)
    res_a = train_stream(STREAMS["stream_a"], load_cached_model=load_cached_model)

    # 2. Train Stream B (MaxViT)
    res_b = train_stream(STREAMS["stream_b"], load_cached_model=load_cached_model)

    # 3. Optimize Ensemble
    w_a = optimize_ensemble(res_a, res_b)
    w_b = 1.0 - w_a

    # 4. Generate Test Predictions
    print("\n=== Generating Submission ===")
    test_probs_a = res_a["test_probs"]
    test_probs_b = res_b["test_probs"]

    final_probs = w_a * test_probs_a + w_b * test_probs_b

    # 5. Format Submission
    # Get class mapping to ensure correct column order
    _, idx_to_breed = get_class_mapping()
    breed_cols = [idx_to_breed[i] for i in range(NUM_CLASSES)]

    # Create DataFrame
    # IDs should match between streams for test set
    test_ids = res_a["test_ids"]
    assert np.array_equal(test_ids, res_b["test_ids"]), "Test IDs mismatch"

    df_sub = pd.DataFrame(final_probs, columns=breed_cols)
    df_sub.insert(0, "id", test_ids)

    # Save
    print(f"Saving submission to {SUBMISSION_PATH}...")
    df_sub.to_csv(SUBMISSION_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    # This block is just for testing the module directly,
    # though the task says "DO NOT include an if __name__ == '__main__': block"
    # for the submitted file content.
    # I will strictly follow the instruction and NOT include this in the final block.
    pass
