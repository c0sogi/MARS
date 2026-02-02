import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from library.config import Config
from library.utils import set_seed, clipped_log_loss
from library.data_loader import load_datasets
from library.preprocessors import get_preprocessor
from library.models import get_expert_model
from library.ensemble import GreedySelector


def train_and_predict_expert(expert_config, X_train, y_train, X_eval):
    """
    Constructs, trains, and predicts with a single expert pipeline.

    Args:
        expert_config (dict): Configuration dictionary for the expert.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_eval (np.ndarray): Evaluation features (Validation or Test).

    Returns:
        np.ndarray: Predicted probabilities on X_eval.
    """
    # 1. Get Components
    preprocessor = get_preprocessor(expert_config["basis"])
    model = get_expert_model(expert_config["model"])

    # 2. Build Pipeline
    # We use a pipeline to ensure preprocessing statistics (e.g., quantiles, means)
    # are computed on the training set and applied correctly to the evaluation set.
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # 3. Train
    pipeline.fit(X_train, y_train)

    # 4. Predict
    # Predict probabilities
    preds = pipeline.predict_proba(X_eval)

    return preds.astype(Config.FLOAT_PRECISION)


def run_mrgde_pipeline(load_cached_data=True):
    """
    Orchestrates the Multi-Resolution Gaussianized Dynamic Ensemble (MRGDE) pipeline.

    Steps:
    1. Load Data (Train/Val/Test).
    2. Phase 1: Train all experts on Train split, predict on Val split.
    3. Selection: Use GreedySelector to find best ensemble weights.
    4. Phase 2: Retrain selected experts on Full (Train+Val) data.
    5. Inference: Predict on Test data.
    6. Submission: Generate submission file.

    Args:
        load_cached_data (bool): Whether to load intermediate validation predictions from cache.
    """
    set_seed(Config.RANDOM_SEED)

    # =========================================================================
    # 1. Load Data
    # =========================================================================
    print("\n[Pipeline] Loading Datasets...")
    data = load_datasets(load_cached_data=load_cached_data)

    classes = data["classes"]

    # Train Split
    y_train = data["train"]["y"]
    X_train_views = data["train"]["views"]

    # Validation Split
    y_val = data["val"]["y"]
    X_val_views = data["val"]["views"]

    # Test Split
    ids_test = data["test"]["ids"]
    X_test_views = data["test"]["views"]

    # =========================================================================
    # 2. Phase 1: Library Evaluation (Train -> Val)
    # =========================================================================
    print("\n[Pipeline] Phase 1: Evaluating Expert Library on Validation Set...")

    expert_library = Config.get_expert_library()
    val_preds_cache_path = os.path.join(Config.WORKING_DIR, "val_preds_dict.npy")

    val_preds_dict = {}

    # Attempt to load validation predictions from cache
    if load_cached_data and os.path.exists(val_preds_cache_path):
        print(f"Loading validation predictions from {val_preds_cache_path}...")
        try:
            val_preds_dict = np.load(val_preds_cache_path, allow_pickle=True).item()
            # Verify all experts in library are in cache
            missing_experts = [
                e["id"] for e in expert_library if e["id"] not in val_preds_dict
            ]
            if missing_experts:
                print(f"Cache incomplete. Missing: {missing_experts}. Recomputing...")
                val_preds_dict = {}  # Reset to force recompute
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")
            val_preds_dict = {}

    if not val_preds_dict:
        print(f"Training {len(expert_library)} experts...")

        for i, expert in enumerate(expert_library):
            eid = expert["id"]
            view_name = expert["view"]

            print(f"  [{i+1}/{len(expert_library)}] Training Expert: {eid}")

            # Select Feature View
            X_tr = X_train_views[view_name]
            X_v = X_val_views[view_name]

            # Train and Predict
            try:
                preds = train_and_predict_expert(expert, X_tr, y_train, X_v)
                val_preds_dict[eid] = preds
            except Exception as e:
                print(f"    FAILED: {eid} - {e}")

        # Save cache
        print(f"Saving validation predictions to {val_preds_cache_path}...")
        np.save(val_preds_cache_path, val_preds_dict)

    # =========================================================================
    # 3. Selection
    # =========================================================================
    print("\n[Pipeline] Running Greedy Forward Selection...")

    selector = GreedySelector(max_iterations=100, tolerance=1e-6)
    selected_weights = selector.fit(val_preds_dict, y_val)

    print(f"Best Validation Log Loss: {selector.best_score}")

    # =========================================================================
    # 4. Phase 2: Retraining & Inference (Full -> Test)
    # =========================================================================
    print("\n[Pipeline] Phase 2: Retraining Selected Experts on Full Data...")

    test_preds_dict = {}

    # Prepare Full Datasets (Train + Val)
    X_full_views = {}
    for view_name in [Config.VIEW_GLOBAL, Config.VIEW_COMBINED]:
        X_full_views[view_name] = np.vstack(
            [X_train_views[view_name], X_val_views[view_name]]
        )

    y_full = np.concatenate([y_train, y_val])

    # Iterate only through selected experts
    selected_expert_ids = list(selected_weights.keys())

    for eid in selected_expert_ids:
        # Find config for this ID
        expert_config = next((e for e in expert_library if e["id"] == eid), None)
        if expert_config is None:
            raise ValueError(f"Selected expert {eid} not found in library.")

        print(f"  Retraining Expert: {eid}")

        view_name = expert_config["view"]
        X_full = X_full_views[view_name]
        X_test = X_test_views[view_name]

        # Train on Full, Predict on Test
        preds = train_and_predict_expert(expert_config, X_full, y_full, X_test)
        test_preds_dict[eid] = preds

    # =========================================================================
    # 5. Submission Generation
    # =========================================================================
    print("\n[Pipeline] Generating Submission...")

    # Aggregate predictions using the selector
    final_test_probs = selector.predict(test_preds_dict)

    # Create DataFrame
    # Columns: id, species_1, species_2, ...
    submission_df = pd.DataFrame(final_test_probs, columns=classes)
    submission_df.insert(0, "id", ids_test)

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    print("Pipeline Complete.")
