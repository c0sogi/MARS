import numpy as np
from collections import Counter
from library.utils import clipped_log_loss
from library.config import FLOAT_PRECISION


def greedy_forward_selection(
    expert_results, expert_configs, y_true, max_iter=50, tol=1e-6
):
    """
    Performs Greedy Forward Selection to find the optimal ensemble of experts
    that minimizes validation log loss.

    Args:
        expert_results (dict): Output from training_engine.train_and_predict_experts.
                               Keys are expert_ids, values are dicts with 'val_preds' and 'frozen_pipeline'.
        expert_configs (list): List of expert configuration dictionaries (to retrieve 'view').
        y_true (np.ndarray): Ground truth labels for the validation set.
        max_iter (int): Maximum number of selection iterations.
        tol (float): Minimum improvement in log loss required to continue adding experts.

    Returns:
        list: A list of dictionaries representing the selected experts and their weights.
              Each dict has keys: 'id', 'weight', 'frozen_pipeline', 'view'.
    """
    print(f"Starting Ensemble Selection (Max Iter: {max_iter}, Tol: {tol})...")

    # 1. Prepare Metadata Lookups
    # Map ID to config (for view) and results (for preds/pipeline)
    expert_lookup = {}
    for config in expert_configs:
        eid = config["id"]
        if eid in expert_results:
            expert_lookup[eid] = {
                "view": config["view"],
                "val_preds": expert_results[eid]["val_preds"],
                "frozen_pipeline": expert_results[eid]["frozen_pipeline"],
            }

    available_ids = list(expert_lookup.keys())
    if not available_ids:
        raise ValueError("No expert results provided for selection.")

    # 2. Find Best Single Expert (Initialization)
    best_single_id = None
    best_single_score = float("inf")

    print("Evaluating individual baselines...")
    for eid in available_ids:
        preds = expert_lookup[eid]["val_preds"]
        score = clipped_log_loss(y_true, preds)
        # print(f"  - {eid}: {score:.6f}")

        if score < best_single_score:
            best_single_score = score
            best_single_id = eid

    print(f"Best single expert: {best_single_id} (Score: {best_single_score:.10f})")

    # 3. Initialize Ensemble
    # We start with the best single expert
    selected_experts = [best_single_id]
    current_ensemble_sum = expert_lookup[best_single_id]["val_preds"].copy()
    current_best_score = best_single_score

    # 4. Iterative Selection
    for i in range(max_iter):
        best_iter_id = None
        best_iter_score = float("inf")

        # Current size is len(selected_experts)
        # We test adding one more expert (size + 1)
        current_size = len(selected_experts)
        next_size = current_size + 1

        # Try adding each available expert (with replacement)
        for eid in available_ids:
            candidate_preds = expert_lookup[eid]["val_preds"]

            # Compute temporary ensemble average
            # (current_sum + candidate) / (n + 1)
            temp_ensemble_avg = (current_ensemble_sum + candidate_preds) / next_size

            score = clipped_log_loss(y_true, temp_ensemble_avg)

            if score < best_iter_score:
                best_iter_score = score
                best_iter_id = eid

        # Check Improvement
        improvement = current_best_score - best_iter_score

        if improvement > tol:
            # Accept addition
            selected_experts.append(best_iter_id)
            current_ensemble_sum += expert_lookup[best_iter_id]["val_preds"]
            current_best_score = best_iter_score
            print(
                f"Iter {i+1}: Added {best_iter_id}. New Score: {current_best_score:.10f} (Imp: {improvement:.10f})"
            )
        else:
            # Stop
            print(
                f"Iter {i+1}: No sufficient improvement ({improvement:.10f} <= {tol}). Stopping."
            )
            break

    # 5. Aggregate Weights
    # Count occurrences of each expert ID
    counts = Counter(selected_experts)

    # 6. Construct Output
    final_selection = []
    print("\nFinal Ensemble Composition:")
    for eid, weight in counts.items():
        info = expert_lookup[eid]
        print(f"  - {eid}: Weight {weight}")
        final_selection.append(
            {
                "id": eid,
                "weight": weight,
                "frozen_pipeline": info["frozen_pipeline"],
                "view": info["view"],
            }
        )

    print(f"Final Validation Score: {current_best_score:.10f}")

    return final_selection
