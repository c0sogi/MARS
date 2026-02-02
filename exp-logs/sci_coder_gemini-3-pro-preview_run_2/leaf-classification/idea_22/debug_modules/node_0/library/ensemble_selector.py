import numpy as np
from collections import Counter
from library.config import Config
from library.utils import calculate_log_loss
from library.expert_models import get_lda_expert, get_lr_expert


class GreedyEnsembleSelector:
    """
    Performs Greedy Forward Selection to determine the optimal ensemble weights.
    This selector iteratively adds the expert that minimizes the log loss of the
    ensemble average on the validation set.
    """

    def __init__(self, n_iterations=None):
        """
        Args:
            n_iterations (int, optional): Number of selection steps (experts to add).
                                          Defaults to Config.SELECTION_ITERATIONS.
        """
        self.n_iterations = (
            n_iterations if n_iterations is not None else Config.SELECTION_ITERATIONS
        )
        self.selected_experts = []
        self.best_score = float("inf")

    def fit(self, predictions_dict, y_true):
        """
        Iteratively selects experts to minimize log loss.

        Args:
            predictions_dict (dict): Dictionary where keys are expert names and values
                                     are (n_samples, n_classes) probability arrays.
            y_true (array-like): Ground truth labels (integers).

        Returns:
            dict: A dictionary mapping expert names to their integer weights (counts).
        """
        available_experts = list(predictions_dict.keys())
        # Assume all predictions have same shape
        first_pred = list(predictions_dict.values())[0]
        n_samples, n_classes = first_pred.shape

        # Initialize ensemble sum (unscaled probabilities)
        # We sum probabilities and divide by count k to get average
        current_ensemble_sum = np.zeros((n_samples, n_classes), dtype=np.float64)
        current_k = 0

        self.selected_experts = []
        self.best_score = float("inf")

        print(
            f"Starting Greedy Forward Selection for {self.n_iterations} iterations..."
        )

        for i in range(self.n_iterations):
            iteration_best_score = float("inf")
            iteration_best_expert = None

            # Try adding each expert to the current ensemble
            for expert_name in available_experts:
                candidate_preds = predictions_dict[expert_name]

                # Calculate candidate ensemble average
                # New Average = (Current Sum + Candidate Preds) / (Current Count + 1)
                temp_sum = current_ensemble_sum + candidate_preds
                temp_avg = temp_sum / (current_k + 1)

                # Calculate metric
                score = calculate_log_loss(y_true, temp_avg)

                if score < iteration_best_score:
                    iteration_best_score = score
                    iteration_best_expert = expert_name

            # Update ensemble with the winner of this round
            if iteration_best_expert is not None:
                self.selected_experts.append(iteration_best_expert)
                current_ensemble_sum += predictions_dict[iteration_best_expert]
                current_k += 1
                self.best_score = iteration_best_score

                print(
                    f"Iteration {i+1}/{self.n_iterations}: Added {iteration_best_expert}, Validation Log Loss: {self.best_score}"
                )
            else:
                print(f"Iteration {i+1}/{self.n_iterations}: No improvement found.")
                break

        # Convert list of selected experts to weights (counts)
        weights = dict(Counter(self.selected_experts))
        return weights


def run_selection_phase(data_loader):
    """
    Orchestrates Phase 1: Training experts on Train split and selecting them on Val split.

    This function:
    1. Loads Phase 1 data (Train/Val split).
    2. Trains the 4 designated experts (Anchor, Orthogonal, Synergistic, Backup).
    3. Generates validation predictions.
    4. Runs Greedy Forward Selection to find optimal weights.

    Args:
        data_loader (DataLoader): Instance of the DataLoader class.

    Returns:
        dict: Optimal weights for the experts (e.g., {'Anchor': 3, 'Orthogonal': 1}).
    """
    print("\n=== Phase 1: Expert Selection ===")

    # 1. Get Data
    # Returns dict with 'train', 'val', 'classes'
    # Data is already preprocessed (Gaussianized) and cast to float64
    data = data_loader.get_phase1_data()

    X_anc_tr = data["train"]["anchor"]
    X_ort_tr = data["train"]["orthogonal"]
    X_syn_tr = data["train"]["synergistic"]
    y_tr = data["train"]["y"]

    X_anc_val = data["val"]["anchor"]
    X_ort_val = data["val"]["orthogonal"]
    X_syn_val = data["val"]["synergistic"]
    y_val = data["val"]["y"]  # Encoded labels

    # 2. Train Experts
    print("Training experts on training split...")

    # Expert 1: Global Anchor (LDA on Provided Features)
    # Uses provided Margin, Shape, Texture histograms
    model_anchor = get_lda_expert()
    model_anchor.fit(X_anc_tr, y_tr)

    # Expert 2: Orthogonal Morphometric (LDA on Extracted Shape Features)
    # Uses Hu Moments and Geometric Scalars extracted from images
    model_ortho = get_lda_expert()
    model_ortho.fit(X_ort_tr, y_tr)

    # Expert 3: Synergistic (LDA on Combined Features)
    # Uses concatenation of Provided and Extracted features
    model_syn = get_lda_expert()
    model_syn.fit(X_syn_tr, y_tr)

    # Expert 4: Discriminative Backup (Calibrated LR on Provided Features)
    # Provides a safety net if Gaussian assumptions fail
    model_backup = get_lr_expert()
    model_backup.fit(X_anc_tr, y_tr)

    print("Generating validation predictions...")

    # 3. Generate Predictions (Float64)
    # Ensure predictions are strictly float64 for numerical stability in log loss
    preds_anchor = model_anchor.predict_proba(X_anc_val).astype(np.float64)
    preds_ortho = model_ortho.predict_proba(X_ort_val).astype(np.float64)
    preds_syn = model_syn.predict_proba(X_syn_val).astype(np.float64)
    preds_backup = model_backup.predict_proba(X_anc_val).astype(np.float64)

    predictions_dict = {
        "Anchor": preds_anchor,
        "Orthogonal": preds_ortho,
        "Synergistic": preds_syn,
        "Backup": preds_backup,
    }

    # 4. Run Selection
    selector = GreedyEnsembleSelector()
    weights = selector.fit(predictions_dict, y_val)

    print("\nSelected Ensemble Weights:")
    for expert, weight in weights.items():
        print(f"  {expert}: {weight}")

    return weights
