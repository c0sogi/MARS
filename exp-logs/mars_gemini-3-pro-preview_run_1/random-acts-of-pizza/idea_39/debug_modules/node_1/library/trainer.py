import numpy as np
from library.config import RF_WEIGHT, MLP_WEIGHT
from library.utils import save_submission, set_seed, compute_auc
from library.model_tree import DualViewRandomForest
from library.model_nn import NeuralNetworkModel


class Trainer:
    """
    Manages the training, evaluation, and ensembling lifecycles for the
    Hybrid Ensemble solution (Dual-View Random Forest + Community-Aware MLP).
    """

    def __init__(self):
        """
        Initializes the Trainer and sets random seeds for reproducibility.
        """
        set_seed()

    def train_rf(self, load_cached_data=True):
        """
        Fits the Random Forest model (Stream A).

        Args:
            load_cached_data (bool): Whether to load pre-processed features from cache.

        Returns:
            tuple: (test_ids, test_preds, val_auc)
        """
        print("\n=== Stream A: Dual-View Random Forest ===")
        model = DualViewRandomForest()
        # Delegates the full pipeline (Load -> Train -> Val -> Test) to the model class
        return model.run(load_cached_data=load_cached_data)

    def train_mlp(self, load_cached_data=True):
        """
        Handles the Neural Network training loop (Stream B).

        Args:
            load_cached_data (bool): Whether to load pre-processed features from cache.

        Returns:
            tuple: (test_ids, test_preds, val_auc)
        """
        print("\n=== Stream B: Community-Aware Dual-Query MLP ===")
        model = NeuralNetworkModel()
        # Delegates the full pipeline (Load -> Train Loop -> Early Stopping -> Test) to the model class
        return model.run(load_cached_data=load_cached_data)

    def evaluate_model(self, y_true, y_pred):
        """
        Computes the validation AUC score.

        Args:
            y_true (array-like): True binary labels.
            y_pred (array-like): Predicted probabilities.

        Returns:
            float: Area Under the ROC Curve.
        """
        return compute_auc(y_true, y_pred)

    def predict_ensemble(self, rf_preds, mlp_preds):
        """
        Generates final predictions by averaging model outputs.

        Args:
            rf_preds (np.ndarray): Predictions from Random Forest.
            mlp_preds (np.ndarray): Predictions from MLP.

        Returns:
            np.ndarray: Weighted average probabilities.
        """
        print(
            f"\nEnsembling predictions with weights: RF={RF_WEIGHT}, MLP={MLP_WEIGHT}"
        )
        ensemble_preds = (rf_preds * RF_WEIGHT) + (mlp_preds * MLP_WEIGHT)
        return ensemble_preds

    def run(self, load_cached_data=True):
        """
        Orchestrates the full pipeline:
        1. Train RF
        2. Train MLP
        3. Ensemble Predictions
        4. Save Submission

        Args:
            load_cached_data (bool): Whether to use cached features.
        """
        # 1. Train Random Forest
        rf_ids, rf_preds, rf_auc = self.train_rf(load_cached_data=load_cached_data)

        # 2. Train MLP
        mlp_ids, mlp_preds, mlp_auc = self.train_mlp(load_cached_data=load_cached_data)

        # 3. Verify Alignment
        if not np.array_equal(rf_ids, mlp_ids):
            raise ValueError("Test IDs from Random Forest and MLP do not match!")

        # 4. Ensemble
        final_preds = self.predict_ensemble(rf_preds, mlp_preds)

        # 5. Summary Metrics (Full Precision)
        print("\n=== Final Ensemble Performance ===")
        print(f"Random Forest Validation AUC: {rf_auc}")
        print(f"MLP Validation AUC: {mlp_auc}")

        # 6. Save Submission
        save_submission(rf_ids, final_preds, filename="submission.csv")

        return final_preds
