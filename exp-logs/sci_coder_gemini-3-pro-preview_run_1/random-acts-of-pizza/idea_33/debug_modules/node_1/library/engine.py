import pandas as pd
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import save_submission
from library.data_loader import DataLoader
from library.feature_engine import FeatureEngine
from library.model_rf import RandomForestStream
from library.model_nn import NeuralNetworkStream


class Engine:
    """
    Orchestrates the Hybrid Ensemble pipeline:
    1. Data Loading
    2. Feature Engineering (RF & MLP streams)
    3. Model Training (Random Forest & MLP)
    4. Ensemble Prediction
    5. Submission Generation
    """

    def __init__(self):
        self.data_loader = DataLoader()
        self.feature_engine = FeatureEngine()
        self.rf_stream = RandomForestStream()
        self.nn_stream = NeuralNetworkStream()

    def train_mlp(self, mlp_data):
        """
        Manages the training lifecycle of the MLP by delegating to NeuralNetworkStream.
        The stream handles the training loop, optimizer, loss, and early stopping.

        Args:
            mlp_data (dict): Dictionary containing training and validation tensors.

        Returns:
            tuple: (Best Validation AUC, Validation Predictions)
        """
        return self.nn_stream.train(mlp_data)

    def predict_mlp(self, mlp_data_test):
        """
        Generates inference probabilities from the trained neural network.

        Args:
            mlp_data_test (dict): Dictionary containing test tensors.

        Returns:
            np.ndarray: Probability predictions for the positive class.
        """
        return self.nn_stream.predict(mlp_data_test)

    def run(self):
        """
        Executes the full pipeline: Data Loading -> Feature Engineering ->
        Dual-Stream Training -> Ensemble -> Submission.
        """
        # Ensure directories exist
        Config.setup()

        # 1. Load Data
        print("--- Step 1: Loading Data ---")
        df_train, df_val, df_test = self.data_loader.load_data(
            debug_size=Config.DEBUG_SAMPLE_SIZE
        )

        # 2. Feature Engineering
        print("\n--- Step 2: Feature Engineering ---")
        # process_data handles caching internally
        rf_data, mlp_data = self.feature_engine.process_data(
            df_train, df_val, df_test, load_cached_data=True
        )

        # 3. Stream A: Random Forest
        print("\n--- Step 3: Stream A (Random Forest) ---")
        # train handles model caching internally
        rf_val_auc, rf_val_preds = self.rf_stream.train(rf_data)
        rf_test_preds = self.rf_stream.predict(rf_data["test"]["X"])

        # 4. Stream B: MLP
        print("\n--- Step 4: Stream B (MLP) ---")
        # train handles model caching internally
        mlp_val_auc, mlp_val_preds = self.train_mlp(mlp_data)
        mlp_test_preds = self.predict_mlp(mlp_data["test"])

        # 5. Ensemble
        print("\n--- Step 5: Ensemble ---")
        w_rf = Config.WEIGHT_RF
        w_mlp = Config.WEIGHT_MLP

        # Normalize weights
        total_weight = w_rf + w_mlp
        w_rf = w_rf / total_weight
        w_mlp = w_mlp / total_weight

        print(f"Ensemble Weights -> RF: {w_rf:.2f}, MLP: {w_mlp:.2f}")

        # Validation Ensemble Evaluation
        val_preds_ensemble = (w_rf * rf_val_preds) + (w_mlp * mlp_val_preds)

        # Get validation targets (available in both data dicts)
        y_val = rf_data["val"]["y"]
        ensemble_auc = roc_auc_score(y_val, val_preds_ensemble)
        print(f"Ensemble Validation ROC AUC: {ensemble_auc}")

        # Test Ensemble Predictions
        test_preds_ensemble = (w_rf * rf_test_preds) + (w_mlp * mlp_test_preds)

        # 6. Submission
        print("\n--- Step 6: Generating Submission ---")
        submission_df = pd.DataFrame(
            {
                "request_id": df_test["request_id"],
                "requester_received_pizza": test_preds_ensemble,
            }
        )

        save_submission(submission_df)
        print("Pipeline Completed Successfully.")
