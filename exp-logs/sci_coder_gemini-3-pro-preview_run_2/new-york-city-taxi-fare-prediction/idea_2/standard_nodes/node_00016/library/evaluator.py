import os
import numpy as np
import pandas as pd
import xgboost as xgb
from library.config import PATH_CONFIG, XGB_PARAMS, TRAIN_CONFIG
from library.data_processor import TaxiDataProcessor


class ModelEvaluator:
    """
    Evaluator class responsible for loading the model, performing inference,
    calculating validation metrics, and generating the final submission file.
    """

    def __init__(self, model_path=None):
        """
        Initialize the evaluator.

        Args:
            model_path (str, optional): Path to the saved model file.
                                        Defaults to the path in config.
        """
        self.model_path = model_path if model_path else PATH_CONFIG["model_save_path"]
        self.processor = TaxiDataProcessor()
        self.model = None

    def load_model(self):
        """
        Loads the XGBoost model from disk. Initializes the regressor with
        the same parameters used during training to ensure compatibility.
        """
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model file not found at {self.model_path}. Please train the model first."
                )

            # Initialize model structure with training parameters
            params = XGB_PARAMS.copy()
            params["n_estimators"] = TRAIN_CONFIG["num_boost_round"]

            self.model = xgb.XGBRegressor(**params)
            self.model.load_model(self.model_path)

        return self.model

    def predict(self, X):
        """
        Generates raw predictions for the given feature set.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Raw predicted fare amounts.
        """
        model = self.load_model()
        return model.predict(X)

    def post_process(self, predictions):
        """
        Applies post-processing rules to predictions.
        Enforces a minimum fare amount of $2.50.

        Args:
            predictions (np.ndarray): Raw predictions.

        Returns:
            np.ndarray: Post-processed predictions.
        """
        # Apply lower bound of $2.50
        return np.maximum(predictions, 2.50)

    def calculate_metrics(self, load_cached_data=True, sample_size=None):
        """
        Calculates and prints the RMSE on the validation set.

        Args:
            load_cached_data (bool): Whether to attempt loading data from cache.
            sample_size (int, optional): If provided, limits the validation set size
                                         for faster debugging.

        Returns:
            float: The calculated RMSE.
        """
        # Load validation data
        val_df = self.processor.get_processed_data(
            "val", load_cached_data=load_cached_data
        )

        # Apply sampling if requested
        if sample_size is not None:
            val_df = val_df.iloc[:sample_size]

        # Separate features and target
        # 'key' is an identifier, 'fare_amount' is the target
        target_col = "fare_amount"
        drop_cols = ["key", target_col]

        if target_col not in val_df.columns:
            raise ValueError(
                f"Target column '{target_col}' missing from validation data."
            )

        y_true = val_df[target_col].values
        X_val = val_df.drop(columns=drop_cols)

        # Generate predictions
        raw_preds = self.predict(X_val)
        final_preds = self.post_process(raw_preds)

        # Calculate RMSE
        mse = np.mean((y_true - final_preds) ** 2)
        rmse = np.sqrt(mse)

        # Print full precision as requested
        print(f"Validation RMSE: {rmse}")

        return rmse

    def generate_submission(self, load_cached_data=True):
        """
        Generates predictions for the test set and saves the submission CSV.

        Args:
            load_cached_data (bool): Whether to attempt loading data from cache.
        """
        # Load test data
        test_df = self.processor.get_processed_data(
            "test", load_cached_data=load_cached_data
        )

        # Prepare features (drop 'key')
        X_test = test_df.drop(columns=["key"])

        # Generate predictions
        raw_preds = self.predict(X_test)
        final_preds = self.post_process(raw_preds)

        # Construct submission DataFrame
        submission = pd.DataFrame({"key": test_df["key"], "fare_amount": final_preds})

        # Save to file
        output_path = PATH_CONFIG["submission_output"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
