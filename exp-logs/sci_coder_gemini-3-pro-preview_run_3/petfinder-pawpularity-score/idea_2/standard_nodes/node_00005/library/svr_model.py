import os
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from library import config, utils


class PetPawpularityRegressor:
    """
    A wrapper class for the SVR regression pipeline.
    Handles hyperparameter tuning, training, and inference.
    """

    def __init__(self):
        """
        Initialize the regressor with model paths from config.
        """
        self.model_path = config.SVR_MODEL_PATH
        self.model = None

        # Ensure the directory for the model exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def fit(self, X, y):
        """
        Fits the SVR model using GridSearchCV to find the best hyperparameters.

        Args:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training target values.
        """
        # Ensure reproducibility
        utils.seed_everything()

        print("Initializing SVR Pipeline and Grid Search...")

        # Define the pipeline: Standardization -> SVR
        pipeline = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])

        # Prepare parameter grid
        # config.SVR_GRID keys (e.g., 'C') need to be prefixed with 'svr__' for the pipeline
        param_grid = {f"svr__{k}": v for k, v in config.SVR_GRID.items()}

        # Define Cross-Validation strategy
        cv = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)

        # Initialize GridSearchCV
        # We use 'neg_root_mean_squared_error' to optimize for RMSE directly
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1,
            refit=True,
        )

        print(f"Starting training on {len(X)} samples with {config.N_FOLDS}-fold CV...")
        grid_search.fit(X, y)

        # Retrieve best results
        # best_score_ is negative RMSE, so we negate it to get positive RMSE
        best_rmse = -grid_search.best_score_
        best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_

        print("Training Complete.")
        print(f"Best Validation RMSE: {best_rmse}")
        print(f"Best Parameters: {best_params}")

        # Save the trained model
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def predict(self, X):
        """
        Generates predictions for the given features.
        Loads the model from disk if it's not currently in memory.

        Args:
            X (np.ndarray): Feature matrix to predict on.

        Returns:
            np.ndarray: Predicted Pawpularity scores, clipped to [1, 100].
        """
        # Load model if not present
        if self.model is None:
            if os.path.exists(self.model_path):
                print(f"Loading model from {self.model_path}...")
                self.model = joblib.load(self.model_path)
            else:
                raise FileNotFoundError(
                    f"Model not found at {self.model_path}. Please call fit() first."
                )

        # Predict
        predictions = self.model.predict(X)

        # Clip predictions to the valid range [1, 100] as per dataset description
        predictions = np.clip(predictions, 1.0, 100.0)

        return predictions


def generate_submission(ids, predictions, output_path=config.SUBMISSION_PATH):
    """
    Creates a submission CSV file in the required format.

    Args:
        ids (np.ndarray): Array of Pet Profile IDs.
        predictions (np.ndarray): Array of predicted Pawpularity scores.
        output_path (str): Path to save the CSV file.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame({"Id": ids, "Pawpularity": predictions})

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
