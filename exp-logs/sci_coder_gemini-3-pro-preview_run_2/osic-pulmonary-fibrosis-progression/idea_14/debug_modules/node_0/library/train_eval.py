import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood
from library.feature_engineering import FeatureProcessor
from library.models import QuantileLinearModel, ResidualElasticModel


class Trainer:
    def __init__(self):
        """
        Initializes the Trainer with models and configuration.
        """
        self.fvc_model = QuantileLinearModel(quantile=Config.QUANTILE)
        self.unc_model = ResidualElasticModel(
            alpha=Config.ELASTIC_NET_ALPHA, l1_ratio=Config.ELASTIC_NET_L1_RATIO
        )
        self.feature_processor = FeatureProcessor()

    def load_data(self, load_cached_data=True):
        """
        Loads preprocessed data using the FeatureProcessor.
        """
        print("Loading and processing data...")
        train_data, val_data, test_data = self.feature_processor.process_pipelines(
            load_cached_data=load_cached_data
        )
        return train_data, val_data, test_data

    def train(self, train_data):
        """
        Executes the Decoupled Residual Regression training pipeline.

        Stage 1: Train FVC Model (Quantile Regression)
        Stage 2: Compute Residuals
        Stage 3: Train Uncertainty Model (Elastic Net on Residuals)
        """
        X_fvc = train_data["X_fvc"]
        X_unc = train_data["X_unc"]
        y = train_data["y"]

        print("Stage 1: Training FVC Model (Quantile Regression)...")
        self.fvc_model.fit(X_fvc, y)

        # Generate predictions on training set to compute residuals
        y_pred_train = self.fvc_model.predict(X_fvc)

        # Calculate Absolute Residuals (MAD target)
        residuals = np.abs(y - y_pred_train)

        print("Stage 2: Training Uncertainty Model (Elastic Net on Residuals)...")
        self.unc_model.fit(X_unc, residuals)

    def evaluate(self, val_data):
        """
        Evaluates the models on the validation set using the Laplace Log Likelihood metric.
        """
        X_fvc = val_data["X_fvc"]
        X_unc = val_data["X_unc"]
        y_true = val_data["y"]

        # Predict Median FVC
        y_pred = self.fvc_model.predict(X_fvc)

        # Predict Uncertainty (MAD)
        mad_pred = self.unc_model.predict(X_unc)

        # Convert MAD to Sigma for Laplace Metric
        # Analytical scaling: sigma = MAD * sqrt(2)
        sigma_pred = mad_pred * np.sqrt(2)

        # Compute Metric
        score = laplace_log_likelihood(y_true, y_pred, sigma_pred)

        print(f"Validation Laplace Log Likelihood: {score}")
        return score

    def generate_submission(self, test_data):
        """
        Generates predictions for the test set and saves the submission file.
        """
        X_fvc = test_data["X_fvc"]
        X_unc = test_data["X_unc"]
        patient_week_ids = test_data["patient_week"]

        print("Generating submission predictions...")

        # Predict FVC
        fvc_pred = self.fvc_model.predict(X_fvc)

        # Predict Uncertainty
        mad_pred = self.unc_model.predict(X_unc)
        sigma_pred = mad_pred * np.sqrt(2)

        # Create DataFrame
        submission_df = pd.DataFrame(
            {
                "Patient_Week": patient_week_ids,
                "FVC": fvc_pred,
                "Confidence": sigma_pred,
            }
        )

        # Save to file
        save_path = Config.SUBMISSION_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    def save_models(self):
        """
        Saves the trained models to the cache directory.
        """
        print("Saving models...")
        self.fvc_model.save("fvc_model.joblib")
        self.unc_model.save("unc_model.joblib")


def run_training_pipeline(load_cached_data=True):
    """
    Main function to run the complete training and evaluation pipeline.
    """
    # 1. Set Seed
    seed_everything(Config.SEED)

    # 2. Initialize Trainer
    trainer = Trainer()

    # 3. Load Data
    train_data, val_data, test_data = trainer.load_data(
        load_cached_data=load_cached_data
    )

    # 4. Train Models
    trainer.train(train_data)

    # 5. Evaluate
    trainer.evaluate(val_data)

    # 6. Generate Submission
    trainer.generate_submission(test_data)

    # 7. Save Models
    trainer.save_models()

    print("Pipeline completed successfully.")
