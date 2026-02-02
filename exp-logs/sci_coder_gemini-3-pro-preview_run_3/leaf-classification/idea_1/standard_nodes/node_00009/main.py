import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Import provided library modules
from library.utils import seed_everything
from library.data_loader import LeafDataManager
from library.model_trainer import EnsembleTrainer


def run_pipeline():
    # Set random seed for reproducibility
    seed_everything(42)

    # ==========================================
    # 1. Data Loading
    # ==========================================
    print("Initializing Data Manager...")
    # Initialize data manager and process data (using cache if available)
    data_manager = LeafDataManager()
    data_manager.process_data(load_cached_data=True)

    # ==========================================
    # 2. Model Training
    # ==========================================
    print("Initializing Trainer...")
    trainer = EnsembleTrainer(data_manager=data_manager)

    # Perform grid search to find optimal regularization strength
    # This ensures the baseline is strong and handles multicollinearity
    print("Starting Grid Search for Regularization Parameter...")
    best_c = trainer.grid_search_regularization()

    # Train the final model using the best C found
    print(f"Training final model with C={best_c}...")
    trainer.train(c=best_c)

    # ==========================================
    # 3. Validation Assessment
    # ==========================================
    print("Performing Validation Inference...")
    X_val, y_val = data_manager.get_val_data()

    # Predict probabilities on validation set
    y_val_probs = trainer.predict_proba(X_val)

    # Calculate Multi-class Log Loss
    # y_val contains integer class labels, y_val_probs contains probabilities for each class
    val_loss = log_loss(y_val, y_val_probs)

    # REQUIRED: Print Final Validation Metric
    print(f"Final Validation Metric: {val_loss}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("Performing Failure Analysis...")

    # Calculate per-sample log loss (error magnitude)
    # We extract the predicted probability for the true class for each sample
    # y_val is an array of true class indices
    row_indices = np.arange(len(y_val))
    true_class_probs = y_val_probs[row_indices, y_val]

    # Clip probabilities to avoid log(0)
    true_class_probs = np.clip(true_class_probs, 1e-15, 1 - 1e-15)

    # Error magnitude is the negative log likelihood of the true class
    error_magnitude = -np.log(true_class_probs)

    # Retrieve feature names for meaningful analysis
    # We read the header of the train metadata file to get feature names
    # The data manager processes features starting with margin, shape, texture
    try:
        df_header = pd.read_csv(data_manager.train_path, nrows=0)
        feature_names = [
            c for c in df_header.columns if c.startswith(("margin", "shape", "texture"))
        ]
    except Exception as e:
        print(f"Could not retrieve feature names: {e}. Using indices.")
        feature_names = [f"feature_{i}" for i in range(X_val.shape[1])]

    # Create a DataFrame with validation features and error magnitude
    # X_val is a numpy array, so we wrap it in a DataFrame
    analysis_df = pd.DataFrame(X_val, columns=feature_names)
    analysis_df["error_magnitude"] = error_magnitude

    # Calculate correlation between features and error magnitude
    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")

    # Identify top 10 features most correlated with high error
    # We look at absolute correlation to find strong relationships (positive or negative)
    top_correlations = correlations.abs().sort_values(ascending=False).head(10)

    print("\nTop 10 Features Correlated with Error Magnitude (Failure Analysis):")
    # Retrieve the original signed correlation values for the top absolute ones
    for feature in top_correlations.index:
        corr_val = correlations[feature]
        print(f"{feature}: {corr_val:.4f}")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    print("\nGenerating Submission...")
    output_path = "./submission/submission.csv"
    trainer.generate_submission(output_path=output_path)
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
