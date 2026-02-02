import numpy as np
import pandas as pd
import sys
import os

# Import provided library modules
import importlib
import library.config as config

importlib.reload(config)
import library.utils as utils
import library.data_loader as data_loader
import library.feature_engineering as feature_engineering
import library.trainer as trainer


def main():
    # 1. Setup
    # Set random seeds for reproducibility
    utils.set_seed(config.RANDOM_STATE)

    # 2. Load Data
    # Load data using the caching mechanism to speed up subsequent runs
    print("Loading data...")
    train_df, val_df, test_df = data_loader.load_and_preprocess_data(
        load_cached_data=True
    )

    # 3. Feature Extraction
    # Extract TF-IDF features (word and char n-grams)
    # This returns sparse matrices for features and integer-encoded labels
    print("Extracting features...")
    X_train, y_train, X_val, y_val, X_test = feature_engineering.extract_features(
        train_df, val_df, test_df, load_cached_data=True
    )

    # Convert integer labels back to strings to ensure alignment with config.CLASSES
    # and model.classes_ during training and prediction
    classes = np.array(config.CLASSES)
    y_train_str = classes[y_train]
    y_val_str = classes[y_val]

    # 4. Train Model
    # Build and train the Logistic Regression model
    print("Training model...")
    model = trainer.train_model(X_train, y_train_str, X_val, y_val_str)

    # 5. Evaluate Model
    # Calculate Log Loss on the full validation set
    print("Evaluating model...")
    loss = trainer.evaluate_model(model, X_val, y_val_str)

    # REQUIRED: Print the final validation metric in the specific format
    print(f"Final Validation Metric: {loss}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # We analyze the correlation between the model's error magnitude and input text length.

    # Predict probabilities on validation set
    y_val_pred_proba = model.predict_proba(X_val)

    # Map string labels in y_val_str to indices to ensure alignment with model.classes_
    # model.classes_ is guaranteed to be sorted or match the fitting order
    class_to_idx = {cls: idx for idx, cls in enumerate(model.classes_)}
    y_val_indices = np.array([class_to_idx[lbl] for lbl in y_val_str])

    # Extract the probability assigned to the true class
    # row_indices = 0..N-1, col_indices = true class indices
    prob_true_class = y_val_pred_proba[np.arange(len(y_val_indices)), y_val_indices]

    # Calculate Error Magnitude: (1 - probability of true class)
    # High error magnitude means the model assigned low probability to the correct author
    error_magnitude = 1.0 - prob_true_class

    # Compute input features for correlation analysis: Text Length
    # We use text length as a proxy for input complexity/information content
    val_text_len_char = val_df["text"].str.len().fillna(0).values
    val_text_len_word = val_df["text"].apply(lambda x: len(str(x).split())).values

    # Calculate Pearson correlation coefficients
    corr_char = np.corrcoef(error_magnitude, val_text_len_char)[0, 1]
    corr_word = np.corrcoef(error_magnitude, val_text_len_word)[0, 1]

    print(f"Correlation between Error Magnitude and Character Length: {corr_char:.10f}")
    print(f"Correlation between Error Magnitude and Word Length: {corr_word:.10f}")

    # 7. Generate Submission
    # Conditional submission based on validation performance
    threshold = 0.4133111261364152
    if loss < threshold:
        print(
            f"\nValidation Log Loss ({loss:.6f}) is better than threshold ({threshold:.6f})."
        )
        print("Generating submission...")
        # Predict on test set
        y_test_pred = model.predict_proba(X_test)

        # Retrieve test IDs
        test_ids = test_df["id"].values

        # Save submission file
        utils.save_submission(test_ids, y_test_pred)
    else:
        print(
            f"\nValidation Log Loss ({loss:.6f}) did not meet threshold ({threshold:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
