import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from library import config, data_loader, preprocessing, model, metrics


def main():
    # Set fixed seed for reproducibility
    np.random.seed(config.SEED)

    # 1. Load Data
    # Load datasets using the provided data_loader with caching enabled
    (
        (X_train_raw, y_train, train_ids),
        (X_val_raw, y_val, val_ids),
        (X_test_raw, test_ids),
    ) = data_loader.load_datasets(load_cached_data=True)

    # 2. Preprocess Data
    # Apply the RobustPipeline (Yeo-Johnson + StandardScaler)
    # The pipeline is fitted on X_train and transforms all sets
    X_train, X_val, X_test = preprocessing.preprocess_datasets(
        X_train_raw, X_val_raw, X_test_raw, load_cached_data=True
    )

    # 3. Train Model
    # Initialize and fit the OAS Linear Discriminant
    clf = model.OASLinearDiscriminant()
    clf.fit(X_train, y_train)

    # 4. Validation & Metrics
    # Generate probability predictions for the validation set
    y_val_pred = clf.predict_proba(X_val)

    # Calculate the Multi-class Log Loss
    # We pass clf.classes_ to ensure correct label alignment
    val_loss = metrics.calculate_log_loss(y_val, y_val_pred, labels=clf.classes_)

    # Print the validation metric in the required format
    print(f"Final Validation Metric: {val_loss}")

    # 5. Failure Analysis
    # Analyze the correlation between feature values and prediction error

    # Map string labels to integer indices based on the model's classes
    le = LabelEncoder()
    le.classes_ = clf.classes_
    y_val_indices = le.transform(y_val)

    # Normalize and clip predictions to strictly match the metric calculation logic
    row_sums = y_val_pred.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    y_val_pred_norm = y_val_pred / row_sums
    y_val_pred_clipped = np.clip(
        y_val_pred_norm, config.PROB_CLIP, 1 - config.PROB_CLIP
    )

    # Extract the probability assigned to the true class
    prob_true = y_val_pred_clipped[np.arange(len(y_val)), y_val_indices]

    # Calculate per-sample Log Loss (Error Magnitude)
    sample_losses = -np.log(prob_true)

    # Create a DataFrame combining features and error magnitude
    # We use the feature names from config to label the columns
    analysis_df = pd.DataFrame(X_val, columns=config.FEATURES)
    analysis_df["error_magnitude"] = sample_losses

    # Calculate correlation between features and error magnitude
    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")
    top_correlations = correlations.abs().sort_values(ascending=False).head(5)

    print("Top 5 features correlated with error magnitude:")
    print(top_correlations)

    # 6. Submission Generation
    # strictly adhere to the threshold requirement
    THRESHOLD = 1.2136771218566717e-09

    if val_loss < THRESHOLD:
        # Generate predictions for the test set
        y_test_pred = clf.predict_proba(X_test)

        # Create submission DataFrame
        submission = pd.DataFrame(y_test_pred, columns=clf.classes_)
        submission.insert(0, "id", test_ids)

        # Save the submission file
        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {val_loss} is not lower than threshold {THRESHOLD}. Submission not generated."
        )


if __name__ == "__main__":
    main()
