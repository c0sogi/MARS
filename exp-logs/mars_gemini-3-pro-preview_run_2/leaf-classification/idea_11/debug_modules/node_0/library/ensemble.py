import os
import numpy as np
import pandas as pd
from library.data_loader import load_and_preprocess
from library.models import get_linear_branch, get_generative_branch, get_kernel_branch


def soft_vote(probabilities_list):
    """
    Aggregates predictions from multiple models using soft voting (averaging).

    Args:
        probabilities_list (list of np.ndarray): List of probability matrices.
                                                 Each matrix has shape (n_samples, n_classes).

    Returns:
        np.ndarray: Averaged probability matrix.
    """
    if not probabilities_list:
        raise ValueError("probabilities_list cannot be empty")

    # Stack along a new axis and compute mean
    stacked_probs = np.array(probabilities_list)
    avg_probs = np.mean(stacked_probs, axis=0)

    return avg_probs


def run_ensemble(load_cached_data=True, random_state=42):
    """
    Orchestrates the Kernel-Augmented Linear Hybrid (KALH) Ensemble.

    1. Loads data (Train+Val combined).
    2. Trains Linear, Generative, and Kernel branches.
    3. Generates predictions on Test set.
    4. Aggregates using Soft Voting.
    5. Saves submission file.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: The final submission dataframe.
    """
    # Set random seeds for reproducibility
    np.random.seed(random_state)

    print("Initializing KALH Ensemble...")

    # 1. Load Data
    # X_train is already scaled and is the combination of original Train + Val
    # This maximizes sample utilization for the final model
    X_train, y_train, X_test, test_ids, le = load_and_preprocess(
        load_cached_data=load_cached_data
    )

    print(f"Data Loaded. Training Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    print(f"Number of classes: {len(le.classes_)}")

    # 2. Initialize Models
    # Branch 1: Discriminative Linear (Logistic Regression)
    linear_model = get_linear_branch(random_state=random_state)

    # Branch 2: Generative Linear (LDA)
    generative_model = get_generative_branch()

    # Branch 3: Discriminative Kernel (Nystroem + LR)
    kernel_model = get_kernel_branch(random_state=random_state)

    # 3. Train Models
    # Note: We train on the combined Train+Val set, so we report Training Accuracy as the metric.

    print("\n--- Training Linear Branch (Logistic Regression) ---")
    linear_model.fit(X_train, y_train)
    train_acc_linear = linear_model.score(X_train, y_train)
    print(f"Linear Branch training complete. Training Accuracy: {train_acc_linear:.6f}")

    print("\n--- Training Generative Branch (LDA) ---")
    generative_model.fit(X_train, y_train)
    train_acc_gen = generative_model.score(X_train, y_train)
    print(
        f"Generative Branch training complete. Training Accuracy: {train_acc_gen:.6f}"
    )

    print("\n--- Training Kernel Branch (Nystroem + LR) ---")
    kernel_model.fit(X_train, y_train)
    train_acc_kernel = kernel_model.score(X_train, y_train)
    print(f"Kernel Branch training complete. Training Accuracy: {train_acc_kernel:.6f}")

    # 4. Inference
    print("\n--- Running Inference on Test Set ---")

    probs_linear = linear_model.predict_proba(X_test)
    probs_gen = generative_model.predict_proba(X_test)
    probs_kernel = kernel_model.predict_proba(X_test)

    print("Predictions generated for all branches.")

    # 5. Soft Voting
    print("\n--- Aggregating Predictions (Soft Voting) ---")
    final_probs = soft_vote([probs_linear, probs_gen, probs_kernel])

    # 6. Submission Generation
    print("\n--- Generating Submission File ---")
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Create DataFrame
    # Columns: id, species_1, species_2, ...
    # Ensure columns match the class order from LabelEncoder
    df_sub = pd.DataFrame(final_probs, columns=le.classes_)

    # Insert 'id' as the first column
    df_sub.insert(0, "id", test_ids)

    # Save
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    return df_sub
