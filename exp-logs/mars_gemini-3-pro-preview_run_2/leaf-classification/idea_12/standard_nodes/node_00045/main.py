import numpy as np
import pandas as pd
import os
import sys
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import provided library functions
from library.config import RANDOM_SEED, PROB_CLIP_EPS
from library.data_manager import load_dataset
from library.models import get_linear_branch, get_generative_branch
from library.engine import train_and_predict_ensemble


def set_seed(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(RANDOM_SEED)

    # 2. Load Validation Data
    print("Loading datasets for validation...")
    X_train, y_train, ids_train = load_dataset("train", load_cached_data=True)
    X_val, y_val, ids_val = load_dataset("val", load_cached_data=True)

    # 3. Preprocessing
    print("Preprocessing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 4. Train Models on Training Split
    print("Training ensemble branches on training split...")

    # Branch 1: Discriminative Linear
    clf_linear = get_linear_branch()
    clf_linear.fit(X_train_scaled, y_train)

    # Branch 2: Generative Linear
    clf_gen = get_generative_branch()
    clf_gen.fit(X_train_scaled, y_train)

    # 5. Inference on Validation Split
    print("Running inference on validation split...")

    # Verify class alignment across models
    classes = clf_linear.classes_
    if not np.array_equal(clf_gen.classes_, classes):
        raise RuntimeError("Class mismatch between ensemble branches.")

    # Get probabilities
    probs_linear = clf_linear.predict_proba(X_val_scaled)
    probs_gen = clf_gen.predict_proba(X_val_scaled)

    # Soft Voting Ensemble (LR + LDA)
    final_probs = (probs_linear + probs_gen) / 2.0

    # Clip probabilities for numerical stability
    final_probs = np.clip(final_probs, PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS)

    # 6. Calculate Metric
    val_loss = log_loss(y_val, final_probs, labels=classes)

    print(f"Final Validation Metric: {val_loss}")

    # 7. Failure Analysis
    print("Performing failure analysis...")

    le = LabelEncoder()
    le.fit(classes)
    y_val_indices = le.transform(y_val)

    true_class_probs = final_probs[np.arange(len(y_val)), y_val_indices]
    sample_losses = -np.log(true_class_probs)

    correlations = X_val.corrwith(pd.Series(sample_losses, index=X_val.index))
    top_correlations = correlations.abs().sort_values(ascending=False).head(5)

    print("Top 5 features correlated with error magnitude (Log Loss):")
    print(top_correlations)

    # 8. Submission Generation
    # Condition: Metric must be lower than 0.00870833951594525
    THRESHOLD = 0.00870833951594525

    if val_loss < THRESHOLD:
        print(
            f"Validation metric ({val_loss}) meets threshold ({THRESHOLD}). Proceeding to full training and submission..."
        )
        # This function retrains on Train + Val and generates submission.csv
        train_and_predict_ensemble(load_cached_data=True)
    else:
        print(
            f"Validation metric ({val_loss}) did not meet threshold ({THRESHOLD}). Submission generation skipped."
        )


if __name__ == "__main__":
    main()
