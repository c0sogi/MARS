import pandas as pd
import numpy as np
import torch
import os
import shutil
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

# Import provided library modules
import library.config
import library.utils
import library.data_loader
import library.models_classical
import library.models_neural
import library.stacking

# ==========================================
# Configuration Patching for Fast Baseline
# ==========================================
# Patch global config and module-level constants to ensure speed
# Reducing Epochs to 2 and Folds to 3 to complete within 2 hours
library.config.EPOCHS = 2
library.config.N_FOLDS = 3

# Patching specific modules that import these constants
library.models_neural.EPOCHS = 2
library.models_neural.N_FOLDS = 3
library.models_classical.N_FOLDS = 3

# Ensure correct device usage
library.models_neural.DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Runtime Device: {library.models_neural.DEVICE}")


def main():
    # Clear stale cache to prevent shape mismatch errors (e.g. 120 vs 14096 samples)
    if os.path.exists(library.config.WORKING_DIR):
        shutil.rmtree(library.config.WORKING_DIR)
    os.makedirs(library.config.WORKING_DIR, exist_ok=True)

    # 1. Load Data
    print("Loading datasets...")
    train_df, val_df, test_df = library.data_loader.load_data()

    # Calculate the split index to separate Train and Val in OOF predictions
    # The base model CV concatenates Train and Val: y_full = concat(y_train, y_val)
    split_idx = len(train_df)

    # 2. Run Base Models
    # Classical Models (LR, NB, XGB)
    print("Running Classical Models CV...")
    classical_preds = library.models_classical.run_classical_cv(load_cached_preds=True)

    # Neural Models (DeBERTa, RoBERTa)
    print("Running Neural Models CV...")
    neural_preds = library.models_neural.run_neural_cv(load_cached_preds=True)

    # 3. Prepare Stacking Data
    # Collect OOF and Test predictions from all models
    oof_preds_dict = {}
    test_preds_dict = {}

    # Merge dictionaries
    for d in [classical_preds, neural_preds]:
        for k, v in d.items():
            if k.endswith("_oof"):
                oof_preds_dict[k] = v
            elif k.endswith("_test"):
                test_preds_dict[k] = v

    # 4. Validation Analysis (Hold-out Strategy)
    # Reconstruct meta-features to calculate metric on the specific hold-out validation set

    # Encode labels consistent with library logic
    le = LabelEncoder()
    le.fit(train_df["author"])
    y_train = le.transform(train_df["author"])
    y_val = le.transform(val_df["author"])

    # Sort keys to ensure deterministic feature order
    model_keys = sorted(oof_preds_dict.keys())
    print(f"Aggregating meta-features from models: {model_keys}")

    # Stack OOF predictions horizontally
    X_meta_full = np.hstack([oof_preds_dict[k] for k in model_keys])

    # Split Meta-Features into Train (for meta-learner training) and Val (for evaluation)
    X_meta_train = X_meta_full[:split_idx]
    X_meta_val = X_meta_full[split_idx:]

    print(
        f"Meta-Train Shape: {X_meta_train.shape} | Meta-Val Shape: {X_meta_val.shape}"
    )

    # Train Proxy Meta-Learner for Analysis
    print("Training proxy meta-learner for validation assessment...")
    meta_model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        random_state=library.config.SEED,
        max_iter=1000,
    )
    meta_model.fit(X_meta_train, y_train)

    # Predict on Hold-out Validation Set
    val_probs = meta_model.predict_proba(X_meta_val)

    # Calculate Final Validation Metric
    # Using library utility to ensure consistent clipping
    val_log_loss = library.utils.calculate_log_loss(y_val, val_probs)
    print(f"Final Validation Metric: {val_log_loss}")

    # 5. Failure Analysis
    print("Performing failure analysis...")
    # Calculate per-sample log loss
    val_probs_clipped = library.utils.clip_probabilities(val_probs)
    # Extract probability assigned to the true class
    true_class_probs = val_probs_clipped[np.arange(len(y_val)), y_val]
    sample_losses = -np.log(true_class_probs)

    # Generate features for correlation analysis
    val_texts = val_df["text"].astype(str)
    char_counts = val_texts.apply(len).values
    word_counts = val_texts.apply(lambda x: len(x.split())).values

    # Calculate correlations
    corr_char = np.corrcoef(sample_losses, char_counts)[0, 1]
    corr_word = np.corrcoef(sample_losses, word_counts)[0, 1]

    print(f"Correlation (Loss vs Char Count): {corr_char}")
    print(f"Correlation (Loss vs Word Count): {corr_word}")

    # 6. Submission Generation
    THRESHOLD = 0.23237805822413304

    if val_log_loss < THRESHOLD:
        print(
            f"Validation metric {val_log_loss} meets threshold ({THRESHOLD}). Generating submission..."
        )
        # Call the library function to train the final meta-learner on all data and save submission
        library.stacking.train_meta_learner(oof_preds_dict, test_preds_dict)
    else:
        print(
            f"Validation metric {val_log_loss} did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
