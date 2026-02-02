import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from library.config import WORKING_DIR, SUBMISSION_DIR, SEED
from library.utils import calculate_log_loss, save_submission, clip_probabilities
from library.data_loader import load_data


def train_meta_learner(oof_preds, test_preds):
    """
    Trains a Level-2 Logistic Regression meta-learner on the predictions from
    base models and generates the final submission.

    Args:
        oof_preds (dict): Dictionary where keys are model identifiers (e.g., 'lr_oof')
                          and values are OOF probability arrays (N_samples, 3).
        test_preds (dict): Dictionary where keys are model identifiers (e.g., 'lr_test')
                           and values are Test probability arrays (N_test, 3).
    """
    print("Initializing Meta-Learner (Stacking)...")

    # 1. Load Data to reconstruct targets and get IDs
    train_df, val_df, test_df = load_data()

    # 2. Prepare Labels (y_full)
    # Must match the order used in base models: Train then Val
    le = LabelEncoder()
    le.fit(train_df["author"])
    class_names = list(le.classes_)  # ['EAP', 'HPL', 'MWS']

    y_train = le.transform(train_df["author"])
    y_val = le.transform(val_df["author"])
    y_full = np.concatenate([y_train, y_val])

    print(f"Constructing meta-features from {len(oof_preds)} base models...")

    # 3. Construct Meta-Features
    # Identify model keys based on '_oof' suffix and sort them for deterministic order
    model_keys = sorted([k for k in oof_preds.keys() if k.endswith("_oof")])

    X_meta_list = []
    X_test_meta_list = []

    for oof_key in model_keys:
        # Derive test key: 'lr_oof' -> 'lr_test'
        base_name = oof_key.replace("_oof", "")
        test_key = base_name + "_test"

        if test_key not in test_preds:
            raise KeyError(
                f"Matching test predictions for {base_name} not found in test_preds."
            )

        print(f"  - Adding model: {base_name}")
        X_meta_list.append(oof_preds[oof_key])
        X_test_meta_list.append(test_preds[test_key])

    # Concatenate horizontally: (N_samples, N_models * 3)
    X_meta = np.hstack(X_meta_list)
    X_test_meta = np.hstack(X_test_meta_list)

    print(f"Meta-Feature Matrix Shape: {X_meta.shape}")

    # 4. Train Meta-Learner
    print("Training Logistic Regression Meta-Learner...")
    # No penalty or low penalty is standard for stacking to allow the meta-learner
    # to trust the base models fully. Default C=1.0 is usually sufficient.
    meta_model = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", random_state=SEED, max_iter=1000
    )

    meta_model.fit(X_meta, y_full)

    # 5. Validate on OOF Data (Self-Check)
    # Note: Strictly speaking, this is training error for the meta-learner,
    # but since the input features are OOF predictions, it represents the
    # ensemble's performance.
    meta_oof_probs = meta_model.predict_proba(X_meta)
    meta_log_loss = calculate_log_loss(y_full, meta_oof_probs)
    print(f"Meta-Learner OOF Log Loss: {meta_log_loss}")

    # 6. Generate Test Predictions
    print("Generating final test predictions...")
    final_test_probs = meta_model.predict_proba(X_test_meta)

    # 7. Save Submission
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    print(f"Saving submission to {submission_path}...")

    save_submission(
        ids=test_df["id"].values,
        probs=final_test_probs,
        class_names=class_names,
        output_path=submission_path,
    )

    print("Stacking complete.")
