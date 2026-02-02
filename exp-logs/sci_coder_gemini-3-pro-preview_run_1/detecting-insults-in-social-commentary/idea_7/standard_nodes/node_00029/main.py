import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.data_processing import load_data, get_tokenizer
from library.training import run_fold
from library.inference import predict_fn, generate_pseudo_labels


def main():
    # 1. Initialization
    seed_everything(Config.SEED)
    device = get_device()
    tokenizer = get_tokenizer()

    print(f"Device: {device}")

    # 2. Load Data
    # load_data handles caching of SVD features
    train_df, val_df, test_df, train_svd, val_svd, test_svd = load_data(
        load_cached_data=True, debug=Config().debug
    )

    # Combine train and validation sets for Stratified K-Fold
    # We want to utilize all available labeled data for cross-validation
    full_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    full_svd = np.vstack([train_svd, val_svd])
    y_full = full_df["Insult"].values

    # Initialize K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Storage for predictions
    teacher_test_preds = np.zeros((len(test_df), Config.N_FOLDS))
    student_test_preds = np.zeros((len(test_df), Config.N_FOLDS))

    # Storage for OOF (Out-Of-Fold) validation predictions (for Stage 2)
    oof_preds = np.zeros(len(full_df))
    oof_targets = np.zeros(len(full_df))

    # ==========================================
    # STAGE 1: Teacher Training
    # ==========================================
    print("\n" + "=" * 40)
    print(" STAGE 1: Teacher Training ")
    print("=" * 40)

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, y_full)):
        # Prepare Fold Data
        f_train_df = full_df.iloc[train_idx].reset_index(drop=True)
        f_val_df = full_df.iloc[val_idx].reset_index(drop=True)
        f_train_svd = full_svd[train_idx]
        f_val_svd = full_svd[val_idx]

        # Train Teacher Model
        model, _ = run_fold(
            fold,
            f_train_df,
            f_val_df,
            f_train_svd,
            f_val_svd,
            tokenizer,
            device,
            stage_name="Teacher",
        )

        # Inference on Test Set
        fold_preds = predict_fn(model, test_df, test_svd, device, tokenizer=tokenizer)
        teacher_test_preds[:, fold] = fold_preds.flatten()

        # Cleanup to save memory
        del model
        torch.cuda.empty_cache()

    # Average Teacher Predictions
    avg_teacher_preds = teacher_test_preds.mean(axis=1)

    # ==========================================
    # Pseudo-Labeling
    # ==========================================
    print("\n" + "=" * 40)
    print(" Generating Pseudo-Labels ")
    print("=" * 40)

    pseudo_df, pseudo_svd = generate_pseudo_labels(
        test_df,
        test_svd,
        avg_teacher_preds,
        high_thresh=Config.PSEUDO_LABEL_HIGH,
        low_thresh=Config.PSEUDO_LABEL_LOW,
    )

    # ==========================================
    # STAGE 2: Student Training
    # ==========================================
    print("\n" + "=" * 40)
    print(" STAGE 2: Student Training ")
    print("=" * 40)

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, y_full)):
        # Base Split (Same as Stage 1 to maintain validation integrity)
        f_train_df = full_df.iloc[train_idx]
        f_val_df = full_df.iloc[val_idx].reset_index(drop=True)
        f_train_svd = full_svd[train_idx]
        f_val_svd = full_svd[val_idx]

        # Augment Training Data with Pseudo-Labels
        if len(pseudo_df) > 0:
            aug_train_df = pd.concat([f_train_df, pseudo_df]).reset_index(drop=True)
            aug_train_svd = np.vstack([f_train_svd, pseudo_svd])
        else:
            aug_train_df = f_train_df.reset_index(drop=True)
            aug_train_svd = f_train_svd

        # Train Student Model
        model, _ = run_fold(
            fold,
            aug_train_df,
            f_val_df,
            aug_train_svd,
            f_val_svd,
            tokenizer,
            device,
            stage_name="Student",
        )

        # Inference on Test Set
        fold_test_preds = predict_fn(
            model, test_df, test_svd, device, tokenizer=tokenizer
        )
        student_test_preds[:, fold] = fold_test_preds.flatten()

        # Inference on Validation Set (OOF)
        fold_val_preds = predict_fn(
            model, f_val_df, f_val_svd, device, tokenizer=tokenizer
        )
        oof_preds[val_idx] = fold_val_preds.flatten()
        oof_targets[val_idx] = f_val_df["Insult"].values

        # Cleanup
        del model
        torch.cuda.empty_cache()

    # ==========================================
    # Evaluation
    # ==========================================
    print("\n" + "=" * 40)
    print(" Evaluation & Analysis ")
    print("=" * 40)

    final_auc = roc_auc_score(oof_targets, oof_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    print("\nFailure Analysis:")
    # Calculate absolute error
    errors = np.abs(oof_targets - oof_preds)

    # Feature 1: Comment Length
    lengths = full_df["Comment"].fillna("").apply(len).values
    corr_len = np.corrcoef(errors, lengths)[0, 1]

    # Feature 2: Caps Ratio
    def get_caps_ratio(text):
        s = str(text)
        if not s:
            return 0.0
        return sum(1 for c in s if c.isupper()) / len(s)

    caps_ratios = full_df["Comment"].fillna("").apply(get_caps_ratio).values
    corr_caps = np.corrcoef(errors, caps_ratios)[0, 1]

    print(f"Correlation between Error and Comment Length: {corr_len:.6f}")
    print(f"Correlation between Error and Caps Ratio: {corr_caps:.6f}")

    # ==========================================
    # Submission
    # ==========================================
    threshold = 0.9586453201970443

    if final_auc > threshold:
        print(
            f"\nValidation metric ({final_auc}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Average Student Predictions
        avg_student_preds = student_test_preds.mean(axis=1)

        # Prepare Submission DataFrame
        sub_path = Config.SUBMISSION_PATH
        sample_sub_path = "./input/sample_submission_null.csv"

        try:
            # Attempt to use sample submission structure
            if os.path.exists(sample_sub_path):
                sample_sub = pd.read_csv(sample_sub_path)
                if "Insult" in sample_sub.columns:
                    sample_sub["Insult"] = avg_student_preds
                    sample_sub.to_csv(sub_path, index=False)
                else:
                    # Fallback if column name differs
                    pd.DataFrame(
                        {"id": range(len(test_df)), "prediction": avg_student_preds}
                    ).to_csv(sub_path, index=False)
            else:
                # Fallback if file missing
                pd.DataFrame(
                    {"id": range(len(test_df)), "prediction": avg_student_preds}
                ).to_csv(sub_path, index=False)

            print(f"Submission saved to {sub_path}")

        except Exception as e:
            print(f"Error generating submission: {e}")
            # Emergency fallback
            pd.DataFrame({"Insult": avg_student_preds}).to_csv(sub_path, index=False)
    else:
        print(
            f"\nValidation metric ({final_auc}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
