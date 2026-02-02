import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

from library.config import Config
from library.utils import seed_everything, get_score, load_checkpoint
from library.data_processing import (
    load_data,
    get_structural_features,
    InsultDataset,
    prepare_student_data,
)
from library.model import HybridDeberta
from library.trainer import run_fold


def get_logits(model, loader, device):
    """
    Custom inference function to retrieve raw logits (before sigmoid).
    """
    model.eval()
    logits_list = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            svd_features = batch["svd_features"].to(device)

            # Forward pass returns logits
            y_logits = model(input_ids, attention_mask, svd_features)
            logits_list.append(y_logits.cpu().numpy())

    return np.concatenate(logits_list)


def predict_ensemble(model_paths, df, svd_features, device, return_logits=False):
    """
    Performs ensemble inference.
    """
    # Create Dataset and Loader
    dataset = InsultDataset(
        texts=df["Comment"].values, svd_features=svd_features, labels=None
    )
    loader = DataLoader(
        dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    ensemble_preds = []

    for path in model_paths:
        model = HybridDeberta(pretrained=False)
        model = load_checkpoint(model, path, device)
        model.to(device)

        logits = get_logits(model, loader, device)

        if return_logits:
            ensemble_preds.append(logits)
        else:
            # Convert logits to probabilities
            probs = 1.0 / (1.0 + np.exp(-logits))
            ensemble_preds.append(probs)

    # Average predictions
    avg_preds = np.mean(ensemble_preds, axis=0)
    return avg_preds


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Ensure output directories exist
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    teacher_dir = os.path.join(Config.OUTPUT_DIR, "teacher")
    student_dir = os.path.join(Config.OUTPUT_DIR, "student")
    os.makedirs(teacher_dir, exist_ok=True)
    os.makedirs(student_dir, exist_ok=True)

    # 2. Data Loading
    print("Loading Data...")
    train_df, val_df, test_df = load_data()

    # Get Structural Features
    print("Generating Structural Features...")
    train_svd, val_svd, test_svd = get_structural_features(
        train_df["Comment"].tolist(),
        val_df["Comment"].tolist(),
        test_df["Comment"].tolist(),
        load_cached_data=True,
    )

    # 3. Stage 1: Teacher Training
    print("\n=== Stage 1: Teacher Training ===")
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    teacher_paths = []

    # We iterate through folds on the Training Data
    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(train_df, train_df["Insult"])
    ):
        print(f"\nTeacher Fold {fold + 1}/{Config.N_FOLDS}")

        # Split Data
        X_train_text = train_df.iloc[train_idx]["Comment"].values
        X_valid_text = train_df.iloc[valid_idx]["Comment"].values
        y_train = train_df.iloc[train_idx]["Insult"].values
        y_valid = train_df.iloc[valid_idx]["Insult"].values

        SVD_train = train_svd[train_idx]
        SVD_valid = train_svd[valid_idx]

        # Create Datasets
        train_ds = InsultDataset(X_train_text, SVD_train, y_train)
        valid_ds = InsultDataset(X_valid_text, SVD_valid, y_valid)

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Train
        save_path = os.path.join(teacher_dir, f"teacher_fold_{fold}.bin")
        run_fold(fold, train_loader, valid_loader, device, save_path)
        teacher_paths.append(save_path)

    # 4. Stage 2: Soft Label Generation
    print("\n=== Stage 2: Generating Soft Labels ===")
    # Get averaged logits from Teacher Ensemble
    # Note: predict_ensemble returns averaged logits if return_logits=True
    avg_logits = predict_ensemble(
        teacher_paths, test_df, test_svd, device, return_logits=True
    )

    # Apply Temperature Scaling
    temperature = Config.TEMPERATURE
    # Sigmoid(Logits / T)
    soft_labels = 1.0 / (1.0 + np.exp(-avg_logits / temperature))

    # Flatten if necessary (output shape is (N, 1))
    soft_labels = soft_labels.flatten()

    print(
        f"Soft Labels Generated. Mean: {soft_labels.mean():.4f}, Std: {soft_labels.std():.4f}"
    )

    # 5. Stage 3: Student Training
    print("\n=== Stage 3: Student Training ===")
    student_paths = []

    # We use the same splits to ensure validation consistency
    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(train_df, train_df["Insult"])
    ):
        print(f"\nStudent Fold {fold + 1}/{Config.N_FOLDS}")

        # Original Labeled Data for this fold
        df_train_fold = train_df.iloc[train_idx]
        svd_train_fold = train_svd[train_idx]

        # Validation Data (Original Labeled)
        X_valid_text = train_df.iloc[valid_idx]["Comment"].values
        y_valid = train_df.iloc[valid_idx]["Insult"].values
        SVD_valid = train_svd[valid_idx]

        # Prepare Combined Student Data (Labeled Train + Soft Labeled Test)
        combined_texts, combined_svd, combined_labels = prepare_student_data(
            df_train_fold, test_df, soft_labels, svd_train_fold, test_svd
        )

        # Create Datasets
        train_ds = InsultDataset(combined_texts, combined_svd, combined_labels)
        valid_ds = InsultDataset(X_valid_text, SVD_valid, y_valid)

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Train
        save_path = os.path.join(student_dir, f"student_fold_{fold}.bin")
        run_fold(fold, train_loader, valid_loader, device, save_path)
        student_paths.append(save_path)

    # 6. Evaluation on Hold-out Validation Set
    print("\n=== Final Evaluation ===")
    # Predict using Student Ensemble on the global validation set
    val_preds = predict_ensemble(
        student_paths, val_df, val_svd, device, return_logits=False
    )
    val_preds = val_preds.flatten()
    val_labels = val_df["Insult"].values

    final_auc = get_score(val_labels, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 7. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(val_labels - val_preds)

    # Feature Engineering for Analysis
    val_comments = val_df["Comment"].fillna("").astype(str)
    char_counts = val_comments.apply(len).values
    word_counts = val_comments.apply(lambda x: len(x.split())).values
    caps_ratios = val_comments.apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    ).values

    # Correlations
    corr_char, _ = pearsonr(errors, char_counts)
    corr_word, _ = pearsonr(errors, word_counts)
    corr_caps, _ = pearsonr(errors, caps_ratios)

    print("Correlation between Error Magnitude and Input Features:")
    print(f"  Char Count: {corr_char:.4f}")
    print(f"  Word Count: {corr_word:.4f}")
    print(f"  Caps Ratio: {corr_caps:.4f}")

    # 8. Submission
    threshold = 0.9603817733990148
    if final_auc > threshold:
        print(
            f"\nValidation metric ({final_auc}) > Threshold ({threshold}). Generating submission..."
        )

        # Predict on Test Set with Student Ensemble
        test_preds = predict_ensemble(
            student_paths, test_df, test_svd, device, return_logits=False
        )
        test_preds = test_preds.flatten()

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {
                "id": range(
                    len(test_preds)
                ),  # Placeholder ID if not in test.csv, but sample usually has ID or we just match rows
                "Insult": test_preds,
            }
        )

        # Check sample submission format
        # The sample_submission_null.csv has columns: Insult, Date, Comment.
        # Usually submission just needs the target or ID+Target.
        # However, looking at the provided sample_submission_null.csv in description, it has "Insult", "Date", "Comment".
        # We should probably just fill the "Insult" column of the test dataframe or create a new one matching the sample.
        # But standard Kaggle style is ID, Pred.
        # Given the instructions: "Your predictions should be a number in the range [0,1]. See 'sample_submissions_null.csv' for the correct format."
        # And "sample_submission_null.csv has 2647 rows and 3 columns."
        # We will reconstruct the submission based on test_df structure.

        # We will use the test_df (which has Date, Comment) and add the Insult column.
        submission_output = test_df.copy()
        submission_output["Insult"] = test_preds

        # Reorder to match sample: Insult, Date, Comment
        # Note: sample_submission_null.csv first row: | | Insult | Date | Comment |
        # It seems Insult is the first column.
        cols = ["Insult", "Date", "Comment"]
        submission_output = submission_output[cols]

        submission_output.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric ({final_auc}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
