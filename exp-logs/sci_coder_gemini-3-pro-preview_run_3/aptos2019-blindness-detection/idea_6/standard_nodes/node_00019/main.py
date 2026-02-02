import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import cv2

# Ensure library imports work
sys.path.append(os.getcwd())

from library.utils import seed_everything, quadratic_weighted_kappa
from library.data import get_datasets, RetinopathyDataset, get_transforms
from library.model import DRModel
from library.engine import train_model


def run_failure_analysis(val_meta_df, y_true, y_pred, input_dir):
    """
    Analyzes the correlation between prediction error and image meta-features.
    """
    print("\n--- Failure Analysis ---")

    # Calculate absolute errors
    errors = np.abs(y_true - y_pred)

    # Extract meta-features for the validation set
    widths = []
    heights = []
    intensities = []

    print("Extracting meta-features for failure analysis...")
    # We iterate through the dataframe to load original image properties
    for _, row in val_meta_df.iterrows():
        path = os.path.join(input_dir, row["file_path"])
        try:
            img = cv2.imread(path)
            if img is None:
                widths.append(0)
                heights.append(0)
                intensities.append(0)
                continue

            h, w, c = img.shape
            widths.append(w)
            heights.append(h)
            # Normalize and mean intensity
            intensities.append(img.mean() / 255.0)
        except Exception:
            widths.append(0)
            heights.append(0)
            intensities.append(0)

    # Calculate correlations using Pandas
    meta_features = {"Width": widths, "Height": heights, "Mean Intensity": intensities}

    print("Correlation between Absolute Error and Meta-features:")
    for name, feature_values in meta_features.items():
        if len(feature_values) != len(errors):
            print(f"Skipping {name}: Length mismatch.")
            continue

        # Create a temp dataframe to compute correlation
        temp_df = pd.DataFrame({"error": errors, "feature": feature_values})

        # Spearman correlation
        corr = temp_df["error"].corr(temp_df["feature"], method="spearman")
        print(f"  {name}: {corr:.4f}")


def main():
    # Configuration
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "processed_data")
    MODEL_DIR = os.path.join(WORKING_DIR, "models")
    SUBMISSION_DIR = "./submission"

    # Hyperparameters
    IMG_SIZE = 768
    BATCH_SIZE = 4
    ACCUMULATION_STEPS = 8
    EPOCHS = 2  # Limited epochs for fast baseline execution
    N_FOLDS = 5
    LR = 1e-4
    SUBMISSION_THRESHOLD = 0.9241120634346159

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    seed_everything(SEED)
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    print("Loading datasets...")
    # get_datasets handles caching. We load train and val to combine them for CV.
    train_ds_base, val_ds_base, test_ds = get_datasets(
        INPUT_DIR,
        METADATA_DIR,
        cache_dir=CACHE_DIR,
        img_size=IMG_SIZE,
        load_cached_data=True,
    )

    # Combine train and val for Stratified K-Fold
    all_images = np.concatenate([train_ds_base.images, val_ds_base.images], axis=0)
    all_labels = np.concatenate([train_ds_base.labels, val_ds_base.labels], axis=0)

    # Load metadata dataframes to track IDs for failure analysis
    # Order must match the concatenation above (Train then Val)
    df_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    df_all = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    # 2. K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_preds = np.zeros(len(all_labels))

    print(f"Starting {N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(all_images, all_labels.astype(int))
    ):
        print(f"\n=== Fold {fold+1}/{N_FOLDS} ===")

        # Prepare Fold Data
        train_imgs_fold = all_images[train_idx]
        train_lbls_fold = all_labels[train_idx]
        val_imgs_fold = all_images[val_idx]
        val_lbls_fold = all_labels[val_idx]

        train_dataset = RetinopathyDataset(
            train_imgs_fold,
            train_lbls_fold,
            transform=get_transforms("train", size=IMG_SIZE),
        )
        val_dataset = RetinopathyDataset(
            val_imgs_fold, val_lbls_fold, transform=get_transforms("val", size=IMG_SIZE)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Initialize Model
        model = DRModel(model_name="convnext_base", pretrained=True, drop_rate=0.0)
        model = model.to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=LR)

        # Train
        save_path = os.path.join(MODEL_DIR, f"model_fold_{fold}.pth")
        train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            DEVICE,
            epochs=EPOCHS,
            accumulation_steps=ACCUMULATION_STEPS,
            patience=2,
            save_path=save_path,
        )

        # Inference on Validation Fold (OOF)
        # Load best model to ensure we use the best checkpoint
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        model.eval()

        val_preds_fold = []
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(DEVICE)
                out = model(imgs)
                val_preds_fold.append(out.cpu().numpy())

        val_preds_fold = np.concatenate(val_preds_fold).flatten()
        oof_preds[val_idx] = val_preds_fold

        # Cleanup
        del model, optimizer, train_loader, val_loader, train_dataset, val_dataset
        torch.cuda.empty_cache()

    # 3. Global Validation Assessment
    print("\n=== Cross-Validation Complete ===")
    final_qwk = quadratic_weighted_kappa(all_labels, oof_preds)
    print(f"Final Validation Metric: {final_qwk}")

    # 4. Failure Analysis
    run_failure_analysis(df_all, all_labels, oof_preds, INPUT_DIR)

    # 5. Submission
    if final_qwk > SUBMISSION_THRESHOLD:
        print(
            f"\nMetric ({final_qwk}) > Threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        ensemble_preds = []

        # Ensemble Inference
        for fold in range(N_FOLDS):
            model_path = os.path.join(MODEL_DIR, f"model_fold_{fold}.pth")
            model = DRModel(model_name="convnext_base", pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model = model.to(DEVICE)
            model.eval()

            fold_preds = []
            with torch.no_grad():
                for imgs in test_loader:
                    imgs = imgs.to(DEVICE)
                    out = model(imgs)
                    fold_preds.append(out.cpu().numpy())

            fold_preds = np.concatenate(fold_preds).flatten()
            ensemble_preds.append(fold_preds)

            del model
            torch.cuda.empty_cache()

        # Average predictions
        avg_preds = np.mean(ensemble_preds, axis=0)

        # Post-process: Round and Clip
        final_predictions = np.rint(avg_preds).clip(0, 4).astype(int)

        # Create Submission DataFrame
        # Load test metadata to get ID codes
        df_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

        submission_df = pd.DataFrame(
            {"id_code": df_test["id_code"], "diagnosis": final_predictions}
        )

        sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nMetric ({final_qwk}) <= Threshold ({SUBMISSION_THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
