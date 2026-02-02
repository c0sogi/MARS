import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data_processing import load_data, InsultDataset
from library.model import InsultDetector
from library.trainer import Trainer


def get_text_features(texts):
    """
    Extracts basic features from text for failure analysis.
    """
    lengths = []
    caps_ratios = []
    exclam_counts = []

    for t in texts:
        s_t = str(t)
        l = len(s_t)
        lengths.append(l)
        caps_ratios.append(sum(1 for c in s_t if c.isupper()) / max(1, l))
        exclam_counts.append(s_t.count("!"))

    return np.array(lengths), np.array(caps_ratios), np.array(exclam_counts)


def inference_loop(model, loader, device):
    """
    Custom inference loop to avoid bug in library.trainer.predict
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            svd_features = batch["svd_features"].to(device)

            outputs = model(input_ids, attention_mask, svd_features)
            preds.append(torch.sigmoid(outputs).cpu().numpy())

    return np.concatenate(preds)


def main():
    # 1. Setup
    seed_everything(Config.seed)
    device = Config.device
    print(f"Using device: {device}")

    # 2. Load Data
    # load_data returns train, val, test datasets based on metadata splits.
    # We combine train and val for K-Fold CV.
    print("Loading data...")
    train_ds_part, val_ds_part, test_ds = load_data(load_cached_data=True)

    # Combine training and validation data for CV
    all_texts = np.concatenate([train_ds_part.texts, val_ds_part.texts])
    all_svd = np.concatenate([train_ds_part.svd_features, val_ds_part.svd_features])
    all_labels = np.concatenate([train_ds_part.labels, val_ds_part.labels])

    tokenizer = train_ds_part.tokenizer

    print(f"Total training samples: {len(all_texts)}")
    print(f"Test samples: {len(test_ds)}")

    # 3. Prepare Test Loader
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 4. Stratified K-Fold Cross Validation
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    oof_preds = np.zeros(len(all_texts))
    test_preds_accumulator = np.zeros((len(test_ds), 1))

    print(f"Starting {Config.n_folds}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_texts, all_labels)):
        print(f"\n{'='*10} Fold {fold} {'='*10}")

        # Create Fold Datasets
        train_fold_ds = InsultDataset(
            texts=all_texts[train_idx],
            svd_features=all_svd[train_idx],
            labels=all_labels[train_idx],
            tokenizer=tokenizer,
            max_len=Config.max_len,
        )

        val_fold_ds = InsultDataset(
            texts=all_texts[val_idx],
            svd_features=all_svd[val_idx],
            labels=all_labels[val_idx],
            tokenizer=tokenizer,
            max_len=Config.max_len,
        )

        # Create Fold Loaders
        train_loader = DataLoader(
            train_fold_ds,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_fold_ds,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Initialize Model and Trainer
        model = InsultDetector().to(device)
        trainer = Trainer(model, train_loader, val_loader, device)

        # Train
        best_model_path = trainer.fit(fold)

        # Load Best Model for Inference
        print(f"Loading best model for Fold {fold} inference...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device)

        # OOF Inference
        val_preds = inference_loop(model, val_loader, device)
        oof_preds[val_idx] = val_preds.flatten()

        # Test Inference
        fold_test_preds = inference_loop(model, test_loader, device)
        test_preds_accumulator += fold_test_preds

        # Cleanup to save memory
        del model, trainer, train_loader, val_loader, train_fold_ds, val_fold_ds
        torch.cuda.empty_cache()
        gc.collect()

    # 5. Global Validation Assessment
    final_auc = calculate_metric(all_labels, oof_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate error magnitude
    errors = np.abs(all_labels - oof_preds)

    # Extract features for analysis
    lengths, caps_ratios, exclam_counts = get_text_features(all_texts)

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "char_length": lengths,
            "caps_ratio": caps_ratios,
            "exclam_count": exclam_counts,
        }
    )

    # Calculate correlations
    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Model Error and Input Features:")
    print(correlations)

    # 7. Submission Generation
    threshold = 0.9586453201970443

    if final_auc > threshold:
        print(
            f"\nValidation AUC ({final_auc}) > Threshold ({threshold}). Generating submission..."
        )

        avg_test_preds = test_preds_accumulator / Config.n_folds

        # Load sample submission to preserve format
        try:
            sub_df = pd.read_csv(Config.sample_submission_path)
        except Exception:
            # Fallback if sample submission is missing, create from test metadata
            # Assuming test_ds order matches metadata/test.csv which matches input/test.csv
            sub_df = pd.DataFrame(index=range(len(avg_test_preds)))
            sub_df["Insult"] = 0  # Placeholder

        # Assign predictions
        if len(sub_df) == len(avg_test_preds):
            sub_df["Insult"] = avg_test_preds

            # Ensure output directory exists
            os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)

            sub_df.to_csv(Config.submission_path, index=False)
            print(f"Submission saved to {Config.submission_path}")
        else:
            print(
                f"Error: Submission dataframe length ({len(sub_df)}) does not match predictions ({len(avg_test_preds)})."
            )
    else:
        print(
            f"\nValidation AUC ({final_auc}) did not meet the threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
