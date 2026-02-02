import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
from scipy.stats import pearsonr

# Import from library
from library.config import PathConfig, ModelConfig, TrainConfig, FeatureConfig
from library.utils import set_seed, calculate_log_loss, clip_probabilities
from library.data_processing import load_data, get_tfidf_features, StylometricDataset
from library.dapt import run_mlm_pretraining
from library.models import StylometricTransformer, StatisticalPredictor
from library.training_utils import run_fold_training
from library.ensemble import LengthAdaptiveBlender


def get_preds_from_loader(model, loader, device):
    """Helper to generate probabilities from a model and loader."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k != "aux_targets" and k != "labels"
            }
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(**inputs)
                logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def main():
    # 1. Setup
    print("--- Starting Runfile Execution ---")
    set_seed(TrainConfig.SEED)
    PathConfig.create_dirs()

    # Override Configs for Fast Baseline (Time Limit: ~40 mins)
    # We use 2 Folds and 2 Epochs to ensure completion.
    TrainConfig.N_FOLDS = 2
    TrainConfig.EPOCHS = 2
    TrainConfig.DAPT_EPOCHS = 1
    print(
        f"Config Overrides: Folds={TrainConfig.N_FOLDS}, Epochs={TrainConfig.EPOCHS}, DAPT Epochs={TrainConfig.DAPT_EPOCHS}"
    )

    # 2. Data Loading
    print("Loading Data...")
    train_df, val_df, test_df = load_data()

    # 3. DAPT (Domain Adaptive Pre-training)
    print("Running DAPT...")
    run_mlm_pretraining(debug=False, load_cached_data=True)

    # 4. Feature Generation (TF-IDF)
    print("Generating TF-IDF Features...")
    X_train_tfidf, X_val_tfidf, X_test_tfidf = get_tfidf_features(
        train_df["text"], val_df["text"], test_df["text"], load_cached_data=True
    )

    # 5. Statistical Branch
    print("Training Statistical Models...")
    stat_model = StatisticalPredictor(seed=TrainConfig.SEED)
    # Convert labels to indices
    y_train_indices = [ModelConfig.LABEL2ID[l] for l in train_df["author"]]
    stat_model.fit(X_train_tfidf, y_train_indices)

    # Predict
    stat_val_probs = stat_model.predict_proba(X_val_tfidf)
    stat_test_probs = stat_model.predict_proba(X_test_tfidf)

    # 6. Neural Branch (Stratified K-Fold)
    # We need to aggregate predictions on val_df and test_df from multiple folds
    neural_val_preds = {
        backbone: np.zeros((len(val_df), ModelConfig.NUM_LABELS))
        for backbone in ModelConfig.BACKBONES
    }
    neural_test_preds = {
        backbone: np.zeros((len(test_df), ModelConfig.NUM_LABELS))
        for backbone in ModelConfig.BACKBONES
    }

    # Prepare fixed loaders for Validation and Test (Hold-out)
    # We will reuse these across folds for inference
    val_texts = val_df["text"].tolist()
    test_texts = test_df["text"].tolist()
    val_labels = [ModelConfig.LABEL2ID[l] for l in val_df["author"]]

    skf = StratifiedKFold(
        n_splits=TrainConfig.N_FOLDS, shuffle=True, random_state=TrainConfig.SEED
    )

    for backbone in ModelConfig.BACKBONES:
        print(f"\nProcessing Backbone: {backbone}")
        tokenizer = AutoTokenizer.from_pretrained(backbone)

        # Create Datasets for Hold-out Val and Test
        ds_val_holdout = StylometricDataset(
            val_texts,
            labels=val_df["author"],
            tokenizer=tokenizer,
            max_length=ModelConfig.MAX_LENGTH,
        )
        ds_test_holdout = StylometricDataset(
            test_texts, tokenizer=tokenizer, max_length=ModelConfig.MAX_LENGTH
        )

        dl_val_holdout = DataLoader(
            ds_val_holdout,
            batch_size=TrainConfig.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=TrainConfig.NUM_WORKERS,
        )
        dl_test_holdout = DataLoader(
            ds_test_holdout,
            batch_size=TrainConfig.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=TrainConfig.NUM_WORKERS,
        )

        fold_val_probs_list = []
        fold_test_probs_list = []

        # K-Fold Loop on Train Data
        for fold_idx, (train_idx, valid_idx) in enumerate(
            skf.split(train_df, train_df["author"])
        ):
            print(f"  Fold {fold_idx + 1}/{TrainConfig.N_FOLDS}")

            # Split Data
            fold_train_df = train_df.iloc[train_idx]
            fold_valid_df = train_df.iloc[valid_idx]

            # Create Datasets
            ds_train = StylometricDataset(
                fold_train_df["text"],
                labels=fold_train_df["author"],
                tokenizer=tokenizer,
                max_length=ModelConfig.MAX_LENGTH,
            )
            ds_valid = StylometricDataset(
                fold_valid_df["text"],
                labels=fold_valid_df["author"],
                tokenizer=tokenizer,
                max_length=ModelConfig.MAX_LENGTH,
            )

            # Create Loaders
            dl_train = DataLoader(
                ds_train,
                batch_size=TrainConfig.BATCH_SIZE,
                shuffle=True,
                num_workers=TrainConfig.NUM_WORKERS,
                drop_last=True,
            )
            dl_valid = DataLoader(
                ds_valid,
                batch_size=TrainConfig.BATCH_SIZE * 2,
                shuffle=False,
                num_workers=TrainConfig.NUM_WORKERS,
            )

            # Initialize Model
            model = StylometricTransformer(
                backbone_name=backbone, num_labels=ModelConfig.NUM_LABELS
            )

            # Train
            # run_fold_training modifies model in-place to best weights
            run_fold_training(
                model, dl_train, dl_valid, fold_idx, backbone, epochs=TrainConfig.EPOCHS
            )

            # Inference on Hold-out Val and Test
            model.to(TrainConfig.DEVICE)
            probs_val = get_preds_from_loader(model, dl_val_holdout, TrainConfig.DEVICE)
            probs_test = get_preds_from_loader(
                model, dl_test_holdout, TrainConfig.DEVICE
            )

            fold_val_probs_list.append(probs_val)
            fold_test_probs_list.append(probs_test)

            # Clean up
            del model, ds_train, ds_valid, dl_train, dl_valid
            torch.cuda.empty_cache()

        # Average predictions across folds
        neural_val_preds[backbone] = np.mean(fold_val_probs_list, axis=0)
        neural_test_preds[backbone] = np.mean(fold_test_probs_list, axis=0)

        del tokenizer, ds_val_holdout, ds_test_holdout, dl_val_holdout, dl_test_holdout
        torch.cuda.empty_cache()

    # 7. Ensemble (Length Adaptive Blending)
    print("\nFitting Length-Adaptive Blender...")

    # Prepare dictionary of predictions for validation set
    val_preds_dict = {"stat": stat_val_probs, **neural_val_preds}

    # Prepare dictionary of predictions for test set
    test_preds_dict = {"stat": stat_test_probs, **neural_test_preds}

    blender = LengthAdaptiveBlender(
        n_bins=FeatureConfig.N_LENGTH_BINS, seed=TrainConfig.SEED
    )
    blender.fit(val_preds_dict, val_labels, val_texts)

    # Generate final predictions
    final_val_probs = blender.predict(val_preds_dict, val_texts)
    final_test_probs = blender.predict(test_preds_dict, test_texts)

    # 8. Evaluation & Failure Analysis
    print("\n--- Evaluation ---")
    final_metric = calculate_log_loss(val_labels, final_val_probs)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate per-sample loss
    # We need to get the probability assigned to the true class
    # y_pred_clipped is handled inside calculate_log_loss, let's replicate for analysis
    row_sums = final_val_probs.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    probs_rescaled = final_val_probs / row_sums[:, np.newaxis]
    probs_clipped = clip_probabilities(probs_rescaled)

    # Extract prob of true class
    true_class_probs = probs_clipped[np.arange(len(val_labels)), val_labels]
    sample_losses = -np.log(true_class_probs)

    # Features for correlation
    lengths = np.array([len(t) for t in val_texts])
    punct_counts = np.array(
        [
            sum(1 for c in t if c in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
            for t in val_texts
        ]
    )

    corr_len, _ = pearsonr(sample_losses, lengths)
    corr_punct, _ = pearsonr(sample_losses, punct_counts)

    print("\n--- Failure Analysis ---")
    print(f"Correlation (Loss vs Text Length): {corr_len:.4f}")
    print(f"Correlation (Loss vs Punctuation Count): {corr_punct:.4f}")

    # 9. Submission
    THRESHOLD = 0.2435629959371868
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        blender.generate_submission(
            test_df["id"], final_test_probs, PathConfig.SUBMISSION_FILE
        )
    else:
        print(
            f"\nMetric ({final_metric}) did not beat threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
