import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data as data
import library.model as model_lib
import library.trainer as trainer_lib


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Set seed for reproducibility
    utils.set_seed(config.SEED)

    # Override config for Fast Baseline
    # We limit the training data size and epochs to ensure quick execution
    config.DEBUG_SAMPLE_SIZE = 50000
    config.NUM_EPOCHS = 2
    config.BATCH_SIZE = 256  # Safe batch size for A100

    print(
        f"Fast Baseline Config: Samples={config.DEBUG_SAMPLE_SIZE}, Epochs={config.NUM_EPOCHS}"
    )

    # ==========================================
    # 2. Training (on Subset)
    # ==========================================
    print("\n--- Starting Training ---")
    # run_training handles data loading (subset), model init, and training loop
    # We pass load_cached_data=True, but since DEBUG_SAMPLE_SIZE is set,
    # the data module will likely recompute the subset.
    model, tokenizer, encoder = trainer_lib.run_training(load_cached_data=True)

    # Ensure model is in eval mode
    model.eval()

    # ==========================================
    # 3. Full Validation & Metric
    # ==========================================
    print("\n--- Starting Full Validation ---")

    # We must validate on the FULL validation set, not the debug subset.
    # We manually load and process the validation data to use the SAME tokenizer/encoder.

    # Load Metadata
    val_meta = pd.read_csv(config.VAL_META_PATH)

    # Load Raw Data (Title + Body)
    df_raw_train = pd.read_csv(
        os.path.join(config.INPUT_DIR, "train.csv"), usecols=["Id", "Title", "Body"]
    )

    # Merge
    val_df = pd.merge(val_meta, df_raw_train, on="Id", how="left")
    val_df["Title"] = val_df["Title"].fillna("")
    val_df["Body"] = val_df["Body"].fillna("")
    val_df["Tags"] = val_df["Tags"].fillna("")

    # Preprocess
    print("Tokenizing full validation set...")
    val_texts = (val_df["Title"] + " " + val_df["Body"]).tolist()
    val_tokens = tokenizer.transform(val_texts)
    val_labels = encoder.transform(val_df["Tags"].tolist())

    # Create DataLoader
    val_dataset = data.StackExchangeDataset(val_tokens, val_labels)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE * 2,  # Inference can handle larger batches
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    # Validation Loop
    all_preds = []
    all_targets = []
    all_lengths = []

    print("Running validation inference...")
    with torch.no_grad():
        for tokens, labels in val_loader:
            tokens = tokens.to(config.DEVICE)

            # Forward
            logits = model(tokens)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())

            # Calculate input lengths (non-padding) for failure analysis
            # Pad token is 0
            lengths = (tokens != 0).sum(dim=1).cpu().numpy()
            all_lengths.append(lengths)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_lengths = np.concatenate(all_lengths, axis=0)

    # Calculate Metric
    val_f1 = utils.calculate_f1_score(
        all_preds, all_targets, threshold=config.PREDICTION_THRESHOLD
    )
    print(f"Final Validation Metric: {val_f1}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n--- Failure Analysis ---")

    # Binarize predictions
    preds_binary = (all_preds >= config.PREDICTION_THRESHOLD).astype(int)
    targets_binary = all_targets.astype(int)

    # Calculate Sample-wise F1
    # F1 = 2 * (intersection) / (pred_sum + target_sum)
    intersection = (preds_binary * targets_binary).sum(axis=1)
    pred_sum = preds_binary.sum(axis=1)
    target_sum = targets_binary.sum(axis=1)

    epsilon = 1e-9
    sample_f1 = 2 * intersection / (pred_sum + target_sum + epsilon)

    # Error Magnitude
    errors = 1.0 - sample_f1

    # Correlation
    # Using numpy for correlation
    if len(errors) > 1:
        corr_matrix = np.corrcoef(errors, all_lengths)
        correlation = corr_matrix[0, 1]
    else:
        correlation = 0.0

    print(f"Correlation between Error Magnitude (1-F1) and Input Length: {correlation}")

    # ==========================================
    # 5. Submission
    # ==========================================
    TARGET_METRIC = 0.33488

    if val_f1 > TARGET_METRIC:
        print(
            f"\nMetric ({val_f1}) > Threshold ({TARGET_METRIC}). Generating submission..."
        )

        # Load Test Data
        test_meta = pd.read_csv(config.TEST_META_PATH)
        df_raw_test = pd.read_csv(
            os.path.join(config.INPUT_DIR, "test.csv"), usecols=["Id", "Title", "Body"]
        )

        test_df = pd.merge(test_meta, df_raw_test, on="Id", how="left")
        test_df["Title"] = test_df["Title"].fillna("")
        test_df["Body"] = test_df["Body"].fillna("")

        print("Tokenizing full test set...")
        test_texts = (test_df["Title"] + " " + test_df["Body"]).tolist()
        test_tokens = tokenizer.transform(test_texts)
        test_ids = test_df["Id"].values

        test_dataset = data.StackExchangeDataset(test_tokens, ids=test_ids)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
        )

        # Inference Loop
        submission_ids = []
        submission_tags = []
        class_names = np.array(encoder.classes_)

        print("Running test inference...")
        with torch.no_grad():
            for tokens, ids in test_loader:
                tokens = tokens.to(config.DEVICE)

                logits = model(tokens)
                probs = torch.sigmoid(logits)
                probs = probs.cpu().numpy()
                ids = ids.numpy()

                preds_binary = probs >= config.PREDICTION_THRESHOLD

                for i in range(len(ids)):
                    row_mask = preds_binary[i]
                    predicted_tags = class_names[row_mask]
                    tag_str = " ".join(predicted_tags)

                    submission_ids.append(ids[i])
                    submission_tags.append(tag_str)

        # Save Submission
        os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
        df_sub = pd.DataFrame({"Id": submission_ids, "Tags": submission_tags})
        df_sub.to_csv(config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {config.SUBMISSION_FILE}")

    else:
        print(
            f"\nMetric ({val_f1}) <= Threshold ({TARGET_METRIC}). Skipping submission."
        )


if __name__ == "__main__":
    main()
