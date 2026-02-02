import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import sys
import os

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, jaccard
from library import engine, data, model, inference


def run():
    # ------------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------------
    # Initialize config with debug=False to use the full dataset.
    # We adjust epochs and batch size to optimize for the available A100 GPU and time limit.
    config = Config(debug=False)
    config.epochs = 3
    config.train_batch_size = 32
    config.valid_batch_size = 64

    # Ensure reproducibility
    config.seed_everything()

    print(
        f"Configuration: Epochs={config.epochs}, Batch Size={config.train_batch_size}, Folds={config.n_folds}"
    )

    # ------------------------------------------------------------------------
    # 2. Training Loop (5-Fold CV)
    # ------------------------------------------------------------------------
    print("\n--- Starting Training ---")
    # We train all folds to create a robust ensemble for the final submission
    for fold in range(config.n_folds):
        engine.train_fold(fold, config)

    # ------------------------------------------------------------------------
    # 3. Validation & Failure Analysis
    # ------------------------------------------------------------------------
    print("\n--- Starting Validation and Failure Analysis ---")

    # To perform proper validation on the requested hold-out set (metadata/val.csv),
    # we first need to generate predictions. Since the library merges train/val for CV,
    # we reconstruct the CV splits to identify which samples were used as validation in each fold
    # and generate Out-Of-Fold (OOF) predictions.

    # Load Dataframes to reconstruct IDs
    df_train_meta = pd.read_csv(config.train_path)
    df_val_meta = pd.read_csv(config.val_path)
    # Replicate the concatenation logic from library/data.py
    df_full = pd.concat([df_train_meta, df_val_meta]).reset_index(drop=True)

    # Dictionary to store OOF predictions: textID -> prediction_details
    oof_preds = {}

    device = config.device

    # Re-initialize StratifiedKFold with the same seed to match training splits
    skf = StratifiedKFold(
        n_splits=config.n_folds, shuffle=True, random_state=config.seed
    )
    sentiments = df_full["sentiment"]

    # Iterate through folds to generate predictions for the validation chunk of each fold
    for fold, (_, val_idx) in enumerate(
        skf.split(np.zeros(len(sentiments)), sentiments)
    ):
        print(f"Generating predictions for Fold {fold} validation set...")

        # Get the validation DataLoader for this fold
        _, val_dl = data.get_fold_dls(fold, config)

        # Load the best model checkpoint for this fold
        model_path = config.get_model_path(fold)
        if not os.path.exists(model_path):
            print(f"Warning: Model for fold {fold} not found. Skipping.")
            continue

        net = model.TweetModel(config.model_name, config.dropout)
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.to(device)
        net.eval()

        # Identify the textIDs corresponding to this validation fold
        # val_dl is not shuffled, so it aligns with df_full.iloc[val_idx]
        fold_val_ids = df_full.iloc[val_idx]["textID"].values

        # Inference Loop
        current_idx_pointer = 0

        with torch.no_grad():
            for batch in tqdm(val_dl, disable=True):  # Silent execution
                input_ids = batch["ids"].to(device)
                attention_mask = batch["mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                offsets = batch["offsets"].cpu().numpy()
                orig_texts = batch["orig_text"]
                batch_sentiments = batch["sentiment"]

                # Forward pass
                s_logits, e_logits = net(input_ids, attention_mask, token_type_ids)

                # Get probabilities
                s_probs = torch.softmax(s_logits, dim=1).cpu().numpy()
                e_probs = torch.softmax(e_logits, dim=1).cpu().numpy()
                mask_np = attention_mask.cpu().numpy()

                # Decode predictions
                for i in range(len(orig_texts)):
                    # Reconstruct text as per data processing logic
                    text = " " + " ".join(orig_texts[i].split())
                    sentiment = batch_sentiments[i]
                    offset = offsets[i]

                    pred_str = ""

                    if sentiment == "neutral":
                        pred_str = text
                    else:
                        s_p = s_probs[i] * mask_np[i]
                        e_p = e_probs[i] * mask_np[i]

                        # Joint probability
                        scores = np.outer(s_p, e_p)
                        scores = np.triu(scores)  # Enforce start <= end

                        best_idx = np.argmax(scores)
                        best_s, best_e = np.unravel_index(best_idx, scores.shape)

                        if best_s < len(offset) and best_e < len(offset):
                            pred_str = text[offset[best_s][0] : offset[best_e][1]]
                        else:
                            pred_str = text

                    # Store result mapped to textID
                    tid = fold_val_ids[current_idx_pointer]
                    current_idx_pointer += 1

                    oof_preds[tid] = {"pred": pred_str.strip(), "text_len": len(text)}

        # Clean up to save memory
        del net
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------------
    # 4. Compute Final Metrics
    # ------------------------------------------------------------------------
    # We now evaluate strictly on the "hold-out" validation set (metadata/val.csv)
    # using the OOF predictions we just generated.

    val_jaccards = []
    val_errors = []
    val_lengths = []

    # Iterate over the specific hold-out validation file
    for _, row in df_val_meta.iterrows():
        tid = row["textID"]
        true_selected_text = str(row["selected_text"])

        if tid in oof_preds:
            pred_data = oof_preds[tid]
            pred_text = pred_data["pred"]

            # Compute Metric
            score = jaccard(pred_text, true_selected_text)

            val_jaccards.append(score)
            val_errors.append(1.0 - score)
            val_lengths.append(pred_data["text_len"])

    final_metric = np.mean(val_jaccards)

    # Print the required metric
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error and Input Length
    if len(val_errors) > 1:
        correlation = np.corrcoef(val_errors, val_lengths)[0, 1]
        print(f"Correlation between Error and Input Length: {correlation}")
    else:
        print("Insufficient data for correlation analysis.")

    # ------------------------------------------------------------------------
    # 5. Submission
    # ------------------------------------------------------------------------
    threshold = 0.7092567967346735

    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )
        inference.predict_test(config)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
