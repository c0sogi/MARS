import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import pandas as pd
import nltk
import sys

# Import from the provided library files
from library.config import Config
from library.dataset import GestureDataset
from library.model import SA_AKN
from library.loss import CascadedLoss
from library.train_eval import (
    train_epoch,
    evaluate,
    run_inference_on_sequence,
    decode_predictions,
)
from library.data_utils import save_submission


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    Config.setup()

    # Ensure reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading datasets...")
    # Load training and validation datasets
    # load_cached_data=True ensures we use preprocessed .npz files if available
    train_dataset = GestureDataset("train", load_cached_data=True)
    val_dataset = GestureDataset("val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    model = SA_AKN().to(device)
    criterion = CascadedLoss().to(device)

    # Use Adam as per design (avoiding AdamW for Recurrent stability)
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # ==========================================
    # 4. Training Loop
    # ==========================================
    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_score = evaluate(model, val_dataset, device)

        duration = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Error Rate: {val_score:.6f} | Time: {duration:.2f}s"
        )

        # Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  New best model saved! Score: {best_score:.6f}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered after {patience_counter} epochs without improvement."
            )
            break

    print("-" * 30)
    print(f"Final Validation Metric: {best_score}")
    print("-" * 30)

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("Performing Failure Analysis on Validation Set...")

    # Load best model
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: Best model not found. Using current model state.")

    model.eval()

    errors = []
    seq_lengths = []
    num_gestures_list = []

    # Inference parameters
    stride = Config.WINDOW_STRIDE_TEST
    window_size = Config.WINDOW_SIZE

    # Iterate over validation set to gather stats
    for i in range(len(val_dataset.sample_ids)):
        # Get features
        features_np = val_dataset.processed_features[i]
        features = torch.from_numpy(features_np).float().to(device)

        # Run inference
        avg_probs = run_inference_on_sequence(
            model, features, device, window_size, stride
        )

        # Decode
        pred_seq = decode_predictions(avg_probs)

        # Ground Truth
        gt_seq = [int(l["id"]) for l in val_dataset.raw_labels_meta[i]]

        # Calculate Metric (Levenshtein Distance)
        dist = nltk.edit_distance(pred_seq, gt_seq)

        # Collect stats
        errors.append(dist)
        seq_lengths.append(features_np.shape[0])
        num_gestures_list.append(len(gt_seq))

    # Calculate Correlations
    df_analysis = pd.DataFrame(
        {"error": errors, "length": seq_lengths, "num_gestures": num_gestures_list}
    )

    # Pearson correlation
    corr_len = df_analysis["error"].corr(df_analysis["length"])
    corr_num = df_analysis["error"].corr(df_analysis["num_gestures"])

    print(f"Correlation (Error vs Sequence Length): {corr_len:.6f}")
    print(f"Correlation (Error vs Num Gestures): {corr_num:.6f}")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    THRESHOLD = 0.2251

    if best_score < THRESHOLD:
        print(
            f"\nValidation score ({best_score:.6f}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Dataset
        test_dataset = GestureDataset("test", load_cached_data=True)
        predictions = []
        sample_ids = test_dataset.sample_ids

        for i in range(len(sample_ids)):
            features_np = test_dataset.processed_features[i]
            features = torch.from_numpy(features_np).float().to(device)

            # Inference
            avg_probs = run_inference_on_sequence(
                model, features, device, window_size, stride
            )

            # Decode
            pred_seq = decode_predictions(avg_probs)
            predictions.append(pred_seq)

        # Save Submission
        save_submission(predictions, sample_ids, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nValidation score ({best_score:.6f}) does not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
