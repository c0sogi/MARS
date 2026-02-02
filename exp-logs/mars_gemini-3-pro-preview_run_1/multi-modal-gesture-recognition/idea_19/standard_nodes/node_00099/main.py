import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from library
from library.config import (
    HYPERPARAMS,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    CHECKPOINT_DIR,
    SUBMISSION_DIR,
    BACKGROUND_CLASS_ID,
    WORKING_DIR,
)
from library.utils import set_seed, decode_predictions, levenshtein_distance
from library.data_loader import GestureDataset, collate_fn
from library.model import GCINet
from library.trainer import Trainer


def main():
    # 1. Setup
    set_seed(HYPERPARAMS["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    # Train Dataset
    train_dataset = GestureDataset(
        TRAIN_METADATA_PATH, split="train", load_cached_data=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=HYPERPARAMS["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # Validation Dataset
    val_dataset = GestureDataset(VAL_METADATA_PATH, split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=HYPERPARAMS["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Training
    trainer = Trainer(train_loader, val_loader, device=device)
    trainer.fit()

    # 4. Evaluation (Load Best Model)
    model = GCINet().to(device)
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    model.eval()

    # Compute Final Metric and Failure Analysis Data
    total_dist = 0
    total_ref_gestures = 0

    # For Failure Analysis
    errors = []
    lengths_list = []
    num_gestures_list = []

    with torch.no_grad():
        for batch in val_loader:
            skeletons, audios, lengths, labels = batch
            skeletons = skeletons.to(device)
            audios = audios.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(skeletons, audios, lengths)
            probs = torch.softmax(logits, dim=2)
            preds = torch.argmax(probs, dim=2)

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            lengths_np = lengths.cpu().numpy()

            for i in range(len(preds_np)):
                curr_len = lengths_np[i]
                p_seq = preds_np[i][:curr_len]
                t_seq = labels_np[i][:curr_len]

                pred_gestures = decode_predictions(
                    p_seq,
                    background_id=BACKGROUND_CLASS_ID,
                    min_len=HYPERPARAMS["min_gesture_length"],
                    median_filter_size=HYPERPARAMS["median_filter_size"],
                )

                target_gestures = decode_predictions(
                    t_seq,
                    background_id=BACKGROUND_CLASS_ID,
                    min_len=1,
                    median_filter_size=1,
                )

                dist = levenshtein_distance(pred_gestures, target_gestures)
                total_dist += dist
                total_ref_gestures += len(target_gestures)

                # Collect stats for failure analysis
                errors.append(dist)
                lengths_list.append(curr_len)
                num_gestures_list.append(len(target_gestures))

    final_metric = (
        total_dist / total_ref_gestures if total_ref_gestures > 0 else float("inf")
    )
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nFailure Analysis:")
    if len(errors) > 0:
        df_analysis = pd.DataFrame(
            {
                "error": errors,
                "seq_length": lengths_list,
                "num_gestures": num_gestures_list,
            }
        )
        corr_len = df_analysis["error"].corr(df_analysis["seq_length"])
        corr_num = df_analysis["error"].corr(df_analysis["num_gestures"])
        print(f"Correlation (Error vs Sequence Length): {corr_len:.10f}")
        print(f"Correlation (Error vs Num Gestures): {corr_num:.10f}")

    # 6. Submission
    THRESHOLD = 0.0765306122
    if final_metric < THRESHOLD:
        print("\nMetric below threshold. Generating submission...")

        # Load Test Data
        test_dataset = GestureDataset(
            TEST_METADATA_PATH, split="test", load_cached_data=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=HYPERPARAMS["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
        )

        submission_lines = []

        with torch.no_grad():
            # Iterate sequentially to map predictions back to sample_ids
            global_idx = 0
            for batch in test_loader:
                skeletons, audios, lengths, _ = batch  # Labels are dummy/empty
                skeletons = skeletons.to(device)
                audios = audios.to(device)
                lengths = lengths.to(device)

                logits = model(skeletons, audios, lengths)
                probs = torch.softmax(logits, dim=2)
                preds = torch.argmax(probs, dim=2)

                preds_np = preds.cpu().numpy()
                lengths_np = lengths.cpu().numpy()

                for i in range(len(preds_np)):
                    curr_len = lengths_np[i]
                    p_seq = preds_np[i][:curr_len]

                    pred_gestures = decode_predictions(
                        p_seq,
                        background_id=BACKGROUND_CLASS_ID,
                        min_len=HYPERPARAMS["min_gesture_length"],
                        median_filter_size=HYPERPARAMS["median_filter_size"],
                    )

                    # Get Sample ID from dataset metadata
                    sample_id = test_dataset.metadata.iloc[global_idx]["sample_id"]
                    global_idx += 1

                    # Format: SessionID,label1,label2...
                    line = f"{sample_id}," + ",".join(map(str, pred_gestures))
                    submission_lines.append(line)

        sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        with open(sub_path, "w") as f:
            for line in submission_lines:
                f.write(line + "\n")
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
