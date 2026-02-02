import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import set_seed, median_filter, rle_decode
from library.model import MPWINet
from library.data_loader import GestureDataset, collate_fn, compute_global_stats


def generate_predictions(
    model_path=os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
    output_path=Config.SUBMISSION_PATH,
    batch_size=Config.BATCH_SIZE,
):
    """
    Generates predictions for the test set using the trained model.

    Args:
        model_path (str): Path to the trained model checkpoint.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    print(f"Initializing Inference...")
    print(f"Model Path: {model_path}")
    print(f"Output Path: {output_path}")

    # 1. Load Statistics for Normalization
    # We prioritize loading the stats used/generated during training to ensure consistency.
    stats = None
    if os.path.exists(Config.TRAIN_METADATA_PATH):
        print("Loading training metadata to ensure correct normalization stats...")
        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        stats = compute_global_stats(train_df)
    elif os.path.exists(Config.STATS_PATH):
        print("Loading stats directly from cache...")
        data = np.load(Config.STATS_PATH)
        stats = (
            torch.from_numpy(data["skel_mean"]),
            torch.from_numpy(data["skel_std"]),
            torch.from_numpy(data["audio_mean"]),
            torch.from_numpy(data["audio_std"]),
        )
    else:
        print(
            "Warning: No training metadata or cached stats found. Normalization may be incorrect."
        )

    # 2. Load Test Dataset
    if not os.path.exists(Config.TEST_METADATA_PATH):
        print(f"Error: Test metadata not found at {Config.TEST_METADATA_PATH}")
        return

    test_dataset = GestureDataset(
        Config.TEST_METADATA_PATH, stats=stats, is_train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 3. Load Model
    model = MPWINet().to(device)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return
    else:
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    model.eval()

    # 4. Inference Loop
    results = []
    print(f"Starting inference on {len(test_dataset)} sequences...")

    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            lengths = batch["lengths"]
            mask = batch["mask"].to(device)
            sample_ids = batch["sample_ids"]

            # Forward pass
            logits = model(skeleton, audio, lengths, mask)  # (B, T, NumClasses)

            # Get hard predictions
            preds = torch.argmax(logits, dim=2)  # (B, T)

            # Move to CPU for post-processing
            preds_np = preds.cpu().numpy()
            lengths_np = lengths.numpy()

            # Process each sequence in the batch
            for i in range(len(sample_ids)):
                sample_id = sample_ids[i]
                valid_len = lengths_np[i]

                # Extract valid frames (remove padding)
                raw_seq = preds_np[i, :valid_len]

                # Post-processing: Median Filter
                smoothed_seq = median_filter(
                    raw_seq, window_size=Config.MEDIAN_FILTER_WINDOW
                )

                # Post-processing: RLE Decode
                decoded_gestures = rle_decode(
                    smoothed_seq,
                    background_label=Config.BACKGROUND_LABEL,
                    min_len=Config.MIN_SEGMENT_LENGTH,
                )

                results.append((sample_id, decoded_gestures))

    # 5. Write Submission
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Writing {len(results)} predictions to {output_path}...")

    try:
        with open(output_path, "w") as f:
            for sample_id, gestures in results:
                # Format: SessionID,Label1,Label2,Label3
                if gestures:
                    gestures_str = ",".join(map(str, gestures))
                    line = f"{sample_id},{gestures_str}\n"
                else:
                    # Handle case with no detected gestures
                    line = f"{sample_id},\n"
                f.write(line)
        print("Submission file generated successfully.")
    except Exception as e:
        print(f"Error writing submission file: {e}")
