import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torchaudio

from library.config import (
    NUM_CLASSES,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE,
    MIXUP_ALPHA,
    CONFIDENCE_THRESHOLD,
    BATCH_SIZE,
    NUM_WORKERS,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    INPUT_DIR,
    SEED,
    get_source_label,
)
from library.utils import set_seed, Mixup
from library.model import get_model
from library.dataset import get_dataloaders
from library.engine import (
    train_model,
    load_noise_files,
    validate,
    generate_submission,
)


def analyze_failures(model, val_loader, device):
    print("\n--- Failure Analysis ---")
    model.eval()

    # Collect predictions and targets
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(max_probs.cpu().numpy())

    # Load validation metadata to map back to files
    df_val = pd.read_csv(VAL_METADATA_PATH)
    # Ensure length match (safety check)
    if len(df_val) != len(all_preds):
        print("Warning: Validation set size mismatch. Using subset for analysis.")
        df_val = df_val.iloc[: len(all_preds)]

    df_val["pred"] = all_preds
    df_val["target"] = all_targets
    df_val["prob"] = all_probs
    df_val["correct"] = (df_val["pred"] == df_val["target"]).astype(int)

    # Calculate Error Magnitude (1 - correct)
    df_val["error"] = 1 - df_val["correct"]

    # Sample for feature extraction (speed optimization)
    sample_size = min(1000, len(df_val))
    df_sample = df_val.sample(n=sample_size, random_state=SEED)

    durations = []
    rms_values = []

    for idx, row in df_sample.iterrows():
        filepath = os.path.join(INPUT_DIR, row["filepath"])
        try:
            # Get Duration
            info = torchaudio.info(filepath)
            dur = info.num_frames / info.sample_rate

            # Get RMS
            wav, _ = torchaudio.load(filepath)
            rms = torch.sqrt(torch.mean(wav**2)).item()
        except:
            dur = 0.0
            rms = 0.0

        durations.append(dur)
        rms_values.append(rms)

    df_sample["duration"] = durations
    df_sample["rms"] = rms_values

    # Correlations
    corr_dur = df_sample["error"].corr(df_sample["duration"])
    corr_rms = df_sample["error"].corr(df_sample["rms"])

    print(f"Correlation (Error vs Duration): {corr_dur:.10f}")
    print(f"Correlation (Error vs RMS): {corr_rms:.10f}")


def main():
    # Setup
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    # load_cached_data=True will use the balanced parquet file if it exists
    # Cite solution_lesson_node_00026: Variance-Aware Balancing implemented in dataset.py
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)
    noise_files = load_noise_files()

    # 2. Model Initialization
    print("\n=== Initializing Model ===")
    model = get_model(num_classes=NUM_CLASSES).to(device)

    # 3. Training Setup
    # Cite solution_lesson_node_00043: Removed Pseudo-Labeling to focus on supervised convergence.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    mixup = Mixup(alpha=MIXUP_ALPHA)

    # 4. Training Loop
    # Cite solution_lesson_node_00029: Training for longer (45 epochs) with Noise Injection
    print(f"\n=== Starting Training for {EPOCHS} Epochs ===")
    model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        EPOCHS,
        PATIENCE,
        mixup,
    )

    # 5. Final Evaluation
    print("\n=== Final Evaluation ===")
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {val_acc}")

    analyze_failures(model, val_loader, device)

    # 6. Submission
    THRESHOLD = 0.9872909698996656
    if val_acc > THRESHOLD:
        print(f"\nValidation metric {val_acc} > {THRESHOLD}. Generating submission...")
        generate_submission(model, device)
    else:
        print(f"\nValidation metric {val_acc} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
