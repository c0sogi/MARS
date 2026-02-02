import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor

# Import from provided libraries
from library.config import Config, set_seed
from library.dataset import get_dataloader, get_metadata_df
from library.model import AudioMobileNet
from library.engine import train_model, evaluate, generate_submission


def perform_failure_analysis(model, val_loader, val_df, device):
    """
    Analyzes model performance against input characteristics.
    """
    print("\n--- Performing Failure Analysis ---")
    model.eval()

    all_targets = []
    all_probs = []

    # 1. Get Predictions
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.numpy())
            all_probs.append(probs.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    # 2. Calculate Error Magnitude per sample
    # Using Mean Absolute Error per sample as the error metric
    # shape: (n_samples,)
    per_sample_error = np.mean(np.abs(all_targets - all_probs), axis=1)

    # 3. Extract Features for Correlation
    # We need original duration and number of labels
    # Optimization: Read headers in parallel
    def get_duration(filepath):
        try:
            full_path = os.path.join(Config.INPUT_ROOT, filepath)
            info = sf.info(full_path)
            return info.duration
        except:
            return 0.0

    print("Extracting audio metadata for analysis...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        durations = list(executor.map(get_duration, val_df["filepath"].tolist()))

    # Number of labels per sample
    # The 'target' column in val_df might be list or array, but we can also sum the binary targets we have
    num_labels = np.sum(all_targets, axis=1)

    # 4. Calculate Correlations
    df_analysis = pd.DataFrame(
        {"error": per_sample_error, "duration": durations, "num_labels": num_labels}
    )

    corr_duration = df_analysis["error"].corr(df_analysis["duration"])
    corr_labels = df_analysis["error"].corr(df_analysis["num_labels"])

    print(f"Correlation (Error vs Duration): {corr_duration:.6f}")
    print(f"Correlation (Error vs Num Labels): {corr_labels:.6f}")

    # Interpretation
    if abs(corr_duration) > 0.1:
        print(
            f"Observation: Model performance is {'negatively' if corr_duration > 0 else 'positively'} associated with audio duration."
        )
    if abs(corr_labels) > 0.1:
        print(
            f"Observation: Clips with more labels tend to have {'higher' if corr_labels > 0 else 'lower'} error rates."
        )


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # Cite Data Lineage Verification: Verify Sample Submission Structure
    print("Verifying Sample Submission...")
    ss = pd.read_csv(Config.SAMPLE_SUBMISSION)
    print(f"Sample Submission Shape: {ss.shape}")
    if ss.shape[1] != 81:
        print(f"WARNING: Sample submission has {ss.shape[1]} columns. Expected 81.")

    # Verify Config matches
    if Config.NUM_CLASSES != (ss.shape[1] - 1):
        print(
            f"WARNING: Config.NUM_CLASSES ({Config.NUM_CLASSES}) does not match sample submission classes ({ss.shape[1]-1})"
        )

    # Override Config for Fast Baseline
    # A100 is fast, but we limit epochs to ensure quick turnaround
    Config.EPOCHS = 10

    # 2. Data Loading
    print("Loading DataLoaders...")
    train_loader = get_dataloader("train", load_cached_data=True)
    val_loader = get_dataloader("val", load_cached_data=True)
    test_loader = get_dataloader("test", load_cached_data=True)

    # 3. Model Initialization
    print("Initializing Model...")
    model = AudioMobileNet(pretrained=Config.PRETRAINED)
    model = model.to(device)

    # 4. Training
    print("Starting Training...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=Config.EPOCHS,
        patience=3,  # Stricter patience for baseline
    )

    # 5. Final Validation Metric
    print("Computing Final Validation Metrics...")
    val_loss, val_lrap = evaluate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {val_lrap}")

    # 6. Failure Analysis
    # Load raw dataframe to get filepaths for duration extraction
    val_df = get_metadata_df("val", load_cached_data=True)
    perform_failure_analysis(model, val_loader, val_df, device)

    # 7. Submission
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

    # Cite Defensive I/O: Verify the generated file
    print("Verifying generated submission file...")
    sub_check = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Generated Submission Shape: {sub_check.shape}")
    print("First 5 rows of submission:")
    print(sub_check.head())

    print("Runfile execution complete.")


if __name__ == "__main__":
    main()
