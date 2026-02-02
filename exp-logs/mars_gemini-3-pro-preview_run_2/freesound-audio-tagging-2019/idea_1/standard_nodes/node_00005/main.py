import os
import sys
import torch
import numpy as np
import soundfile as sf
from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model import AudioMobileNet
from library.engine import run, validate, predict


def failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude (BCE loss) and audio duration.
    """
    print("\n==== Failure Analysis ====")
    model.eval()

    all_losses = []
    all_durations = []

    # Loss function for error magnitude (per sample, averaged over classes)
    # reduction='none' allows us to get the loss for each sample individually
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    # Access the dataframe to map fname to file_path for metadata retrieval
    val_df = val_loader.dataset.df.set_index("fname")

    with torch.no_grad():
        for images, targets, fnames in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            # Calculate loss per sample:
            # outputs: (Batch, Classes), targets: (Batch, Classes)
            # loss: (Batch, Classes) -> mean(dim=1) -> (Batch,)
            loss = criterion(outputs, targets).mean(dim=1)
            all_losses.extend(loss.cpu().numpy())

            # Get durations for the current batch
            for fname in fnames:
                try:
                    rel_path = val_df.loc[fname, "file_path"]
                    full_path = os.path.join(Config.INPUT_ROOT, rel_path)
                    # Use soundfile to get metadata (duration) quickly without decoding the whole file
                    info = sf.info(full_path)
                    all_durations.append(info.duration)
                except Exception as e:
                    # Fallback if file read fails (should not happen with curated data)
                    all_durations.append(0.0)

    # Calculate Pearson Correlation using Numpy
    if len(all_losses) > 1 and len(all_durations) > 1:
        corr = np.corrcoef(all_durations, all_losses)[0, 1]
    else:
        corr = 0.0

    print(
        f"Correlation between Error Magnitude (BCE Loss) and Input Feature (Duration): {corr:.4f}"
    )


def main():
    # ==== 1. Setup & Configuration ====
    # Increase epochs for better convergence within time limit
    Config.MAX_EPOCHS = 50

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ==== 2. Data Loading ====
    print("Loading Datasets...")
    # load_cached_data=True uses parquet files in ./working if available for speed
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # ==== 3. Model Initialization ====
    print("Initializing Model...")
    model = AudioMobileNet().to(device)

    # ==== 4. Training & Inference ====
    # The run function handles:
    # - Training loop
    # - Validation monitoring & Scheduler
    # - Early Stopping & Checkpointing (saving best_model.pth)
    # - Loading the best model state
    # Cite solution_lesson_node_00004: Synchronizing Early Stopping with Learning Rate Decay
    # Increased patience to 7 to allow model to converge after LR decay.
    run(
        model,
        train_loader,
        val_loader,
        Config.MAX_EPOCHS,
        device,
        patience=7,
    )

    # ==== 5. Final Validation Assessment ====
    print("Performing Final Validation Assessment...")

    # The 'model' object now contains the best state loaded by run()
    criterion = torch.nn.BCEWithLogitsLoss()

    # Compute metrics on the full validation set
    val_loss, val_score = validate(model, val_loader, criterion, device)

    # Print the required metric string with full precision
    print(f"Final Validation Metric: {val_score}")

    # Generate Submission if threshold passed
    if val_score > 0.6688040623183806:
        print("Score threshold passed. Generating submission...")
        predict(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"Score {val_score} did not pass threshold 0.6688040623183806. Skipping submission."
        )

    # ==== 6. Failure Analysis ====
    failure_analysis(model, val_loader, device)


if __name__ == "__main__":
    main()
