import os
import torch
import pandas as pd
import numpy as np
import torchaudio
import soundfile as sf
from library.config import Config
from library.trainer import train_model, validate, predict_submission
from library.dataset import get_dataloaders
from library.model import ConvNeXtAudio


def run_pipeline():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for the task requirements
    # 12 epochs is a good balance for ConvNeXt-Tiny to converge reasonably well
    # without exceeding the time limit.
    Config.epochs = 12
    Config.subset_size = None  # Use full dataset for best performance

    # Ensure reproducibility
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    print("=== Starting Training Phase ===")
    # train_model handles the loop, saving best_model.pth, and generating an initial submission
    train_model(load_cached_data=True, subset_size=Config.subset_size, patience=5)

    # -------------------------------------------------------------------------
    # 3. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n=== Starting Validation & Failure Analysis ===")
    device = torch.device(Config.device)

    # Load the best model
    model = ConvNeXtAudio(
        model_name=Config.model_name, num_classes=Config.num_classes, pretrained=False
    )
    model_path = Config.model_save_path
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Get DataLoaders (we only need val_loader here)
    _, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, subset_size=Config.subset_size
    )

    # A. Calculate Final Metric
    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # Print required metric
    print(f"Final Validation Metric: {val_acc}")

    # B. Failure Analysis
    print("Performing failure analysis on validation set...")

    # 1. Get Per-Sample Predictions
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # 2. Calculate Error Mask (1 for incorrect, 0 for correct)
    errors = [1 if p != l else 0 for p, l in zip(all_preds, all_labels)]

    # 3. Extract Features for Correlation
    # We iterate through the validation dataset metadata to get file paths
    # Note: val_loader (shuffle=False) preserves order of val_dataset.df
    df_val = val_loader.dataset.df

    durations = []
    rmss = []
    zcrs = []

    # Limit analysis if dataset is huge, but 12k is manageable.
    # We'll process all to be accurate.
    print(f"Extracting signal features for {len(df_val)} validation files...")

    for idx, row in df_val.iterrows():
        filepath = os.path.join(Config.input_root, row["filepath"])

        # Default values in case of read error
        d, r, z = 0.0, 0.0, 0.0

        if os.path.exists(filepath):
            try:
                # Duration via soundfile (fast)
                info = sf.info(filepath)
                d = info.duration

                # Signal features via torchaudio
                # Load only necessary duration to speed up if file is huge (though these are short)
                wav, sr = torchaudio.load(filepath)
                wav = torch.mean(wav, dim=0)  # Mono

                # RMS
                r = torch.sqrt(torch.mean(wav**2)).item()

                # Zero Crossing Rate
                # Simple approximation
                z = (
                    torch.sum(torch.abs(torch.diff(torch.sign(wav)))) / 2 / wav.shape[0]
                ).item()

            except Exception:
                pass

        durations.append(d)
        rmss.append(r)
        zcrs.append(z)

    # 4. Calculate Correlations
    df_analysis = pd.DataFrame(
        {"error": errors, "duration": durations, "rms": rmss, "zcr": zcrs}
    )

    print("Correlation between Error Magnitude and Input Features:")
    correlations = df_analysis.corr()["error"].drop("error")
    print(correlations)

    # -------------------------------------------------------------------------
    # 4. Submission Logic
    # -------------------------------------------------------------------------
    THRESHOLD = 0.9596989966555184

    if val_acc > THRESHOLD:
        print(f"\nValidation metric ({val_acc}) meets threshold ({THRESHOLD}).")
        # Check if submission exists (created by train_model)
        if not os.path.exists(Config.submission_path):
            print("Generating submission file...")
            predict_submission(test_loader, device)
        else:
            print("Submission file already generated.")
    else:
        print(f"\nValidation metric ({val_acc}) does NOT meet threshold ({THRESHOLD}).")
        # Delete submission if it exists
        if os.path.exists(Config.submission_path):
            print("Removing submission file...")
            os.remove(Config.submission_path)


if __name__ == "__main__":
    run_pipeline()
