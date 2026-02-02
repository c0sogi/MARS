import os
import torch
import numpy as np
import pandas as pd
import soundfile as sf
from library.config import Config
from library.utils import seed_everything, calculate_auc
from library.data import get_dataloaders
from library.engine import fit_one_seed, generate_submission, predict, WhaleConvNeXt


def analyze_failures(val_ids, y_true, y_pred):
    """
    Performs failure analysis by correlating error magnitude with input features.
    """
    print("\nStarting Failure Analysis...")

    # Calculate Error Magnitude
    errors = np.abs(y_true - y_pred)

    # Load metadata to get file paths
    df_val = pd.read_csv(Config.VAL_CSV)
    # Ensure alignment (val_ids should match df_val['clip'])
    # Create a mapping from clip to filepath
    clip_to_path = pd.Series(df_val.filepath.values, index=df_val.clip).to_dict()

    durations = []

    # Extract features (Original Duration)
    # We process all validation samples. sf.info is fast.
    for clip_id in val_ids:
        rel_path = clip_to_path.get(clip_id)
        if rel_path:
            full_path = os.path.join(Config.INPUT_ROOT, rel_path)
            try:
                info = sf.info(full_path)
                durations.append(info.duration)
            except:
                durations.append(Config.DURATION)
        else:
            durations.append(Config.DURATION)

    # Create DataFrame for correlation analysis
    df_analysis = pd.DataFrame(
        {"error": errors, "duration": durations, "target": y_true, "prediction": y_pred}
    )

    # Calculate correlations
    correlations = df_analysis.corr()["error"].drop("error")

    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    return df_analysis


def main():
    # 1. Configuration Overrides for Fast Baseline
    # We limit epochs and seeds to ensure execution within the time limit.
    Config.EPOCHS = 5
    Config.SEEDS = [42, 101]  # Reduced from 10 to 2 for speed
    Config.NUM_WORKERS = 2

    print(
        f"Running Fast Baseline with {len(Config.SEEDS)} seeds for {Config.EPOCHS} epochs each."
    )

    seed_everything(Config.SEEDS[0])
    device = Config.DEVICE

    # 2. Data Loading
    # Load cached data if available to save time
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=False, load_cached_data=True
    )

    # 3. Training Loop (Ensemble)
    model_paths = []
    val_aucs = []

    for seed in Config.SEEDS:
        seed_everything(seed)
        path, best_auc = fit_one_seed(train_loader, val_loader, seed, device)
        model_paths.append(path)
        val_aucs.append(best_auc)

    print(f"\nTraining Complete. Best Val AUCs per seed: {val_aucs}")

    # 4. Ensemble Validation & Metric Calculation
    print("\nRunning Ensemble Validation...")

    # We need to aggregate predictions from all models on the validation set
    # val_loader returns (data, targets). We need to extract targets once.

    # Get Ground Truth and IDs
    y_true = []
    val_ids = val_loader.dataset.ids  # Access IDs from dataset directly

    # Iterate loader to get targets (order is preserved as shuffle=False)
    for _, targets in val_loader:
        y_true.extend(targets.numpy())
    y_true = np.array(y_true)

    # Accumulate probabilities
    accumulated_probs = np.zeros_like(y_true)

    for model_path in model_paths:
        # Load Model
        model = WhaleConvNeXt(
            backbone_name=Config.BACKBONE,
            pretrained=False,
            in_channels=Config.IN_CHANNELS,
            num_classes=Config.NUM_CLASSES,
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Inference
        probs_list = []
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                outputs = model(data).squeeze(1)
                probs = torch.sigmoid(outputs).cpu().numpy()
                probs_list.append(probs)

        accumulated_probs += np.concatenate(probs_list)

    # Average
    y_pred_ensemble = accumulated_probs / len(model_paths)

    # Calculate Final Metric
    final_metric = calculate_auc(y_true, y_pred_ensemble)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    analyze_failures(val_ids, y_true, y_pred_ensemble)

    # 6. Submission
    # Threshold check
    THRESHOLD = 0.9956103812188066

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(test_loader, model_paths, device)
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
