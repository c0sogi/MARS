import os
import pandas as pd
import numpy as np
import torch
import soundfile as sf
from scipy.stats import pearsonr

from library.config import TrainConfig, PathConfig, ModelConfig
from library.utils import set_seed, load_checkpoint
from library.dataset import get_dataloaders
from library.model import WhaleEfficientNet
from library.trainer import Trainer


def analyze_failures(model, val_loader, val_meta_path, input_dir, device):
    """
    Performs failure analysis by correlating prediction errors with input signal characteristics.
    """
    print("\n--- Failure Analysis ---")
    model.eval()

    # 1. Get Predictions and Targets
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).squeeze(1)

            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Error Magnitude
    errors = np.abs(all_preds - all_targets)

    # 2. Extract Audio Features for Validation Set
    # We need to map the loader order to the metadata order.
    # The val_loader is sequential (shuffle=False), matching val.csv.
    val_df = pd.read_csv(val_meta_path)

    if len(val_df) != len(errors):
        print(
            f"Warning: Mismatch between validation set size ({len(errors)}) and metadata ({len(val_df)}). Skipping detailed analysis."
        )
        return

    print("Extracting audio features for validation set...")
    mean_amps = []
    std_amps = []
    durations = []

    for _, row in val_df.iterrows():
        full_path = os.path.join(input_dir, row["file_path"])
        try:
            data, sr = sf.read(full_path)
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            mean_amps.append(np.mean(np.abs(data)))
            std_amps.append(np.std(data))
            durations.append(len(data) / sr)
        except Exception:
            mean_amps.append(0)
            std_amps.append(0)
            durations.append(0)

    # 3. Calculate Correlations
    features = {
        "Mean Amplitude": mean_amps,
        "Std Amplitude": std_amps,
        "Duration": durations,
    }

    print("Correlation between Error Magnitude and Input Features:")
    for name, values in features.items():
        if len(values) > 0:
            corr, _ = pearsonr(errors, values)
            print(f"  {name}: {corr:.4f}")


def main():
    # --- Configuration ---
    # Cite solution_lesson_node_00034: Batch Size vs. Training Duration
    # Using larger batch size (64) and standard supervised training
    train_config = TrainConfig()
    path_config = PathConfig()
    model_config = ModelConfig()

    # Create directories
    path_config.create_dirs()

    # Set Seed
    set_seed(train_config.seed)

    print("==================================================")
    print("   Optimized Supervised Training (SpecAugment + Mixup)")
    print("==================================================")

    # Get Dataloaders
    # Cite solution_lesson_node_00024: Regularization Stacking (SpecAugment added in dataset.py)
    dataloaders = get_dataloaders(debug=train_config.debug, load_cached_data=True)

    # Initialize Model
    # Cite solution_lesson_node_00022: Optimize the Supervised Baseline
    model = WhaleEfficientNet(model_config)
    trainer = Trainer(model, dataloaders["train"], dataloaders["val"], train_config)

    # Train
    # Using checkpoints/best_model.pth as the save path
    save_path = os.path.join(path_config.checkpoint_dir, "best_model.pth")
    trainer.train(save_path)

    # ---------------------------------------------------------
    # Validation & Analysis
    # ---------------------------------------------------------
    print("\n[Evaluation] Validating Model...")

    # Reload best weights (already done in train(), but ensuring consistency)
    load_checkpoint(model, save_path, device=train_config.device)

    # Final Validation Metric
    final_auc = trainer.validate()
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    analyze_failures(
        model,
        dataloaders["val"],
        path_config.val_meta,
        path_config.input_dir,
        train_config.device,
    )

    # ---------------------------------------------------------
    # Submission
    # ---------------------------------------------------------
    THRESHOLD = 0.9960914834372254

    if final_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({final_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission(dataloaders["test"], path_config.submission_path)
    else:
        print(
            f"\nValidation AUC ({final_auc}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
