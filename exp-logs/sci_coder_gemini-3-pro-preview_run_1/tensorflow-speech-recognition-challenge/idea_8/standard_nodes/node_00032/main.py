import os
import torch
import numpy as np
import pandas as pd
import torchaudio
from library import config, utils, dataset, model, train


def run():
    # =========================================================================
    # 1. Configuration Setup
    # =========================================================================
    # Override configuration for a fast baseline run.
    # Reducing epochs to 10 ensures the run completes within the 2-hour limit.
    config.TrainConfig.epochs = 10

    # Set seeds for reproducibility
    utils.set_seed(config.TrainConfig.seed)

    # =========================================================================
    # 2. Model Training
    # =========================================================================
    print("Initializing Trainer...")
    trainer = train.Trainer()

    print("Starting Training Loop...")
    trainer.fit()

    # =========================================================================
    # 3. Validation & Metric Calculation
    # =========================================================================
    print("Performing Final Validation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best model checkpoint
    best_model = model.ContextAwareEfficientNet().to(device)
    checkpoint_path = config.TrainConfig.checkpoint_path

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    best_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    best_model.eval()

    # Use the validation loader from the trainer
    val_loader = trainer.val_loader
    mapper = utils.LabelMapper()

    correct_count = 0
    total_count = 0

    # Lists to store data for failure analysis
    error_flags = []  # 0 for correct, 1 for error
    file_paths = []  # Corresponding file paths

    # Get all filepaths from the dataset to align with loader iteration
    val_df = val_loader.dataset.df
    all_val_paths = val_df["filepath"].tolist()

    with torch.no_grad():
        batch_start_idx = 0
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.size(0)

            # Forward pass
            outputs = best_model(inputs)
            _, pred_indices = torch.max(outputs, 1)

            # Decode to fine-grained labels
            pred_labels_fine = mapper.decode(pred_indices)
            target_labels_fine = mapper.decode(targets)

            # Map to 12-class submission format
            pred_sub = [mapper.map_to_submission(l) for l in pred_labels_fine]
            target_sub = [mapper.map_to_submission(l) for l in target_labels_fine]

            # Calculate accuracy and track errors
            for i in range(batch_size):
                is_correct = pred_sub[i] == target_sub[i]
                if is_correct:
                    correct_count += 1

                error_flags.append(0 if is_correct else 1)

                # Store corresponding filepath
                current_idx = batch_start_idx + i
                if current_idx < len(all_val_paths):
                    file_paths.append(all_val_paths[current_idx])

            total_count += batch_size
            batch_start_idx += batch_size

    final_acc = correct_count / total_count if total_count > 0 else 0.0

    # Print the required metric
    print(f"Final Validation Metric: {final_acc}")

    # =========================================================================
    # 4. Failure Analysis
    # =========================================================================
    print("Running Failure Analysis...")

    # Sample a subset to analyze (to keep runtime low)
    n_analysis = 1000
    if len(error_flags) > n_analysis:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(error_flags), n_analysis, replace=False)
    else:
        indices = np.arange(len(error_flags))

    sampled_errors = [error_flags[i] for i in indices]
    sampled_paths = [file_paths[i] for i in indices]

    durations = []
    rmss = []
    input_root = "./input"

    for rel_path in sampled_paths:
        full_path = os.path.join(input_root, rel_path)
        try:
            # Extract Duration
            info = torchaudio.info(full_path)
            dur = info.num_frames / info.sample_rate
            durations.append(dur)

            # Extract RMS
            wav, _ = torchaudio.load(full_path)
            rms = torch.sqrt(torch.mean(wav**2)).item()
            rmss.append(rms)
        except Exception:
            # Handle potentially missing or corrupt files gracefully
            durations.append(0.0)
            rmss.append(0.0)

    # Calculate Correlations
    if len(set(sampled_errors)) > 1:
        corr_dur = np.corrcoef(sampled_errors, durations)[0, 1]
        corr_rms = np.corrcoef(sampled_errors, rmss)[0, 1]
        print(f"Correlation (Error vs Duration): {corr_dur}")
        print(f"Correlation (Error vs RMS): {corr_rms}")
    else:
        print(
            "Correlation analysis skipped: Model predictions are either all correct or all wrong."
        )

    # =========================================================================
    # 5. Submission Generation
    # =========================================================================
    THRESHOLD = 0.9872909698996656

    if final_acc > THRESHOLD:
        print(
            f"Validation metric {final_acc} exceeds threshold {THRESHOLD}. Generating submission..."
        )

        # Reload dataloaders to get the test loader (Trainer does not store it)
        print("Loading Test Data...")
        _, _, test_loader = dataset.get_dataloaders(load_cached_data=True)

        test_preds = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = best_model(inputs)
                _, pred_indices = torch.max(outputs, 1)

                # Decode and Map
                pred_labels_fine = mapper.decode(pred_indices)
                pred_sub = [mapper.map_to_submission(l) for l in pred_labels_fine]
                test_preds.extend(pred_sub)

        # Create Submission DataFrame
        test_df = test_loader.dataset.df
        fnames = [os.path.basename(p) for p in test_df["filepath"]]

        submission_df = pd.DataFrame({"fname": fnames, "label": test_preds})

        sub_dir = "./submission"
        os.makedirs(sub_dir, exist_ok=True)
        save_path = os.path.join(sub_dir, "submission.csv")
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"Validation metric {final_acc} did not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    run()
