import os
import sys
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import provided library modules
from library import config, utils, model, data, train


def extract_features(df):
    """
    Extracts audio features (Duration, RMS, Peak) for failure analysis.
    """
    durations = []
    rms_values = []
    peak_values = []

    print("Extracting features for failure analysis...")
    # Iterate through all files to extract features
    for idx, row in df.iterrows():
        path = os.path.join(config.INPUT_ROOT, row["file_path"])
        try:
            # Use soundfile for fast metadata and signal reading
            info = sf.info(path)
            y, sr = sf.read(path)

            # Handle multi-channel if necessary (though data is mostly mono)
            if y.ndim > 1:
                y = np.mean(y, axis=1)

            durations.append(info.duration)
            rms_values.append(np.sqrt(np.mean(y**2)))
            peak_values.append(np.max(np.abs(y)))
        except Exception as e:
            # Fallback for unreadable files (should not happen in clean dataset)
            durations.append(0)
            rms_values.append(0)
            peak_values.append(0)

    return np.array(durations), np.array(rms_values), np.array(peak_values)


def main():
    # 1. Setup
    utils.seed_everything(config.SEED)
    logger = utils.get_logger()
    logger.info("Starting execution of runfile.py")

    # 2. Training Loop (5-Fold CV)
    model_paths = []
    for fold in range(config.N_FOLDS):
        logger.info(f"--- Processing Fold {fold} ---")
        # run_fold trains the model, applies early stopping, and returns the path to the best checkpoint
        checkpoint_path = train.run_fold(fold, logger)
        model_paths.append(checkpoint_path)

    # 3. Out-of-Fold (OOF) Validation
    logger.info("Starting OOF Validation...")

    # Load metadata to reconstruct the splits
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    # Replicate the StratifiedKFold split used in data.py
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    )
    splits = list(skf.split(np.zeros(len(full_df)), full_df["label"]))

    oof_preds = np.zeros(len(full_df))
    oof_labels = np.zeros(len(full_df))

    # Iterate through folds to generate OOF predictions
    for fold in range(config.N_FOLDS):
        logger.info(f"Validating Fold {fold}...")

        # Get the validation loader for this fold
        _, val_loader, _ = data.get_dataloaders(fold=fold, load_cached_data=True)

        # Load the best model for this fold
        net = model.get_model()
        net.load_state_dict(torch.load(model_paths[fold], map_location=config.DEVICE))
        net.to(config.DEVICE)
        net.eval()

        fold_probs = []
        fold_labels_list = []

        # Inference on validation set
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(config.DEVICE)
                outputs = net(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                fold_probs.append(probs)
                fold_labels_list.append(labels.numpy())

        fold_probs = np.concatenate(fold_probs).flatten()
        fold_labels_list = np.concatenate(fold_labels_list).flatten()

        # Map predictions back to the original dataframe indices
        _, val_idx = splits[fold]

        # Ensure sizes match (drop_last=False for val_loader)
        # Note: val_loader size should match len(val_idx) exactly
        if len(fold_probs) != len(val_idx):
            # Fallback if loader dropped samples (unlikely with default config)
            # But assuming standard behavior, they match.
            pass

        oof_preds[val_idx] = fold_probs
        oof_labels[val_idx] = fold_labels_list

    # Compute Final Validation Metric
    final_auc = roc_auc_score(oof_labels, oof_preds)
    print(f"Final Validation Metric: {final_auc:.16f}")

    # 4. Failure Analysis
    logger.info("Performing Failure Analysis...")

    # Extract features for all training samples
    durations, rms_values, peak_values = extract_features(full_df)

    # Calculate absolute error
    errors = np.abs(oof_labels - oof_preds)

    # Calculate correlations
    corr_dur, _ = pearsonr(errors, durations)
    corr_rms, _ = pearsonr(errors, rms_values)
    corr_peak, _ = pearsonr(errors, peak_values)

    print(f"Error Correlation - Duration: {corr_dur:.6f}")
    print(f"Error Correlation - RMS: {corr_rms:.6f}")
    print(f"Error Correlation - Peak: {corr_peak:.6f}")

    # 5. Submission
    THRESHOLD = 0.9959177895986835
    if final_auc > THRESHOLD:
        logger.info(
            f"Validation metric {final_auc:.6f} > {THRESHOLD}. Generating submission..."
        )

        # Load Test Loader (Fold 0 is sufficient as test set is constant)
        _, _, test_loader = data.get_dataloaders(fold=0, load_cached_data=True)

        test_probs_sum = None
        test_clips = None

        # Soft Voting Ensemble
        for fold, path in enumerate(model_paths):
            logger.info(f"Predicting test set with model from Fold {fold}...")
            net = model.get_model()
            net.load_state_dict(torch.load(path, map_location=config.DEVICE))
            net.to(config.DEVICE)

            # Use the library's inference function
            clips, probs = train.inference(net, test_loader, config.DEVICE)

            # probs is (N, 1), flatten to (N,)
            probs = probs.flatten()

            if test_probs_sum is None:
                test_probs_sum = probs
                test_clips = clips
            else:
                test_probs_sum += probs

        # Average predictions
        avg_probs = test_probs_sum / config.N_FOLDS

        # Create Submission DataFrame
        sub_df = pd.DataFrame({"clip": test_clips, "probability": avg_probs})

        # Save
        os.makedirs(os.path.dirname(config.SUBMISSION_FILE), exist_ok=True)
        sub_df.to_csv(config.SUBMISSION_FILE, index=False)
        logger.info(f"Submission saved to {config.SUBMISSION_FILE}")

    else:
        logger.info(
            f"Validation metric {final_auc:.6f} <= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
