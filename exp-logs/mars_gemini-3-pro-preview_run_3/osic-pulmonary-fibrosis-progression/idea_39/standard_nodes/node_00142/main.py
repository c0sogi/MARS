import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, metric_score
from library.data import get_data, LungDataset
from library.model import CI_OP_DS_Net
from library.train import run_training


def main():
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)

    # Ensure submission directory exists
    # We use the path defined in Config, but also ensure the local ./submission exists
    # as per the prompt's specific save instruction.
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 2. Training
    # We run for Config.EPOCHS to match the scheduler horizon.
    # Cite Lesson 00100.
    print("Starting training pipeline...")
    run_training(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=False)

    # 3. Inference Setup
    device = torch.device(Config.DEVICE)
    model = CI_OP_DS_Net().to(device)

    # Load the best checkpoint saved during training
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Reload data to access validation set and fitted processor
    # We use debug=False to get the full validation set
    _, val_ds, _, processor = get_data(debug=False)

    # scaler_target is needed to inverse transform predictions
    scaler_target = processor.scalers["target_fvc"]
    std_fvc = scaler_target.scale_[0]
    mean_fvc = scaler_target.mean_[0]

    # 4. Validation & Metric Calculation
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    all_mu = []
    all_sigma = []
    all_true = []
    all_metas = []

    print("Running validation inference...")
    with torch.no_grad():
        for img, meta_a, meta_b, target in val_loader:
            img = img.to(device)
            meta_a_gpu = meta_a.to(device)
            meta_b_gpu = meta_b.to(device)

            # Predict
            mu_scaled, sigma_scaled = model(img, meta_a_gpu, meta_b_gpu)

            # Inverse Transform
            # mu = mu_scaled * std + mean
            # sigma = sigma_scaled * std (scale only)
            mu = mu_scaled.cpu().numpy() * std_fvc + mean_fvc
            sigma = sigma_scaled.cpu().numpy() * std_fvc

            # Target is also scaled in the dataset, so we unscale it
            target_unscaled = target.numpy().flatten() * std_fvc + mean_fvc

            all_mu.append(mu)
            all_sigma.append(sigma)
            all_true.append(target_unscaled)
            all_metas.append(meta_a.numpy())

    y_pred_mu = np.concatenate(all_mu)
    y_pred_sigma = np.concatenate(all_sigma)
    y_true = np.concatenate(all_true)
    meta_matrix = np.concatenate(all_metas, axis=0)

    # Compute Metric
    # Note: metric_score handles the clipping internally (sigma >= 70, delta <= 1000)
    final_metric = metric_score(y_true, y_pred_mu, y_pred_sigma)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    errors = np.abs(y_true - y_pred_mu)

    # meta_a structure from library.data.LungDataset:
    # [age_s, sex_c, smoke_c, rel_weeks_s, base_fvc_s]
    feature_names = ["Age", "Sex", "SmokingStatus", "RelWeeks", "BaseFVC"]

    print("Correlation between Absolute Error and Input Features:")
    for i, name in enumerate(feature_names):
        feat_vals = meta_matrix[:, i]
        # Compute Pearson correlation
        if np.std(feat_vals) > 0 and np.std(errors) > 0:
            corr = np.corrcoef(errors, feat_vals)[0, 1]
        else:
            corr = 0.0
        print(f"  {name}: {corr:.4f}")

    # 6. Submission Generation
    THRESHOLD = -6.573619738753321

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric:.6f}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Load sample submission to get the required Patient_Week combinations
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)

        # Parse Patient and Weeks from "Patient_Week" (e.g., "ID000..._12")
        # We use rsplit to handle potential underscores in IDs safely, though IDs are standard.
        sample_sub["Patient"] = sample_sub["Patient_Week"].apply(
            lambda x: x.rsplit("_", 1)[0]
        )
        sample_sub["Weeks"] = sample_sub["Patient_Week"].apply(
            lambda x: int(x.rsplit("_", 1)[1])
        )

        # Load test metadata to get static features (Age, Sex, Smoking, image_path)
        # metadata/test.csv contains one row per patient with baseline info
        test_meta = pd.read_csv(Config.TEST_CSV)

        # Merge static info into the submission dataframe
        # We need: Age, Sex, SmokingStatus, image_path
        sub_df = sample_sub.merge(
            test_meta[["Patient", "Age", "Sex", "SmokingStatus", "image_path"]],
            on="Patient",
            how="left",
        )

        # Transform features using the fitted processor
        # This handles scaling and creates 'RelWeeks', 'BaseFVC' etc.
        sub_df_transformed = processor.transform(sub_df)

        # Create Dataset for the submission set
        # mode='test' ensures we don't look for target columns
        sub_ds = LungDataset(sub_df_transformed, mode="test")
        sub_loader = DataLoader(
            sub_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        sub_mu = []
        sub_sigma = []

        print(f"Predicting for {len(sub_ds)} samples...")
        with torch.no_grad():
            for img, meta_a, meta_b, _ in sub_loader:
                img = img.to(device)
                meta_a = meta_a.to(device)
                meta_b = meta_b.to(device)

                mu_scaled, sigma_scaled = model(img, meta_a, meta_b)

                # Inverse transform
                mu = mu_scaled.cpu().numpy() * std_fvc + mean_fvc
                sigma = sigma_scaled.cpu().numpy() * std_fvc

                sub_mu.append(mu)
                sub_sigma.append(sigma)

        final_mu = np.concatenate(sub_mu)
        final_sigma = np.concatenate(sub_sigma)

        # Post-processing for submission
        # 1. Clip sigma at 70ml (Task requirement)
        final_sigma = np.maximum(final_sigma, 70)

        # 2. Assign to dataframe
        sample_sub["FVC"] = final_mu
        sample_sub["Confidence"] = final_sigma

        # 3. Save
        # Save to the Config-defined directory
        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        sample_sub[["Patient_Week", "FVC", "Confidence"]].to_csv(
            submission_path, index=False
        )

        # Also save to ./submission/submission.csv as explicitly requested in the prompt
        alt_submission_path = "./submission/submission.csv"
        sample_sub[["Patient_Week", "FVC", "Confidence"]].to_csv(
            alt_submission_path, index=False
        )

        print(f"Submission saved to {submission_path} and {alt_submission_path}")

    else:
        print(
            f"\nMetric ({final_metric:.6f}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
