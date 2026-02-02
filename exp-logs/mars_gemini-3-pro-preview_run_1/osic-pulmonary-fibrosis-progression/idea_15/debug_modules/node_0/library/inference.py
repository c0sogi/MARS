import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.dataset import LungDataset
from library.model import GranularTabularNetwork


def predict(
    model_path=Config.BEST_MODEL_PATH,
    output_path=Config.SUBMISSION_FILE,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    device=Config.DEVICE,
):
    """
    Loads the trained model and generates predictions for the test set.

    Args:
        model_path (str): Path to the trained model checkpoint.
        output_path (str): Path to save the submission CSV.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of worker threads for data loading.
        device (str): Device to run inference on ('cpu' or 'cuda').
    """
    print(f"Initializing inference on device: {device}")

    # 1. Load Test Dataset
    # The LungDataset in 'test' mode reads from metadata/test.csv
    # and prepares the necessary tensors including time_delta and priors.
    test_dataset = LungDataset(mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
    )

    print(f"Test dataset loaded. Samples: {len(test_dataset)}")

    # 2. Load Model
    model = GranularTabularNetwork()
    model.to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Model weights loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    # 3. Inference Loop
    results = []

    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            age = batch["age"].to(device)
            sex = batch["sex"].to(device)
            smoke = batch["smoke"].to(device)
            percent = batch["percent"].to(device)
            priors = batch["priors"].to(device)
            time_delta = batch["time_delta"].to(device)
            patient_weeks = batch["patient_week"]

            # Forward pass
            # The model returns the final FVC and Confidence calculated via:
            # FVC = Baseline + alpha * delta
            # Conf = sigma_base + sigma_growth * |delta|
            fvc_pred, conf_pred = model(
                axial, coronal, age, sex, smoke, percent, priors, time_delta
            )

            # Move to CPU and convert to numpy
            fvc_pred = fvc_pred.cpu().numpy()
            conf_pred = conf_pred.cpu().numpy()

            # Aggregate results
            for pw, fvc, conf in zip(patient_weeks, fvc_pred, conf_pred):
                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

    # 4. Generate Submission File
    submission_df = pd.DataFrame(results)

    # Ensure columns are in the correct order required by the competition
    submission_df = submission_df[["Patient_Week", "FVC", "Confidence"]]

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission generated and saved to {output_path}")
    print(submission_df.head())
