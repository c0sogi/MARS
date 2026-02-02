import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import laplace_log_likelihood_loss, seed_everything
from library.model import IASDANet


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Executes one training epoch.

    Args:
        model (nn.Module): The IAS-DAN model.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): PyTorch optimizer.
        device (str): Device to run on ('cuda' or 'cpu').
        epoch (int): Current epoch number.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch_idx, data in enumerate(dataloader):
        # Move data to device
        axial_img = data["axial_img"].to(device)
        coronal_img = data["coronal_img"].to(device)
        tabular = data["tabular"].to(device)
        time_delta = data["time_delta"].to(device)
        baseline_fvc = data["baseline_fvc"].to(device)
        target = data["target"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        fvc_pred, sigma_pred = model(
            axial_img=axial_img,
            coronal_img=coronal_img,
            tabular=tabular,
            time_delta=time_delta,
            baseline_fvc=baseline_fvc,
        )

        # Calculate Loss
        # We minimize the negative metric, so Loss = -Metric
        loss = laplace_log_likelihood_loss(
            y_true=target,
            y_pred=fvc_pred,
            sigma=sigma_pred,
            clip_sigma=Config.Q_SIGMA_THRESHOLD,
            clip_error=Config.ERROR_THRESHOLD,
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The IAS-DAN model.
        dataloader (DataLoader): Validation data loader.
        device (str): Device to run on.

    Returns:
        float: Validation score (Negative of the average loss).
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for data in dataloader:
            # Move data to device
            axial_img = data["axial_img"].to(device)
            coronal_img = data["coronal_img"].to(device)
            tabular = data["tabular"].to(device)
            time_delta = data["time_delta"].to(device)
            baseline_fvc = data["baseline_fvc"].to(device)
            target = data["target"].to(device)

            # Forward pass
            fvc_pred, sigma_pred = model(
                axial_img=axial_img,
                coronal_img=coronal_img,
                tabular=tabular,
                time_delta=time_delta,
                baseline_fvc=baseline_fvc,
            )

            # Calculate Loss
            loss = laplace_log_likelihood_loss(
                y_true=target,
                y_pred=fvc_pred,
                sigma=sigma_pred,
                clip_sigma=Config.Q_SIGMA_THRESHOLD,
                clip_error=Config.ERROR_THRESHOLD,
            )

            running_loss += loss.item()
            num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    # Metric is negative loss (Higher is better)
    score = -avg_loss
    return score


def train_model(train_loader, val_loader):
    """
    Orchestrates the training process with Early Stopping.

    Args:
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Initializing IAS-DAN Model on {device}...")
    model = IASDANet()
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # Early Stopping Variables
    best_score = -float("inf")
    patience_counter = 0
    best_model_path = Config.BEST_MODEL_PATH

    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    print("Starting Training Loop...")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_score = evaluate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Print Metrics
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Score: {val_score:.8f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.8f}"
        )

        # Early Stopping Logic
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New Best Model Saved! Score: {best_score:.8f}")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.PATIENCE}")

            if patience_counter >= Config.PATIENCE:
                print("Early Stopping Triggered.")
                break

    print(f"Training Complete. Best Validation Score: {best_score:.8f}")


def predict(test_loader):
    """
    Generates predictions for the test set and creates the submission file.

    Args:
        test_loader (DataLoader): Test data loader.
    """
    device = torch.device(Config.DEVICE)
    best_model_path = Config.BEST_MODEL_PATH

    print(f"Loading Best Model from {best_model_path}...")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Model file not found at {best_model_path}. Train model first."
        )

    model = IASDANet()
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    results = []

    print("Generating Predictions...")
    with torch.no_grad():
        for data in test_loader:
            # Move data to device
            axial_img = data["axial_img"].to(device)
            coronal_img = data["coronal_img"].to(device)
            tabular = data["tabular"].to(device)
            time_delta = data["time_delta"].to(device)
            baseline_fvc = data["baseline_fvc"].to(device)
            patient_ids = data["patient_id"]  # List of strings

            # Forward pass
            fvc_pred, sigma_pred = model(
                axial_img=axial_img,
                coronal_img=coronal_img,
                tabular=tabular,
                time_delta=time_delta,
                baseline_fvc=baseline_fvc,
            )

            # Post-process
            fvc_pred = fvc_pred.cpu().numpy().flatten()
            sigma_pred = sigma_pred.cpu().numpy().flatten()

            # Retrieve original time delta to reconstruct Week
            # Note: In test dataset, time_delta = Predict_Week - Baseline_Week
            # We need the actual Predict_Week for the submission ID.
            # However, the dataset logic for test mode doesn't explicitly pass 'Predict_Week'
            # as a tensor, but we can reconstruct the submission ID if we have the patient ID
            # and the week.
            # A safer way is to rely on the fact that test_loader iterates sequentially
            # through the test.csv which was built from sample_submission.csv.
            # But since we have batching, we need to be careful.

            # Let's look at how we can reconstruct the Patient_Week ID.
            # The test.csv in metadata has 'Patient', 'Predict_Week'.
            # We can't easily pass strings through the model forward, but we have them in the batch 'data' dictionary.
            # We just need to grab 'Predict_Week' from the dataframe logic or pass it in __getitem__.
            # The current Dataset __getitem__ returns 'time_delta'.
            # We can calculate Week = time_delta + Baseline_Week.
            # But Baseline_Week is not passed in the batch dict in the provided Dataset code?
            # Wait, looking at Dataset code provided in prompt:
            # It returns 'baseline_fvc', 'time_delta', 'patient_id'.
            # It does NOT return 'Predict_Week' or 'Baseline_Week' explicitly in the dict.
            # However, we can infer the Week if we had Baseline_Week.

            # ALTERNATIVE: The submission file format requires 'Patient_Week'.
            # The test dataset is built from test.csv which is built from sample_submission.
            # The order is preserved if shuffle=False.
            # We can just collect predictions and merge them with the metadata dataframe later.
            # But to be robust, let's assume we need to match rows.

            # Since I cannot modify dataset.py, I will rely on the order of the DataLoader
            # matching the order of Config.TEST_CSV.
            # I will collect all predictions into a list.

            for i in range(len(patient_ids)):
                results.append({"FVC": fvc_pred[i], "Confidence": sigma_pred[i]})

    # Convert results to DataFrame
    pred_df = pd.DataFrame(results)

    # Load the test metadata to get the Patient_Week identifiers
    # We assume the DataLoader was created with shuffle=False and sequentially matches this CSV.
    meta_test_df = pd.read_csv(Config.TEST_CSV)

    # Synchronize metadata slicing with dataset truncation in Debug mode (Cite debug_lesson_2)
    if Config.DEBUG:
        meta_test_df = meta_test_df.iloc[: len(pred_df)]

    if len(pred_df) != len(meta_test_df):
        raise ValueError(
            f"Prediction count ({len(pred_df)}) does not match Test Metadata count ({len(meta_test_df)})."
        )

    # Combine
    submission = pd.DataFrame()
    submission["Patient_Week"] = meta_test_df["Patient_Week"]
    submission["FVC"] = pred_df["FVC"]
    submission["Confidence"] = pred_df["Confidence"]

    # Apply Confidence Clipping (as per metric requirement, though loss handles it,
    # the submission file should also reflect reasonable values, usually clipped at 70?
    # The metric clips at 70. The prompt says "confidence values are clipped at 70 ml".
    # Usually this means in the metric calculation. But for submission,
    # it's good practice to ensure we don't submit extremely low confidence.)
    # The prompt says: "The metric is computed as: sigma_clipped = max(sigma, 70)".
    # It doesn't strictly say submission must be clipped, but it's safe to do so.
    submission["Confidence"] = submission["Confidence"].apply(lambda x: max(x, 70))

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission.head())
