import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, targets, image_ids in dataloader:
        # Move images to device
        images = list(image.to(device) for image in images)

        # Move targets to device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)

        # Aggregate losses
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0
    print(f"Epoch [{epoch}] Train Loss: {avg_loss:.6f}")
    return avg_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Returns the average validation loss.
    """
    # Set model to train mode to retrieve loss dict, but disable gradients
    model.train()
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, targets, image_ids in dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            running_loss += losses.item()
            num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0
    print(f"Validation Loss: {avg_loss:.6f}")
    return avg_loss


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    patience,
    save_path,
):
    """
    Main training loop with Early Stopping.
    """
    best_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {num_epochs} epochs on {device}...")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = evaluate(model, val_loader, device)

        if scheduler:
            scheduler.step()

        # Checkpoint and Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with loss {best_loss:.6f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")


def predict(model, test_loader, test_df, device, submission_path):
    """
    Runs inference on the test set and generates the submission file.

    Args:
        model: The trained model.
        test_loader: DataLoader for the test set.
        test_df: DataFrame containing test metadata (to map image_id to study_id).
        device: Torch device.
        submission_path: Path to save the CSV.
    """
    model.eval()

    results_image = []
    results_study = {}  # Map study_id to list of probability vectors

    # Map image_id to study_id for aggregation
    # Ensure IDs are strings
    image_to_study = dict(
        zip(test_df["image_id"].astype(str), test_df["study_id"].astype(str))
    )

    # Mapping from full class names to submission short names
    short_name_map = {
        "Negative for Pneumonia": "negative",
        "Typical Appearance": "typical",
        "Indeterminate Appearance": "indeterminate",
        "Atypical Appearance": "atypical",
    }

    print("Starting inference...")

    with torch.no_grad():
        for images, targets, image_ids in test_loader:
            images = list(image.to(device) for image in images)

            # Forward pass: returns (detections, study_probs)
            detections, study_probs = model(images)

            # Process batch
            for i, img_id in enumerate(image_ids):
                # --- 1. Image Level Prediction ---
                det = detections[i]
                boxes = det["boxes"].cpu().numpy()
                scores = det["scores"].cpu().numpy()

                prediction_strings = []

                # Filter by threshold
                valid_indices = scores >= Config.CONF_THRESHOLD
                valid_boxes = boxes[valid_indices]
                valid_scores = scores[valid_indices]

                if len(valid_boxes) > 0:
                    for box, score in zip(valid_boxes, valid_scores):
                        # Format: opacity conf xmin ymin xmax ymax
                        pred_str = f"opacity {score:.4f} {box[0]:.1f} {box[1]:.1f} {box[2]:.1f} {box[3]:.1f}"
                        prediction_strings.append(pred_str)
                    final_image_str = " ".join(prediction_strings)
                else:
                    # No findings
                    final_image_str = "none 1 0 0 1 1"

                results_image.append(
                    {"Id": f"{img_id}_image", "PredictionString": final_image_str}
                )

                # --- 2. Study Level Accumulation ---
                study_id = image_to_study.get(img_id)
                if study_id:
                    probs = study_probs[i].cpu().numpy()
                    if study_id not in results_study:
                        results_study[study_id] = []
                    results_study[study_id].append(probs)

    # --- Aggregate Study Predictions ---
    study_rows = []
    for study_id, prob_list in results_study.items():
        # Average probabilities across images in the study
        avg_probs = np.mean(prob_list, axis=0)
        best_class_idx = np.argmax(avg_probs)
        confidence = avg_probs[best_class_idx]

        # Get label name and map to short format
        label_name = Config.STUDY_ID_TO_LABEL[best_class_idx]
        short_label = short_name_map.get(label_name, "negative")

        # Format: label conf 0 0 1 1
        pred_str = f"{short_label} {confidence:.4f} 0 0 1 1"

        study_rows.append({"Id": f"{study_id}_study", "PredictionString": pred_str})

    # --- Combine and Save ---
    df_image = pd.DataFrame(results_image)
    df_study = pd.DataFrame(study_rows)

    # Concatenate study and image predictions
    submission_df = pd.concat([df_study, df_image], ignore_index=True)

    # Save to CSV
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
