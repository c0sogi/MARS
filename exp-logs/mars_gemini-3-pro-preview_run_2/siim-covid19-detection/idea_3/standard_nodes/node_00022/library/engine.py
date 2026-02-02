import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library import utils
from library.dataset import CovidDataset, get_transforms
from library.model import CovidMultiTaskModel


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_meter = AverageMeter()
    start_time = time.time()

    for step, (images, targets, image_ids) in enumerate(data_loader):
        # Move images and targets to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass (returns dict of losses in train mode)
        loss_dict = model(images, targets)

        # Sum all losses (RPN + ROI + Study)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Logging
        loss_val = losses.item()
        loss_meter.update(loss_val, len(images))

        if step % 10 == 0:
            # Simple print to track progress without progress bar
            sys.stdout.write(
                f"\rEpoch: [{epoch}] Step: [{step}/{len(data_loader)}] Loss: {loss_val:.5f} Avg: {loss_meter.avg:.5f}"
            )
            sys.stdout.flush()

    print(
        f"\nEpoch [{epoch}] Complete. Avg Loss: {loss_meter.avg:.10f}. Time: {time.time() - start_time:.2f}s"
    )
    return loss_meter.avg


def evaluate_loss(model, data_loader, device):
    # To get validation loss, we must use model.train() mode
    # but wrap in no_grad to disable gradient tracking.
    # Standard FasterRCNN in eval() mode returns detections, not losses.
    model.train()
    loss_meter = AverageMeter()

    print("Evaluating validation loss...")
    with torch.no_grad():
        for step, (images, targets, image_ids) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            loss_meter.update(losses.item(), len(images))

    print(f"Validation Loss: {loss_meter.avg:.10f}")
    return loss_meter.avg


def inference(model, device):
    model.eval()

    # 1. Load Test Data
    print("Loading test data for inference...")
    test_dataset = CovidDataset(
        subset="test",
        transforms=get_transforms("val"),  # No augmentation for test
        load_cached_data=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=utils.collate_fn,
    )

    # Storage for aggregation
    # study_id -> {'probs': list of [p0, p1, p2, p3], 'box_scores': list of {cls: max_score}}
    study_data = {}

    # Storage for image predictions
    # image_id -> list of box strings
    image_predictions = {}

    # Map image_id to study_id for aggregation
    test_df = test_dataset.df
    image_to_study = dict(zip(test_df.image_id, test_df.StudyInstanceUID))

    print("Generating predictions...")

    with torch.no_grad():
        for images, targets, image_ids in test_loader:
            images = list(image.to(device) for image in images)

            outputs = model(images)

            for i, output in enumerate(outputs):
                img_id = image_ids[i]

                # 1. Study Prediction (Global Head)
                # output['study_prediction'] is (4,) tensor of probabilities
                study_probs = output["study_prediction"].cpu().numpy()

                # 2. Box Scores (Detection Head)
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                # Calculate max box score per class for this image to help study classification
                # Classes: 1=Typical, 2=Indeterminate, 3=Atypical
                img_box_max_scores = {1: 0.0, 2: 0.0, 3: 0.0}

                valid_box_strings = []

                for b, s, l in zip(boxes, scores, labels):
                    # Track max score for study ensemble
                    if l in img_box_max_scores:
                        img_box_max_scores[l] = max(img_box_max_scores[l], s)

                    # Format for image prediction
                    # "opacity confidence xmin ymin xmax ymax"
                    if s > Config.BOX_SCORE_THRESH:
                        valid_box_strings.append(
                            f"opacity {s:.4f} {b[0]:.1f} {b[1]:.1f} {b[2]:.1f} {b[3]:.1f}"
                        )

                # Default to "none" if no boxes found
                image_predictions[img_id] = (
                    " ".join(valid_box_strings)
                    if valid_box_strings
                    else "none 1 0 0 1 1"
                )

                # Store for Study Aggregation
                study_id = image_to_study.get(img_id, "unknown")
                if study_id not in study_data:
                    study_data[study_id] = {"probs": [], "box_scores": []}

                study_data[study_id]["probs"].append(study_probs)
                study_data[study_id]["box_scores"].append(img_box_max_scores)

    # ---------------------------------------------------------------------
    # Post-Processing & Submission Generation
    # ---------------------------------------------------------------------

    submission_rows = []

    # 1. Process Studies
    for study_id, data in study_data.items():
        if study_id == "unknown":
            continue

        # Average global probabilities across images in study
        avg_probs = np.mean(data["probs"], axis=0)  # (4,)

        # Max box scores across images in study
        max_box_scores = {1: 0.0, 2: 0.0, 3: 0.0}
        for img_scores in data["box_scores"]:
            for cls in [1, 2, 3]:
                max_box_scores[cls] = max(max_box_scores[cls], img_scores[cls])

        # Weighted Ensemble Score
        final_scores = np.zeros(4)
        # Class 0 (Negative): Purely global head
        final_scores[0] = avg_probs[0]
        # Classes 1-3: Average of global head and max box score
        for cls in [1, 2, 3]:
            final_scores[cls] = (avg_probs[cls] + max_box_scores[cls]) / 2.0

        # Determine Label
        pred_idx = np.argmax(final_scores)
        pred_label = Config.STUDY_ID_TO_LABEL[pred_idx]
        pred_conf = final_scores[pred_idx]

        # Format: "class_name confidence 0 0 1 1"
        pred_string = f"{pred_label} {pred_conf:.4f} 0 0 1 1"

        submission_rows.append(
            {"id": f"{study_id}_study", "PredictionString": pred_string}
        )

        # 2. Consistency Check for Images
        # If study is Negative, force all images in that study to "none"
        if pred_idx == 0:
            study_images = test_df[test_df.StudyInstanceUID == study_id][
                "image_id"
            ].values
            for img_id in study_images:
                image_predictions[img_id] = "none 1 0 0 1 1"

    # 3. Add Image Rows
    for img_id, pred_str in image_predictions.items():
        submission_rows.append({"id": f"{img_id}_image", "PredictionString": pred_str})

    # Create DataFrame and Save
    submission_df = pd.DataFrame(submission_rows)
    print(f"Saving submission to {Config.SUBMISSION_PATH}")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)


def fit(debug=False):
    utils.seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Data Loaders
    print("Initializing DataLoaders...")
    train_dataset = CovidDataset(
        subset="train", transforms=get_transforms("train"), load_cached_data=True
    )
    val_dataset = CovidDataset(
        subset="val", transforms=get_transforms("val"), load_cached_data=True
    )

    if debug:
        print("Debug mode: Using subset of data.")
        indices = torch.arange(min(100, len(train_dataset)))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        val_dataset = torch.utils.data.Subset(val_dataset, indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=utils.collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=utils.collate_fn,
        pin_memory=True,
    )

    # 2. Model
    print("Initializing Model...")
    model = CovidMultiTaskModel()
    model.to(device)

    # 3. Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.LR_DECAY_STEP, gamma=Config.LR_GAMMA
    )

    # 4. Training Loop
    best_loss = float("inf")

    print("Starting Training...")
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{Config.NUM_EPOCHS} ---")

        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch + 1)
        val_loss = evaluate_loss(model, val_loader, device)

        lr_scheduler.step()

        # Checkpointing
        if val_loss < best_loss:
            print(
                f"Validation loss improved from {best_loss:.10f} to {val_loss:.10f}. Saving model..."
            )
            best_loss = val_loss
            torch.save(
                model.state_dict(), os.path.join(Config.WORKING_DIR, "best_model.pth")
            )

        # Save latest checkpoint
        torch.save(
            model.state_dict(), os.path.join(Config.WORKING_DIR, "checkpoint.pth")
        )

    print("Training Complete.")

    # 5. Inference
    print("Running Inference on Test Set...")
    # Load best model
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model not found, using last checkpoint.")

    inference(model, device)
