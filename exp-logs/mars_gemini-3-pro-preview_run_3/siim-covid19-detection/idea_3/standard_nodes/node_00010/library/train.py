import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library import config, utils, dataset, model, loss


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    running_seg_loss = 0.0
    running_class_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        seg_logits, class_logits = model(images)

        total_loss, metrics = criterion(seg_logits, class_logits, masks, labels)

        total_loss.backward()
        optimizer.step()

        # Accumulate losses weighted by batch size
        batch_size = images.size(0)
        running_loss += metrics["total_loss"] * batch_size
        running_seg_loss += metrics["seg_loss"] * batch_size
        running_class_loss += metrics["class_loss"] * batch_size

    dataset_size = len(loader.dataset)
    return {
        "loss": running_loss / dataset_size,
        "seg_loss": running_seg_loss / dataset_size,
        "class_loss": running_class_loss / dataset_size,
    }


def validate(model, loader, criterion, device):
    """
    Executes validation loop and calculates metrics.
    """
    model.eval()
    running_loss = 0.0
    running_seg_loss = 0.0
    running_class_loss = 0.0

    correct_class = 0
    total_dice = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["label"].to(device)

            seg_logits, class_logits = model(images)

            total_loss, metrics = criterion(seg_logits, class_logits, masks, labels)

            batch_size = images.size(0)
            running_loss += metrics["total_loss"] * batch_size
            running_seg_loss += metrics["seg_loss"] * batch_size
            running_class_loss += metrics["class_loss"] * batch_size

            # Classification Accuracy
            pred_classes = torch.argmax(class_logits, dim=1)
            true_classes = torch.argmax(labels, dim=1)
            correct_class += (pred_classes == true_classes).sum().item()

            # Dice Score (Approximate from DiceLoss which is 1 - Dice)
            total_dice += (1.0 - metrics["seg_loss"]) * batch_size

    dataset_size = len(loader.dataset)
    return {
        "loss": running_loss / dataset_size,
        "seg_loss": running_seg_loss / dataset_size,
        "class_loss": running_class_loss / dataset_size,
        "accuracy": correct_class / dataset_size,
        "dice": total_dice / dataset_size,
    }


def generate_submission(model, device, load_cached_data=True):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    model.eval()

    # Load Test Data
    if not os.path.exists(config.TEST_METADATA_PATH):
        print("Test metadata not found. Skipping submission generation.")
        return

    df_test = pd.read_csv(config.TEST_METADATA_PATH)
    test_dataset = dataset.SIIMDataset(
        df_test, split="test", load_cached_data=load_cached_data
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    study_preds = {}  # study_id -> list of prob arrays
    image_preds = {}  # image_id -> prediction string

    print("Generating predictions on test set...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            study_ids = batch["study_id"]
            image_ids = batch["image_id"]

            seg_logits, class_logits = model(images)

            # Process probabilities
            seg_probs = torch.sigmoid(seg_logits).cpu().numpy()
            class_probs = torch.softmax(class_logits, dim=1).cpu().numpy()

            for i in range(len(images)):
                s_id = study_ids[i]
                i_id = image_ids[i]
                c_prob = class_probs[i]
                mask = seg_probs[i, 0]

                # Store study probs
                if s_id not in study_preds:
                    study_preds[s_id] = []
                study_preds[s_id].append(c_prob)

                # Determine dominant class for consistency check
                # Index 0 is "Negative for Pneumonia"
                pred_idx = np.argmax(c_prob)

                if pred_idx == 0:
                    # If predicted negative, force no boxes
                    image_preds[i_id] = "none 1 0 0 1 1"
                else:
                    # Generate boxes from mask
                    boxes = utils.mask_to_boxes(mask, threshold=config.MASK_THRESHOLD)
                    image_preds[i_id] = utils.format_prediction_string(boxes)

    # Create Submission DataFrame
    submission_rows = []

    # Process Study Level
    for s_id, probs_list in study_preds.items():
        # Average probabilities if multiple images per study
        avg_probs = np.mean(probs_list, axis=0)
        pred_idx = np.argmax(avg_probs)
        conf = avg_probs[pred_idx]
        label = config.STUDY_LABELS[pred_idx]

        # Map full label to submission format: "Negative for Pneumonia" -> "negative"
        short_label = label.split(" ")[0].lower()

        submission_rows.append(
            {"id": f"{s_id}_study", "PredictionString": f"{short_label} {conf} 0 0 1 1"}
        )

    # Process Image Level
    for i_id, pred_str in image_preds.items():
        submission_rows.append({"id": f"{i_id}_image", "PredictionString": pred_str})

    df_sub = pd.DataFrame(submission_rows)
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")


def run_training(load_cached_data=True, epochs=config.EPOCHS, debug=False):
    """
    Main entry point for training and submission generation.
    """
    utils.seed_everything(config.SEED)

    # 1. Load Metadata
    if not os.path.exists(config.TRAIN_METADATA_PATH) or not os.path.exists(
        config.VAL_METADATA_PATH
    ):
        raise FileNotFoundError(
            "Metadata files not found. Ensure metadata generation was successful."
        )

    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(config.VAL_METADATA_PATH)

    if debug:
        print("Debug mode: Using reduced dataset.")
        df_train = df_train.head(50)
        df_val = df_val.head(50)

    # 2. Initialize Datasets & Loaders
    print("Initializing datasets...")
    train_dataset = dataset.SIIMDataset(
        df_train, split="train", load_cached_data=load_cached_data
    )
    val_dataset = dataset.SIIMDataset(
        df_val, split="val", load_cached_data=load_cached_data
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize Model, Loss, Optimizer
    print("Initializing model...")
    net = model.MultiTaskUNet(pretrained=True)
    net = net.to(config.DEVICE)

    criterion = loss.MultiTaskLoss(seg_weight=1.0, class_weight=1.0)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on {config.DEVICE}...")

    for epoch in range(epochs):
        # Train
        train_metrics = train_one_epoch(
            net, train_loader, criterion, optimizer, config.DEVICE
        )

        # Validate
        val_metrics = validate(net, val_loader, criterion, config.DEVICE)

        # Logging
        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_metrics['loss']}")
        print(f"Train Seg Loss: {train_metrics['seg_loss']}")
        print(f"Train Class Loss: {train_metrics['class_loss']}")
        print(f"Val Loss: {val_metrics['loss']}")
        print(f"Val Seg Loss: {val_metrics['seg_loss']}")
        print(f"Val Class Loss: {val_metrics['class_loss']}")
        print(f"Val Accuracy: {val_metrics['accuracy']}")
        print(f"Val Dice: {val_metrics['dice']}")

        # Checkpointing & Early Stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(net.state_dict(), config.CHECKPOINT_PATH)
            print("New best model saved.")
        else:
            patience_counter += 1
            print(
                f"EarlyStopping counter: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete.")

    # 5. Generate Submission
    print("Loading best model for inference...")
    net.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE))
    generate_submission(net, config.DEVICE, load_cached_data=load_cached_data)
