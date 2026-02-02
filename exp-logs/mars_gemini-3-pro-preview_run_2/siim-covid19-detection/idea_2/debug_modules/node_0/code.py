import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.dataset import SIIMDataset
from library.model import MultiTaskFasterRCNN
from library.engine import train_model
from library.utils import seed_everything, get_device, collate_fn


def run_demo():
    print("=== Starting Library Verification and Demo Script ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for rapid execution...")

    # Override Config defaults to run a fast debug session
    Config.DEBUG = True
    Config.MAX_TRAIN_SAMPLES = 10  # Only use 10 images for training
    Config.MAX_VAL_SAMPLES = 6  # Only use 6 images for validation
    Config.IMG_SIZE = 320  # Small image size for speed
    Config.BATCH_SIZE = 2
    Config.NUM_EPOCHS = 1
    Config.WORKING_DIR = "./working/demo"
    Config.SUBMISSION_DIR = "./working/demo/submission"

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seeds
    seed_everything(Config.SEED)
    device = get_device()
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Dataset Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Dataset Logic...")

    # Instantiate Datasets
    train_ds = SIIMDataset(split="train", debug=True)
    val_ds = SIIMDataset(split="val", debug=True)

    # Verify lengths
    print(f"    Train Dataset Length: {len(train_ds)}")
    print(f"    Val Dataset Length: {len(val_ds)}")

    assert (
        len(train_ds) == Config.MAX_TRAIN_SAMPLES
    ), f"Expected {Config.MAX_TRAIN_SAMPLES} training samples, got {len(train_ds)}"
    assert (
        len(val_ds) == Config.MAX_VAL_SAMPLES
    ), f"Expected {Config.MAX_VAL_SAMPLES} validation samples, got {len(val_ds)}"

    # Create DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    # Fetch one batch
    images, targets = next(iter(train_loader))

    # Verify Batch Structure
    print("    Verifying batch structure...")
    assert len(images) == Config.BATCH_SIZE
    assert len(targets) == Config.BATCH_SIZE

    # Verify Image Tensor
    # Shape should be (C, H, W) -> (3, 320, 320)
    img_shape = images[0].shape
    assert img_shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {img_shape}"

    # Verify Target Keys
    required_keys = {"boxes", "labels", "study_label", "area", "iscrowd"}
    assert required_keys.issubset(
        targets[0].keys()
    ), f"Missing keys in target. Found: {targets[0].keys()}"

    print("    Dataset verification passed.")

    # ---------------------------------------------------------
    # 3. Model Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = MultiTaskFasterRCNN()
    model.to(device)

    # Move batch to device
    images_dev = list(img.to(device) for img in images)
    targets_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]

    # A. Training Forward Pass
    print("    Running Training Forward Pass...")
    model.train()
    loss_dict = model(images_dev, targets_dev)

    # Check if we got losses
    assert isinstance(loss_dict, dict)
    assert "loss_classifier" in loss_dict
    assert "loss_box_reg" in loss_dict
    assert "loss_objectness" in loss_dict
    assert "loss_rpn_box_reg" in loss_dict
    assert "loss_global_classifier" in loss_dict

    total_loss = sum(loss for loss in loss_dict.values())
    print(f"    Total Loss: {total_loss.item():.4f}")

    # B. Inference Forward Pass
    print("    Running Inference Forward Pass...")
    model.eval()
    with torch.no_grad():
        detections, global_logits = model(images_dev)

    # Check outputs
    assert len(detections) == Config.BATCH_SIZE
    assert global_logits.shape == (Config.BATCH_SIZE, Config.NUM_STUDY_CLASSES)
    assert "boxes" in detections[0]
    assert "scores" in detections[0]
    assert "labels" in detections[0]

    print("    Model verification passed.")

    # ---------------------------------------------------------
    # 4. Training Loop Verification
    # ---------------------------------------------------------
    print("\n[4] Verifying Training Engine...")

    # Setup DataLoaders
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # Setup Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Run Training
    print(f"    Training for {Config.NUM_EPOCHS} epoch(s)...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,  # Skip scheduler for demo
        device=device,
        num_epochs=Config.NUM_EPOCHS,
        patience=1,
    )

    # Verify Checkpoint creation
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created!"
    print(f"    Checkpoint confirmed at {checkpoint_path}")

    # ---------------------------------------------------------
    # 5. Inference Logic & Submission Format Verification
    # ---------------------------------------------------------
    print("\n[5] Verifying Inference Output Format...")

    # We will simulate the inference logic on the validation batch
    # to ensure the string formatting is correct.

    model.eval()
    results = []

    with torch.no_grad():
        # Use the validation loader as a proxy for test data
        for i, (imgs, tgts) in enumerate(val_loader):
            imgs = list(img.to(device) for img in imgs)

            dets, logits = model(imgs)

            for j in range(len(imgs)):
                # Mock IDs
                image_id = f"demo_img_{i}_{j}"
                study_id = f"demo_study_{i}_{j}"

                # Global Head (Study)
                probs = torch.softmax(logits[j], dim=0).cpu().numpy()
                best_cls_idx = np.argmax(probs)

                # Detection Head (Image)
                boxes = dets[j]["boxes"].cpu().numpy()
                scores = dets[j]["scores"].cpu().numpy()

                # Filter by threshold
                valid_mask = (
                    scores > 0.001
                )  # Low threshold to ensure we get some boxes for demo
                valid_boxes = boxes[valid_mask]
                valid_scores = scores[valid_mask]

                # Construct Prediction String
                if len(valid_boxes) == 0:
                    pred_str = "none 1 0 0 1 1"
                else:
                    parts = []
                    for b, s in zip(valid_boxes, valid_scores):
                        # Format: opacity conf xmin ymin xmax ymax
                        parts.append(
                            f"opacity {s:.4f} {b[0]:.1f} {b[1]:.1f} {b[2]:.1f} {b[3]:.1f}"
                        )
                    pred_str = " ".join(parts)

                results.append(
                    {"id": f"{image_id}_image", "PredictionString": pred_str}
                )

                # Verify Study String format logic
                study_classes = ["negative", "typical", "indeterminate", "atypical"]
                study_pred = (
                    f"{study_classes[best_cls_idx]} {probs[best_cls_idx]:.4f} 0 0 1 1"
                )
                results.append(
                    {"id": f"{study_id}_study", "PredictionString": study_pred}
                )

            # Just check one batch for the demo
            break

    # Convert to DataFrame
    df = pd.DataFrame(results)
    print("    Generated Sample Predictions:")
    print(df.head())

    # Assertions
    assert "id" in df.columns
    assert "PredictionString" in df.columns
    assert len(df) > 0

    # Check format of a study prediction
    study_row = df[df["id"].str.contains("_study")].iloc[0]
    parts = study_row["PredictionString"].split()
    assert parts[0] in [
        "negative",
        "typical",
        "indeterminate",
        "atypical",
    ], f"Invalid class in study prediction: {parts[0]}"
    assert parts[2:] == [
        "0",
        "0",
        "1",
        "1",
    ], f"Invalid bounding box in study prediction: {parts[2:]}"

    print("    Inference format verification passed.")
    print("\n=== All Tests Passed Successfully ===")


if __name__ == "__main__":
    run_demo()
