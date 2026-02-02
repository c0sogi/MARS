import torch
import torch.optim as optim
import pandas as pd
import os
import sys
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, get_image_prediction_string
from library.dataset import get_dataloader
from library.model import SwinDyHeadNet
from library.engine import Engine


def generate_submission(model, device):
    print("Generating submission...")
    model.eval()
    test_loader = get_dataloader(
        split="test", batch_size=Config.BATCH_SIZE, shuffle=False
    )

    results = []

    short_name_map = {
        "Negative for Pneumonia": "negative",
        "Typical Appearance": "typical",
        "Indeterminate Appearance": "indeterminate",
        "Atypical Appearance": "atypical",
    }

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            outputs = model(images)

            cls_logits = outputs["cls_logits"]
            bbox_preds = outputs["bbox_preds"]
            anchors = outputs["anchors"]
            study_logits = outputs["study_logits"]

            batch_size = images.size(0)

            for i in range(batch_size):
                # Study
                study_probs = torch.softmax(study_logits[i], dim=0)
                study_label_idx = torch.argmax(study_probs).item()
                study_conf = study_probs[study_label_idx].item()
                study_name = Config.STUDY_ID_MAP[study_label_idx]
                study_short = short_name_map[study_name]
                study_str = f"{study_short} {study_conf:.6f} 0 0 1 1"

                # Image
                scores = cls_logits[i].sigmoid().squeeze(-1)
                mask = scores > Config.CONF_THRESHOLD

                if mask.any():
                    valid_scores = scores[mask]
                    valid_bbox_preds = bbox_preds[i][mask]
                    valid_anchors = anchors[mask]

                    # Decode
                    cx, cy, stride = (
                        valid_anchors[:, 0],
                        valid_anchors[:, 1],
                        valid_anchors[:, 2],
                    )
                    l, t, r, b = (
                        valid_bbox_preds[:, 0],
                        valid_bbox_preds[:, 1],
                        valid_bbox_preds[:, 2],
                        valid_bbox_preds[:, 3],
                    )
                    x1 = cx - l * stride
                    y1 = cy - t * stride
                    x2 = cx + r * stride
                    y2 = cy + b * stride

                    # Clip
                    h, w = images.shape[2], images.shape[3]
                    x1 = x1.clamp(0, w)
                    y1 = y1.clamp(0, h)
                    x2 = x2.clamp(0, w)
                    y2 = y2.clamp(0, h)

                    decoded_boxes = torch.stack([x1, y1, x2, y2], dim=1)

                    # Rescale to original size
                    orig_h, orig_w = targets[i]["orig_size"]
                    orig_h = float(orig_h)
                    orig_w = float(orig_w)

                    if orig_h > orig_w:
                        scale = Config.IMG_SIZE / orig_h
                        new_w = orig_w * scale
                        pad_w = (Config.IMG_SIZE - new_w) / 2
                        pad_h = 0.0
                    else:
                        scale = Config.IMG_SIZE / orig_w
                        new_h = orig_h * scale
                        pad_w = 0.0
                        pad_h = (Config.IMG_SIZE - new_h) / 2

                    decoded_boxes[:, 0] = (decoded_boxes[:, 0] - pad_w) / scale
                    decoded_boxes[:, 1] = (decoded_boxes[:, 1] - pad_h) / scale
                    decoded_boxes[:, 2] = (decoded_boxes[:, 2] - pad_w) / scale
                    decoded_boxes[:, 3] = (decoded_boxes[:, 3] - pad_h) / scale

                    image_str = get_image_prediction_string(
                        decoded_boxes.cpu().numpy(), valid_scores.cpu().numpy()
                    )
                else:
                    image_str = "none 1 0 0 1 1"

                results.append(
                    {
                        "Id": targets[i]["study_id"] + "_study",
                        "PredictionString": study_str,
                    }
                )
                results.append(
                    {
                        "Id": targets[i]["image_id"] + "_image",
                        "PredictionString": image_str,
                    }
                )

    df = pd.DataFrame(results)
    df = df.drop_duplicates(subset=["Id"])
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    seed_everything(Config.SEED)

    # Configure for Full Run
    Config.DEBUG = False

    device = Config.DEVICE
    print(f"Using device: {device}")

    # Data
    print("Loading Data...")
    train_loader = get_dataloader(
        split="train", batch_size=Config.BATCH_SIZE, shuffle=True
    )
    val_loader = get_dataloader(
        split="val", batch_size=Config.BATCH_SIZE, shuffle=False
    )

    # Model
    print("Initializing Model...")
    model = SwinDyHeadNet().to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # Engine
    engine = Engine(model, optimizer, device, scheduler)

    # Train
    print("Starting Training...")
    engine.fit(train_loader, val_loader)

    # Load Best Model
    best_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path))
        print("Loaded best model.")
    else:
        print("Warning: Best model not found, using current weights.")

    # Generate Submission
    generate_submission(model, device)


if __name__ == "__main__":
    main()
