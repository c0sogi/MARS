import os
import time
import torch
import numpy as np
import pandas as pd
import cv2
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import calc_f1_score, get_affine_transform, affine_transform
from library.loss import CenterNetLoss
from library.model import HRNetCenterNet, transpose_and_gather_feat
from library.dataset import KuzushijiDataset


def load_val_gt_info(metadata_path, load_cached_data=True):
    """
    Loads validation ground truths and image sizes.
    Uses caching to avoid repeated I/O overhead.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, "val_gt_cache.npy")

    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True).item()
            return data
        except Exception:
            pass  # Fallback to processing if load fails

    print("Processing validation metadata for evaluation...")
    df = pd.read_csv(metadata_path)
    char2id, _ = Config.get_class_mappings()

    data = {}

    for _, row in df.iterrows():
        img_id = row["image_id"]
        file_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, file_path)

        # Read image dimensions
        if os.path.exists(full_path):
            img = cv2.imread(full_path)
            if img is not None:
                h, w = img.shape[:2]
            else:
                h, w = Config.INPUT_SIZE, Config.INPUT_SIZE
        else:
            h, w = Config.INPUT_SIZE, Config.INPUT_SIZE

        # Parse labels
        gts = []
        if pd.notna(row["labels"]) and isinstance(row["labels"], str):
            parts = row["labels"].strip().split(" ")
            if len(parts) > 1:
                for i in range(0, len(parts), 5):
                    try:
                        u_char = parts[i]
                        x = int(parts[i + 1])
                        y = int(parts[i + 2])
                        bw = int(parts[i + 3])
                        bh = int(parts[i + 4])

                        if u_char in char2id:
                            gts.append(
                                {"box": (x, y, bw, bh), "label": char2id[u_char]}
                            )
                    except (ValueError, IndexError):
                        continue

        data[img_id] = {"size": (h, w), "gts": gts}

    np.save(cache_path, data)
    return data


def decode_outputs(hm, wh, reg, K=1200):
    """
    Decodes model outputs into bounding boxes/points.
    """
    batch, cat, height, width = hm.size()

    # 1. MaxPool NMS
    pad = (3 - 1) // 2
    hmax = F.max_pool2d(hm, (3, 3), stride=1, padding=pad)
    keep = (hmax == hm).float()
    hm = hm * keep

    # 2. Top K
    hm = hm.view(batch, -1)
    scores, inds = torch.topk(hm, K)

    clses = inds // (height * width)
    inds = inds % (height * width)

    ys = inds // width
    xs = inds % width

    # 3. Add Regression Offsets
    reg = transpose_and_gather_feat(reg, inds)
    reg = reg.cpu()

    xs = xs.cpu().float()
    ys = ys.cpu().float()

    xs = xs + reg[..., 0]
    ys = ys + reg[..., 1]

    # 4. Scale to Input Size (stride 4)
    xs = xs * 4.0
    ys = ys * 4.0

    return scores.cpu().numpy(), clses.cpu().numpy(), xs.numpy(), ys.numpy()


def train_one_epoch(model, optimizer, data_loader, device, criterion, epoch):
    model.train()
    loss_meter = 0.0

    for i, batch in enumerate(data_loader):
        img = batch["image"].to(device)

        # Forward
        outputs = model(img)
        loss, _, _, _ = criterion(outputs, batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
        optimizer.step()

        loss_meter += loss.item()

    return loss_meter / len(data_loader)


def evaluate(model, data_loader, device, criterion, val_gt_data):
    model.eval()
    loss_meter = 0.0

    all_preds = []
    all_gts = []

    with torch.no_grad():
        for batch in data_loader:
            img = batch["image"].to(device)
            img_ids = batch["image_id"]

            # Forward
            outputs = model(img)
            loss, _, _, _ = criterion(outputs, batch)
            loss_meter += loss.item()

            # Decode
            hm, wh, reg = outputs
            scores, clses, xs, ys = decode_outputs(hm, wh, reg, K=Config.MAX_PREDS)

            # Map back to original image space for F1 calculation
            for b in range(len(img_ids)):
                img_id = img_ids[b]
                if img_id not in val_gt_data:
                    continue

                orig_h, orig_w = val_gt_data[img_id]["size"]
                gts = val_gt_data[img_id]["gts"]

                # Get Inverse Transform
                trans_inv = get_affine_transform(
                    (orig_h, orig_w), Config.INPUT_SIZE, inverse=True
                )

                img_preds = []

                # Filter by confidence
                valid_mask = scores[b] > Config.CONF_THRESHOLD
                v_scores = scores[b][valid_mask]
                v_clses = clses[b][valid_mask]
                v_xs = xs[b][valid_mask]
                v_ys = ys[b][valid_mask]

                for k in range(len(v_scores)):
                    # Transform point
                    pt = affine_transform([v_xs[k], v_ys[k]], trans_inv)

                    # Clip to image
                    px = min(max(0, pt[0]), orig_w - 1)
                    py = min(max(0, pt[1]), orig_h - 1)

                    img_preds.append(
                        {
                            "point": (px, py),
                            "label": v_clses[k],
                            "score": float(v_scores[k]),
                        }
                    )

                all_preds.append(img_preds)
                all_gts.append(gts)

    f1 = calc_f1_score(all_preds, all_gts)
    return loss_meter / len(data_loader), f1


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    patience=5,
):
    criterion = CenterNetLoss()
    val_gt_data = load_val_gt_info(Config.VAL_METADATA_PATH)

    best_f1 = 0.0
    epochs_no_improve = 0
    save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {num_epochs} epochs on {device}...")

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, criterion, epoch
        )

        # Evaluate
        val_loss, val_f1 = evaluate(model, val_loader, device, criterion, val_gt_data)

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        duration = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Time: {duration:.1f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val F1: {val_f1:.10f}"
        )

        # Checkpointing & Early Stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved! (F1: {best_f1:.10f})")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement. Patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation F1: {best_f1:.10f}")


def generate_submission(model, device):
    """
    Generates submission file for the test set.
    """
    print("Generating submission...")
    test_dataset = KuzushijiDataset(split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    _, id2char = Config.get_class_mappings()
    results = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            img_ids = batch["image_id"]

            hm, wh, reg = model(imgs)
            scores, clses, xs, ys = decode_outputs(hm, wh, reg, K=Config.MAX_PREDS)

            for b in range(len(img_ids)):
                img_id = img_ids[b]

                # Load original image size for inverse transform
                # Assuming test images are in input/test_images/
                path = os.path.join(Config.INPUT_DIR, "test_images", f"{img_id}.jpg")
                if not os.path.exists(path):
                    # Try to find it if extension is different or path issue
                    # This is a fallback, usually path is predictable
                    path = os.path.join(
                        Config.INPUT_DIR, "test_images", f"{img_id}.jpg"
                    )

                if os.path.exists(path):
                    orig_img = cv2.imread(path)
                    if orig_img is not None:
                        oh, ow = orig_img.shape[:2]
                    else:
                        oh, ow = Config.INPUT_SIZE, Config.INPUT_SIZE
                else:
                    oh, ow = Config.INPUT_SIZE, Config.INPUT_SIZE

                trans_inv = get_affine_transform(
                    (oh, ow), Config.INPUT_SIZE, inverse=True
                )

                label_strs = []

                valid_mask = scores[b] > Config.CONF_THRESHOLD
                v_scores = scores[b][valid_mask]
                v_clses = clses[b][valid_mask]
                v_xs = xs[b][valid_mask]
                v_ys = ys[b][valid_mask]

                for k in range(len(v_scores)):
                    pt = affine_transform([v_xs[k], v_ys[k]], trans_inv)
                    px = int(min(max(0, pt[0]), ow - 1))
                    py = int(min(max(0, pt[1]), oh - 1))

                    char_code = id2char[v_clses[k]]
                    label_strs.append(f"{char_code} {px} {py}")

                results.append({"image_id": img_id, "labels": " ".join(label_strs)})

    sub_df = pd.DataFrame(results)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def train_pipeline(debug=False):
    """
    Main entry point to setup and run training.
    """
    Config.setup()
    Config.seed_everything(Config.SEED)

    # Datasets
    debug_size = 100 if debug else None
    train_ds = KuzushijiDataset(split="train", debug_size=debug_size)
    val_ds = KuzushijiDataset(split="val", debug_size=debug_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    model = HRNetCenterNet().to(Config.DEVICE)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
    )

    # Run Fit
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        num_epochs=Config.NUM_EPOCHS,
    )

    # Generate Submission
    # Load best model first
    best_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=Config.DEVICE))
        generate_submission(model, Config.DEVICE)
    else:
        print("No best model found to generate submission.")
