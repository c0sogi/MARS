import os
import random
import numpy as np
import pandas as pd
import torch
from library.losses import CombinedLoss
from library.utils import do_kaggle_metric, unpad_image, rle_encode


def set_seed(seed=42):
    """Sets fixed random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, device, epoch, max_batches=None):
    """
    Trains the model for one epoch.
    """
    model.train()
    loss_func = CombinedLoss()
    total_loss = 0.0
    count = 0

    for i, (images, masks, depths, ids) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        optimizer.zero_grad()
        logits = model(images, depths)
        loss = loss_func(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        count += images.size(0)

    avg_loss = total_loss / count if count > 0 else 0.0
    print(f"Epoch {epoch} Training Loss: {avg_loss:.6f}")
    return avg_loss


def evaluate(model, loader, device, threshold=0.5, max_batches=None):
    """
    Evaluates the model on the validation set and returns the mAP score.
    """
    model.eval()
    predictions = []
    truths = []

    with torch.no_grad():
        for i, (images, masks, depths, ids) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            images = images.to(device)
            # Use provided depths for validation
            depths = depths.to(device)

            logits = model(images, depths)
            probs = torch.sigmoid(logits)
            probs_np = probs.cpu().numpy()
            masks_np = masks.numpy()

            for j in range(len(probs_np)):
                # Unpad back to 101x101
                p = unpad_image(probs_np[j, 0], original_size=(101, 101))
                t = unpad_image(masks_np[j, 0], original_size=(101, 101))
                predictions.append(p)
                truths.append(t)

    predictions = np.array(predictions)
    truths = np.array(truths)

    score = do_kaggle_metric(predictions, truths, threshold=threshold)
    print(f"Validation mAP (threshold={threshold}): {score}")
    return score


def optimize_threshold(model, loader, device, max_batches=None):
    """
    Sweeps thresholds on the validation set to find the one that maximizes mAP.
    """
    model.eval()
    predictions = []
    truths = []

    # print("Optimizing threshold on validation set...")
    with torch.no_grad():
        for i, (images, masks, depths, ids) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            images = images.to(device)
            depths = depths.to(device)

            logits = model(images, depths)
            probs = torch.sigmoid(logits)
            probs_np = probs.cpu().numpy()
            masks_np = masks.numpy()

            for j in range(len(probs_np)):
                p = unpad_image(probs_np[j, 0], original_size=(101, 101))
                t = unpad_image(masks_np[j, 0], original_size=(101, 101))
                predictions.append(p)
                truths.append(t)

    predictions = np.array(predictions)
    truths = np.array(truths)

    # Sweep thresholds from 0.3 to 0.7
    thresholds = np.linspace(0.3, 0.7, 21)
    best_score = -1.0
    best_thresh = 0.5

    for t in thresholds:
        score = do_kaggle_metric(predictions, truths, threshold=t)
        if score > best_score:
            best_score = score
            best_thresh = t

    # print(f"Optimal Threshold: {best_thresh:.4f} with mAP: {best_score}")
    return best_thresh


def evaluate_multithreshold(model, loader, device, max_batches=None):
    """
    Evaluates model and returns the best mAP achieved across a range of thresholds.
    Cite {solution_lesson_node_00033}: Decouple prediction probability from decision boundaries during validation.
    """
    model.eval()
    predictions = []
    truths = []

    with torch.no_grad():
        for i, (images, masks, depths, ids) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            images = images.to(device)
            depths = depths.to(device)

            logits = model(images, depths)
            probs = torch.sigmoid(logits)
            probs_np = probs.cpu().numpy()
            masks_np = masks.numpy()

            for j in range(len(probs_np)):
                p = unpad_image(probs_np[j, 0], original_size=(101, 101))
                t = unpad_image(masks_np[j, 0], original_size=(101, 101))
                predictions.append(p)
                truths.append(t)

    predictions = np.array(predictions)
    truths = np.array(truths)

    thresholds = np.linspace(0.3, 0.7, 21)
    best_score = -1.0

    for t in thresholds:
        score = do_kaggle_metric(predictions, truths, threshold=t)
        if score > best_score:
            best_score = score

    return best_score


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs=50,
    patience=10,
    save_path="./working/best_model.pth",
    max_batches=None,
):
    """
    Orchestrates the training loop with Early Stopping.
    """
    set_seed(42)
    best_score = -1.0
    patience_counter = 0

    # Ensure working dir exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, max_batches=max_batches
        )
        # Monitor with default threshold 0.5
        val_score = evaluate(
            model, val_loader, device, threshold=0.5, max_batches=max_batches
        )

        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with mAP: {best_score}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best mAP: {best_score}")
    # Load best model weights
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    return model


def predict_test(
    model, loader, device, threshold=0.5, output_path="./submission/submission.csv"
):
    """
    Generates predictions for the test set using TTA and saves to CSV.
    Forces depth to 0 during inference.
    """
    model.eval()
    ids_list = []
    rles_list = []

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Generating predictions for test set...")
    with torch.no_grad():
        for images, _, _, ids in loader:
            images = images.to(device)
            # Force depth to 0 for test time (Strategy: Robust Self-Training)
            depths = torch.zeros(
                (images.size(0), 1), device=device, dtype=torch.float32
            )

            # TTA: Original
            logits = model(images, depths)
            probs = torch.sigmoid(logits)

            # TTA: Horizontal Flip
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip, depths)
            probs_flip = torch.sigmoid(logits_flip)
            probs_flip = torch.flip(probs_flip, dims=[3])

            # Average
            avg_probs = (probs + probs_flip) / 2.0
            avg_probs_np = avg_probs.cpu().numpy()

            for i in range(len(avg_probs_np)):
                img_id = ids[i]
                # Unpad back to 101x101
                prob_map = unpad_image(avg_probs_np[i, 0], original_size=(101, 101))

                # Binarize
                mask = (prob_map > threshold).astype(np.uint8)

                # RLE Encode
                rle = rle_encode(mask)
                ids_list.append(img_id)
                rles_list.append(rle)

    df = pd.DataFrame({"id": ids_list, "rle_mask": rles_list})
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
