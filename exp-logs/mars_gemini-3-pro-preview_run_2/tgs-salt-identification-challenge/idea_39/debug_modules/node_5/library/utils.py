import os
import sys
import random
import logging
import numpy as np
import pandas as pd
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(log_dir):
    """
    Initializes and returns a logger that outputs to both file and stdout.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(Config.PROJECT_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid adding handlers multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

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


def rle_encode(img):
    """
    Encodes a binary mask to Run-Length Encoding (RLE) format.
    The pixels are 1-indexed and numbered from top to bottom, then left to right (Column-Major).

    Args:
        img (np.ndarray): Binary mask (0 or 1).
    Returns:
        str: RLE string.
    """
    # Flatten column-major
    pixels = img.T.flatten()
    # Pad with 0s to detect runs at start/end
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # runs[1::2] are ends, runs[::2] are starts
    # Lengths = ends - starts
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes an RLE string to a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Shape of the output mask (H, W).
    Returns:
        np.ndarray: Binary mask.
    """
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    # Create flattened array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape column-major
    return img.reshape(shape, order="F")


def do_kaggle_metric(predict, truth, threshold=0.5):
    """
    Calculates the Mean Average Precision at IoU thresholds (0.5 to 0.95, step 0.05).

    Args:
        predict (torch.Tensor or np.ndarray): Predicted masks (probabilities or binary).
        truth (torch.Tensor or np.ndarray): Ground truth masks (binary).
        threshold (float): Threshold to binarize predictions if they are probabilities.
    Returns:
        float: The mean average precision score.
    """
    # Convert to numpy
    if isinstance(predict, torch.Tensor):
        predict = predict.detach().cpu().numpy()
    if isinstance(truth, torch.Tensor):
        truth = truth.detach().cpu().numpy()

    # Binarize predictions
    if predict.dtype == np.float32 or predict.dtype == np.float64:
        predict = (predict > threshold).astype(np.uint8)
    else:
        predict = predict.astype(np.uint8)

    truth = truth.astype(np.uint8)

    ious = []
    # Calculate IoU for each image in batch
    for p, t in zip(predict, truth):
        p_flat = p.flatten()
        t_flat = t.flatten()

        intersection = np.sum(p_flat * t_flat)
        union = np.sum(p_flat) + np.sum(t_flat) - intersection

        if union == 0:
            # Both empty -> Perfect match
            iou = 1.0
        else:
            iou = intersection / union
        ious.append(iou)

    ious = np.array(ious)

    # Thresholds: 0.5, 0.55, ..., 0.95
    thresholds = np.arange(0.5, 1.0, 0.05)

    # Compare IoUs to thresholds
    # Shape: (Batch_Size, Num_Thresholds)
    hits = ious[:, None] > thresholds[None, :]

    # Average precision per image (mean over thresholds)
    ap_per_image = np.mean(hits, axis=1)

    # Mean Average Precision over dataset
    return np.mean(ap_per_image)


def load_or_process(file_name, process_fn, load_cached_data=True, **kwargs):
    """
    Loads data from cache if available, otherwise processes and saves it.

    Args:
        file_name (str): Name of the file (e.g., 'data.npy').
        process_fn (callable): Function to generate data if cache miss.
        load_cached_data (bool): Whether to attempt loading from cache.
        **kwargs: Arguments passed to process_fn.
    Returns:
        The loaded or processed data.
    """
    cache_path = os.path.join(Config.CACHE_DIR, file_name)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        try:
            if cache_path.endswith(".npy"):
                return np.load(cache_path, allow_pickle=True)
            elif cache_path.endswith(".parquet"):
                return pd.read_parquet(cache_path)
            elif cache_path.endswith(".csv"):
                return pd.read_csv(cache_path)
        except Exception as e:
            print(f"Error loading cache {cache_path}: {e}. Recomputing...")

    # Process data
    data = process_fn(**kwargs)

    # Save data
    if cache_path.endswith(".npy"):
        # Explicitly construct object array to avoid broadcasting errors (Cite debug_lesson_26)
        obj_arr = np.empty(len(data), dtype=object)
        obj_arr[:] = data
        np.save(cache_path, obj_arr)
    elif cache_path.endswith(".parquet") and isinstance(data, pd.DataFrame):
        data.to_parquet(cache_path)
    elif cache_path.endswith(".csv") and isinstance(data, pd.DataFrame):
        data.to_csv(cache_path, index=False)

    return data


def generate_submission(model, test_loader, device, threshold=0.5):
    """
    Generates submission file using Marginalized Depth Scan strategy.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): Loader for test data (yields images, ids).
        device (str): Device to run inference on.
        threshold (float): Threshold for binarization.
    """
    model.eval()

    ids_list = []
    rles_list = []

    scan_depths = Config.SCAN_DEPTHS
    if not scan_depths:
        scan_depths = [0.0]

    print(f"Generating submission with Depth Scan: {scan_depths}")

    with torch.no_grad():
        for batch in test_loader:
            # Handle different loader yields
            if len(batch) == 2:
                images, ids = batch
            else:
                # Fallback if loader yields more
                images, ids = batch[0], batch[1]

            images = images.to(device, dtype=torch.float32)
            batch_size = images.size(0)

            # Accumulate probabilities
            avg_probs = None

            for z_val in scan_depths:
                # Create depth input
                z_tensor = torch.full(
                    (batch_size, 1), z_val, device=device, dtype=torch.float32
                )

                # Forward pass
                # Expecting model(x, z)
                output = model(images, z_tensor)

                # Handle tuple output (e.g., logits, aux)
                if isinstance(output, (tuple, list)):
                    logits = output[0]
                else:
                    logits = output

                probs = torch.sigmoid(logits)

                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs += probs

            # Average
            avg_probs /= len(scan_depths)

            # Post-processing: Crop and Encode
            # Current shape: (B, 1, 128, 128) -> Target: (B, 1, 101, 101)
            h, w = avg_probs.shape[2], avg_probs.shape[3]
            th, tw = Config.ORIG_SIZE, Config.ORIG_SIZE

            if h != th or w != tw:
                start_h = (h - th) // 2
                start_w = (w - tw) // 2
                avg_probs = avg_probs[
                    :, :, start_h : start_h + th, start_w : start_w + tw
                ]

            # Binarize
            pred_masks = (avg_probs > threshold).cpu().numpy().astype(np.uint8)

            for i, img_id in enumerate(ids):
                rle = rle_encode(pred_masks[i, 0])
                ids_list.append(img_id)
                rles_list.append(rle)

    # Save submission
    sub_df = pd.DataFrame({"id": ids_list, "rle_mask": rles_list})
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
