import os
import random
import numpy as np
import torch
import cv2
from PIL import Image
import pandas as pd
import io
import rasterio
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_image(path, size=None):
    """
    Robustly loads an image from the specified path.
    Tries multiple methods:
    1. OpenCV (Standard)
    2. PIL (Standard)
    3. Manual Extraction of JPEG/JP2 Codestream + OpenCV/PIL/Rasterio

    Returns:
        np.ndarray: Loaded image array (H, W) or (H, W, C).
    """
    if not os.path.exists(path):
        # If file is missing, we can't do anything.
        # But to prevent crash, we might want to return black?
        # Dataset.py handles FileNotFoundError specifically for contralateral.
        # For target, it expects it to exist.
        # Let's return black and log warning to be safe.
        print(f"Warning: File not found {path}. Returning black image.")
        s = size if size is not None else 512
        return np.zeros((s, s), dtype=np.uint8)

    img = None

    # 1. Try OpenCV (Standard)
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except Exception:
        pass

    # 2. Try PIL (Standard)
    if img is None:
        try:
            with Image.open(path) as pil_img:
                img = np.array(pil_img)
        except Exception:
            pass

    # 3. Manual DICOM Parsing (Fallback for JPEG 2000 / JPEG)
    if img is None:
        try:
            with open(path, "rb") as f:
                content = f.read()

            # Search for JPEG 2000 Codestream (FF 4F FF 51)
            # or JPEG (FF D8)
            jp2_start = content.find(b"\xff\x4f\xff\x51")
            jpeg_start = content.find(b"\xff\xd8")

            img_bytes = None
            if jp2_start != -1:
                img_bytes = content[jp2_start:]
            elif jpeg_start != -1:
                img_bytes = content[jpeg_start:]

            if img_bytes is not None:
                # 3a. Try OpenCV on bytes
                try:
                    img_array = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
                except Exception:
                    pass

                # 3b. Try PIL on bytes
                if img is None:
                    try:
                        with Image.open(io.BytesIO(img_bytes)) as pil_img:
                            img = np.array(pil_img)
                    except Exception:
                        pass

                # 3c. Try Rasterio on bytes (GDAL often has JP2 support)
                if img is None:
                    try:
                        with rasterio.MemoryFile(img_bytes) as memfile:
                            with memfile.open() as dataset:
                                img = dataset.read(1)  # Read first band
                    except Exception:
                        pass
        except Exception:
            pass

    # 4. Final Fallback
    if img is None:
        print(f"Warning: Failed to decode {path}. Returning black image.")
        s = size if size is not None else 512
        img = np.zeros((s, s), dtype=np.uint8)

    # 5. Post-processing (Resize and Channel check)
    # Ensure image is numpy array
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    # Handle dimensions (we want H, W)
    if img.ndim == 3:
        # If 3 channels, convert to grayscale if needed or keep?
        # Task is mammography (grayscale).
        # If RGB, convert to Gray.
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 1:
            img = img[:, :, 0]
        # If 4 channels (RGBA), drop alpha
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    # Resize if requested
    if size is not None:
        h, w = img.shape[:2]
        if h != size or w != size:
            try:
                img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
            except Exception:
                # If resize fails (e.g. empty image), return black
                img = np.zeros((size, size), dtype=np.uint8)

    return img


def get_cached_data(cache_path, processing_func, load_cached=True, **kwargs):
    """
    Manages data caching using Parquet (for DataFrames) or NPY (for arrays).

    Logic:
    1. If load_cached is True and cache file exists, load and return it.
    2. Otherwise, execute processing_func(**kwargs), save the result to cache_path, and return it.

    Args:
        cache_path (str): Destination path for the cache file.
        processing_func (callable): Function to generate data if cache is missed.
        load_cached (bool): Whether to attempt loading from cache.
        **kwargs: Arguments passed to processing_func.

    Returns:
        The processed data (pd.DataFrame or np.ndarray).
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Attempt to load from cache
    if load_cached and os.path.exists(cache_path):
        try:
            if cache_path.endswith(".parquet"):
                return pd.read_parquet(cache_path)
            elif cache_path.endswith(".npy"):
                return np.load(cache_path, allow_pickle=False)
            else:
                # If extension is unknown, we can't load safely, proceed to compute
                pass
        except Exception as e:
            # If loading fails (e.g. corruption), proceed to recompute
            print(
                f"Warning: Failed to load cache at {cache_path} ({e}). Recomputing..."
            )

    # Compute data
    data = processing_func(**kwargs)

    # Save to cache
    if cache_path.endswith(".parquet"):
        if isinstance(data, pd.DataFrame):
            data.to_parquet(cache_path)
        else:
            raise ValueError(
                f"Expected pandas DataFrame for .parquet cache, got {type(data)}."
            )
    elif cache_path.endswith(".npy"):
        if isinstance(data, np.ndarray):
            np.save(cache_path, data)
        else:
            raise ValueError(
                f"Expected numpy ndarray for .npy cache, got {type(data)}."
            )
    else:
        raise ValueError("Cache file extension must be .parquet or .npy")

    return data
