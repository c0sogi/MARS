import os
import cv2
import math
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("features")

# Global variables for worker processes
_GLOBAL_BASIS = None
_GLOBAL_INDICES = None


def _factorial(n):
    """Compute factorial of n."""
    return math.factorial(n)


def _zernike_radial(n, m, rho):
    """
    Compute the radial polynomial R_nm(rho).

    Args:
        n (int): Order.
        m (int): Repetition.
        rho (numpy.ndarray): Radial distance grid (0 <= rho <= 1).

    Returns:
        numpy.ndarray: The radial polynomial values.
    """
    R = np.zeros_like(rho)

    # The sum has (n - |m|) / 2 + 1 terms
    n_minus_m_div_2 = (n - m) // 2

    for k in range(n_minus_m_div_2 + 1):
        # Calculate numerator and denominator for the coefficient
        # (-1)^k * (n-k)!
        # ---------------------------------------
        # k! * ((n+m)/2 - k)! * ((n-m)/2 - k)!

        numerator = ((-1) ** k) * _factorial(n - k)
        denominator = (
            _factorial(k) * _factorial((n + m) // 2 - k) * _factorial((n - m) // 2 - k)
        )

        coeff = numerator / denominator
        R += coeff * (rho ** (n - 2 * k))

    return R


def _init_worker(order, img_size):
    """
    Initializer for worker processes to precompute Zernike basis functions.
    This avoids recomputing or pickling the basis for every image.
    """
    global _GLOBAL_BASIS, _GLOBAL_INDICES

    width, height = img_size

    # 1. Create coordinate grid mapped to [-1, 1]
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xv, yv = np.meshgrid(x, y)

    # 2. Convert to polar coordinates
    rho = np.sqrt(xv**2 + yv**2)
    theta = np.arctan2(yv, xv)

    # 3. Create mask for unit disk
    mask = rho <= 1.0

    # 4. Generate list of valid (n, m) pairs
    indices = []
    for n in range(order + 1):
        for m in range(n + 1):
            # m must be such that n-m is even
            if (n - m) % 2 == 0:
                indices.append((n, m))

    _GLOBAL_INDICES = indices

    # 5. Precompute Basis Functions V_nm* = R_nm(rho) * exp(-j * m * theta)
    # We store them as a list or array.
    # Since we need dot product sum(Img * V*), we compute V* directly.
    basis_list = []

    for n, m in indices:
        # Radial component
        R = _zernike_radial(n, m, rho)

        # Angular component (conjugate implies -j)
        # V_nm = R * exp(j * m * theta)
        # V_nm_conj = R * exp(-j * m * theta)
        angular = np.exp(-1j * m * theta)

        basis_func = R * angular

        # Apply mask (zero out pixels outside unit disk)
        basis_func[~mask] = 0

        basis_list.append(basis_func)

    _GLOBAL_BASIS = basis_list


def _process_single_image(image_path_rel, img_size):
    """
    Worker function to process a single image.

    Args:
        image_path_rel (str): Relative path to image.
        img_size (tuple): Target size (W, H).

    Returns:
        list: Calculated Zernike moment magnitudes.
    """
    global _GLOBAL_BASIS, _GLOBAL_INDICES

    full_path = os.path.join(Config.INPUT_DIR, image_path_rel)

    # 1. Load Image
    if not os.path.exists(full_path):
        # Return zeros if file missing (should not happen based on EDA)
        return [0.0] * len(_GLOBAL_INDICES)

    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [0.0] * len(_GLOBAL_INDICES)

    # 2. Preprocessing
    # The dataset description says: "binary black leaves against white backgrounds".
    # Invert so leaf is white (1/255) and background is black (0).
    img = cv2.bitwise_not(img)

    # Threshold to ensure binary
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find moments for Centering
    moments = cv2.moments(img)
    if moments["m00"] == 0:
        # Empty image
        return [0.0] * len(_GLOBAL_INDICES)

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    # Find Max Radius for Scaling
    # Get all non-zero points
    points = cv2.findNonZero(img)
    if points is None:
        return [0.0] * len(_GLOBAL_INDICES)

    points = points.reshape(-1, 2)  # (N, 2) -> (x, y)

    # Calculate distances from centroid
    dists = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
    max_dist = np.max(dists)

    if max_dist == 0:
        max_dist = 1.0  # Avoid div by zero

    # Desired radius in the target image (mapped to unit disk [-1, 1])
    # We map max_dist to 0.95 * (img_size/2) to fit comfortably
    target_radius = 0.95 * (img_size[0] / 2)
    scale = target_radius / max_dist

    # Construct Affine Matrix for Centering + Scaling
    # Shift (-cx, -cy) -> Scale (s) -> Shift (+W/2, +H/2)
    # M = [ s, 0, -s*cx + W/2 ]
    #     [ 0, s, -s*cy + H/2 ]

    tx = -scale * cx + img_size[0] / 2
    ty = -scale * cy + img_size[1] / 2

    M = np.float32([[scale, 0, tx], [0, scale, ty]])

    # Warp
    img_processed = cv2.warpAffine(img, M, img_size, flags=cv2.INTER_LINEAR)

    # Normalize pixel values to [0, 1]
    img_normalized = img_processed.astype(np.float64) / 255.0

    # 3. Compute Zernike Moments
    # A_nm = (n+1)/pi * sum(Img * V_nm*)
    # We use the precomputed V_nm* in _GLOBAL_BASIS

    features = []

    for idx, (n, m) in enumerate(_GLOBAL_INDICES):
        basis_func_conj = _GLOBAL_BASIS[idx]

        # Discrete approximation of the integral
        # Sum over pixels
        z = np.sum(img_normalized * basis_func_conj)

        # Normalization factor (n+1)/pi
        # Note: Some implementations use different factors.
        # We stick to the standard definition for orthogonality on unit disk.
        # Since we mapped image to [-1, 1] grid, the area of pixel is dx*dy.
        # dx = 2/W, dy = 2/H. Area = 4/(W*H).
        # Integral ~ sum * Area.
        # However, for feature extraction, relative magnitude matters.
        # We can omit constant factors or keep them. Let's keep (n+1)/pi.

        z = z * (n + 1) / np.pi

        # We only use Magnitude for rotation invariance
        features.append(abs(z))

    return features


class ZernikeExtractor:
    """
    Engineers Zernike Moment features from images.
    """

    def __init__(self):
        self.order = Config.ZERNIKE_ORDER
        self.img_size = Config.IMG_LOAD_SIZE

    def extract_features(self, metadata_path, cache_path, load_cached=True):
        """
        Extracts features for the dataset provided in metadata_path.

        Args:
            metadata_path (str): Path to the metadata CSV (train/val/test).
            cache_path (str): Path to save/load parquet cache.
            load_cached (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: DataFrame containing 'id' and Zernike features.
        """
        # 1. Check Cache
        if load_cached and os.path.exists(cache_path):
            logger.info(f"Loading cached Zernike features from {cache_path}")
            return pd.read_parquet(cache_path)

        logger.info(
            f"Computing Zernike features (Order={self.order}) for {os.path.basename(metadata_path)}..."
        )

        # 2. Load Metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df_meta = pd.read_csv(metadata_path)
        image_paths = df_meta["image_path"].tolist()
        ids = df_meta["id"].tolist()

        # 3. Parallel Processing
        # We use a ProcessPoolExecutor to handle the heavy lifting
        # The _init_worker ensures basis is computed once per process

        results = []
        with ProcessPoolExecutor(
            max_workers=Config.N_JOBS,
            initializer=_init_worker,
            initargs=(self.order, self.img_size),
        ) as executor:

            # Map returns results in order
            futures = executor.map(
                _process_single_image, image_paths, [self.img_size] * len(image_paths)
            )

            results = list(futures)

        # 4. Construct DataFrame
        # Generate column names
        col_names = []
        # We need to regenerate indices locally to name columns correctly
        # Logic must match _init_worker
        for n in range(self.order + 1):
            for m in range(n + 1):
                if (n - m) % 2 == 0:
                    col_names.append(f"zernike_n{n}_m{m}")

        # Verify shape
        if len(results) > 0 and len(results[0]) != len(col_names):
            logger.warning(
                f"Feature count mismatch! Expected {len(col_names)}, got {len(results[0])}"
            )

        df_features = pd.DataFrame(results, columns=col_names)
        df_features.insert(0, "id", ids)

        # 5. Save Cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_features.to_parquet(cache_path, index=False)
        logger.info(f"Saved Zernike features to {cache_path}")

        return df_features


def run_extraction():
    """
    Main execution function to generate features for all splits.
    """
    extractor = ZernikeExtractor()

    # Train
    extractor.extract_features(
        metadata_path=Config.TRAIN_METADATA_PATH,
        cache_path=Config.CACHE_ZERNIKE_TRAIN,
        load_cached=True,
    )

    # Val
    extractor.extract_features(
        metadata_path=Config.VAL_METADATA_PATH,
        cache_path=Config.CACHE_ZERNIKE_VAL,
        load_cached=True,
    )

    # Test
    extractor.extract_features(
        metadata_path=Config.TEST_METADATA_PATH,
        cache_path=Config.CACHE_ZERNIKE_TEST,
        load_cached=True,
    )

    logger.info("Zernike feature extraction complete.")
