import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import cv2
from scipy import ndimage
from library import config, utils, data_processing


class SuperpixelClassifier:
    """
    A wrapper around LightGBM for superpixel-based segmentation.
    """

    def __init__(self):
        self.params = config.LGBM_PARAMS
        self.model = None
        # Features must match those generated in data_processing.py
        self.feature_cols = [
            "mean",
            "std",
            "mean_prev",
            "mean_next",
            "cent_y",
            "cent_x",
        ]

    def fit(self, train_df, val_df):
        """
        Trains the LightGBM model with early stopping.
        """
        X_train = train_df[self.feature_cols]
        y_train = train_df["label"]
        X_val = val_df[self.feature_cols]
        y_val = val_df["label"]

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        # Callbacks for logging and early stopping
        callbacks = [
            lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=50),
        ]

        print("Starting LightGBM training...")
        self.model = lgb.train(
            self.params,
            train_set,
            num_boost_round=config.N_ESTIMATORS,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        if self.model.best_score:
            score = self.model.best_score["valid"]["multi_logloss"]
            print(f"Training finished. Best validation multi_logloss: {score:.10f}")

    def predict(self, df):
        """
        Predicts class labels for the given superpixel features.
        Returns:
            np.ndarray: Array of predicted class indices.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        X = df[self.feature_cols]
        # predict returns probabilities (N_samples, N_classes)
        probs = self.model.predict(X)
        # Return class index with highest probability
        return np.argmax(probs, axis=1)

    def save(self, path):
        """Saves the trained model to disk."""
        joblib.dump(self.model, path)

    def load(self, path):
        """Loads a trained model from disk."""
        self.model = joblib.load(path)


def train_model(train_df, val_df):
    """
    Helper function to instantiate and train the classifier.
    """
    clf = SuperpixelClassifier()
    clf.fit(train_df, val_df)
    return clf


def _process_volume_inference(group_df, model):
    """
    Processes a single Case/Day volume for inference.
    1. Loads images.
    2. Generates superpixels and features.
    3. Predicts classes.
    4. Reconstructs 3D volume.
    5. Applies 3D Connected Component Analysis.
    6. Generates RLEs.
    """
    # Ensure sorted by slice index for correct 3D context
    group_df = group_df.sort_values("slice")
    slices = group_df["slice"].values
    ids = group_df["id"].values

    # Load all images for this volume to handle context
    imgs = {}
    # We need to track original dimensions for resizing back later
    orig_dims = {}

    for _, row in group_df.iterrows():
        try:
            # load_image resizes to config.IMG_SIZE
            imgs[row["slice"]] = utils.load_image(row["file_path"])
            orig_dims[row["slice"]] = (row["img_height"], row["img_width"])
        except Exception:
            pass

    if not imgs:
        return []

    # Initialize volume prediction container
    # Shape: (Depth, Height, Width)
    D = len(slices)
    H, W = config.IMG_SIZE
    vol_preds = np.zeros((D, H, W), dtype=np.uint8)

    # Map slice number to 0-based index in the volume array
    slice_to_idx = {s: i for i, s in enumerate(slices)}

    # --- Prediction Phase ---
    for s in slices:
        if s not in imgs:
            continue

        img_curr = imgs[s]
        # Handle boundary context by replicating edge slices
        img_prev = imgs.get(s - 1, img_curr)
        img_next = imgs.get(s + 1, img_curr)

        # Generate superpixels
        segments = data_processing.get_superpixels(img_curr)

        # Extract features
        feats, u_labels = data_processing.extract_features_single(
            img_curr, img_prev, img_next, segments
        )

        if feats.empty:
            continue

        # Predict
        preds = model.predict(feats)

        # Reconstruct mask from superpixels
        # Create a blank mask
        mask = np.zeros((H, W), dtype=np.uint8)

        # Assign predicted label to all pixels in the corresponding superpixel
        # Optimized assignment using a lookup array
        # u_labels are the segment IDs present in 'segments'
        # preds are the classes for those segments
        # We assume segment IDs are continuous enough or we use a mapping
        max_seg_id = segments.max()
        lookup = np.zeros(max_seg_id + 1, dtype=np.uint8)
        lookup[u_labels] = preds

        mask = lookup[segments]

        vol_preds[slice_to_idx[s]] = mask

    # --- Post-Processing Phase (3D) ---
    final_results = []

    # Iterate over organ classes (skipping background=0)
    for cls_idx, cls_name in enumerate(config.CLASSES):
        if cls_idx == 0:
            continue

        # Extract binary volume for this class
        cls_vol = vol_preds == cls_idx

        # Apply 3D Connected Component Analysis
        # This removes small floating artifacts and keeps the main organ structure
        labeled_vol, num_features = ndimage.label(cls_vol)

        if num_features > 1:
            # Find the largest component by volume
            component_sizes = ndimage.sum(
                cls_vol, labeled_vol, range(1, num_features + 1)
            )
            largest_label = np.argmax(component_sizes) + 1
            cls_vol = labeled_vol == largest_label

        # --- RLE Encoding Phase ---
        for i, s in enumerate(slices):
            # Get original dimensions
            orig_h, orig_w = orig_dims.get(s, config.IMG_SIZE)

            # Extract slice mask
            mask_slice = cls_vol[i].astype(np.uint8)

            # Resize back to original resolution if necessary
            if (H, W) != (orig_h, orig_w):
                # Use Nearest Neighbor to preserve binary 0/1 values
                mask_slice = cv2.resize(
                    mask_slice, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                )

            # Encode
            rle = utils.rle_encode(mask_slice)

            final_results.append({"id": ids[i], "class": cls_name, "predicted": rle})

    return final_results


def generate_submission(model):
    """
    Generates the submission file by running inference on the test set.
    """
    print("Generating submission...")

    # Load test metadata
    test_meta = utils.load_metadata("test")

    # Group by Case and Day to process as volumes
    groups = test_meta.groupby(["case", "day"])

    all_predictions = []

    for (case, day), group_df in groups:
        # Process each volume
        vol_preds = _process_volume_inference(group_df, model)
        all_predictions.extend(vol_preds)

    # Convert to DataFrame
    submission_df = pd.DataFrame(all_predictions)

    # Ensure correct column order
    submission_df = submission_df[["id", "class", "predicted"]]

    # Save
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
