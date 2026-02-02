import os
import cv2
import numpy as np
import pandas as pd
import torch

from library.config import Config
from library.model import DnCNN
from library.utils import save_submission_file


class InferenceEngine:
    """
    Handles the inference process for the DnCNN model on the test dataset.
    """

    def __init__(self):
        """
        Initializes the model and loads the trained weights.
        """
        self.device = Config.DEVICE
        self.model_path = Config.MODEL_SAVE_PATH

        # Initialize Model Architecture
        self.model = DnCNN(
            depth=Config.DEPTH,
            n_channels=Config.N_CHANNELS,
            image_channels=Config.IN_CHANNELS,
        )

        # Load Weights
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Please train the model first."
            )

        self.model.to(self.device)
        self.model.eval()

    def run(self):
        """
        Generates predictions for the test set and saves the submission file.
        """
        if not os.path.exists(Config.TEST_METADATA_PATH):
            raise FileNotFoundError(
                f"Test metadata not found at {Config.TEST_METADATA_PATH}"
            )

        df_test = pd.read_csv(Config.TEST_METADATA_PATH)
        predictions = {}

        print(
            f"Starting inference on {len(df_test)} test images using {self.device}..."
        )

        with torch.no_grad():
            for _, row in df_test.iterrows():
                img_id = row["image_id"]
                input_path = os.path.join(Config.INPUT_DIR, row["input_path"])

                # Load full image as Grayscale
                # We do not patch test images; DnCNN is fully convolutional and can handle variable sizes.
                img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Could not read image {input_path}. Skipping.")
                    continue

                # Normalize to [0, 1]
                img_norm = img.astype(np.float32) / 255.0

                # Prepare Input Tensor: (1, 1, H, W)
                # Batch size of 1 is used because test images may have different dimensions
                input_tensor = (
                    torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(self.device)
                )

                # Forward Pass: Predict Noise Residual
                pred_noise = self.model(input_tensor)

                # Denoise: Clean Prediction = Noisy Input - Predicted Noise
                pred_clean = input_tensor - pred_noise

                # Post-processing
                # Clip values to valid range [0, 1]
                pred_clean = torch.clamp(pred_clean, 0, 1)

                # Convert back to numpy array (H, W)
                pred_clean_np = pred_clean.squeeze().cpu().numpy()

                predictions[img_id] = pred_clean_np

        # Save results using the provided utility function
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        save_submission_file(predictions, Config.SUBMISSION_PATH)
        print("Inference complete.")
