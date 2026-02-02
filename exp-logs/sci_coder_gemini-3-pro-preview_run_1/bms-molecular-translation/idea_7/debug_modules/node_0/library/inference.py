import os
import torch
import pandas as pd
import numpy as np

from library.config import Config
from library.model import AMViT
from library.tokenizer import Tokenizer


class Predictor:
    """
    Inference class for the Attribute-Modulated Visual Transformer (AM-ViT).
    Handles model loading, greedy decoding, and submission file generation.
    """

    def __init__(self, model_path=None, device=None):
        """
        Args:
            model_path (str, optional): Path to the trained model weights.
                                        Defaults to Config.MODEL_PATH.
            device (torch.device, optional): Device to run inference on.
                                             Defaults to Config.DEVICE.
        """
        self.device = device if device else torch.device(Config.DEVICE)
        self.model_path = model_path if model_path else Config.MODEL_PATH

        # 1. Load Tokenizer
        # We expect the tokenizer to have been built and saved during the training phase.
        self.tokenizer = Tokenizer()
        if os.path.exists(Config.TOKENIZER_PATH):
            print(f"Loading tokenizer from {Config.TOKENIZER_PATH}")
            self.tokenizer.load(Config.TOKENIZER_PATH)
        else:
            print("Warning: Tokenizer artifact not found at configured path.")
            # Attempt to fit on metadata if available as a fallback
            if os.path.exists(Config.TRAIN_METADATA):
                print("Attempting to fit tokenizer on training metadata...")
                self.tokenizer.fit_on_texts(load_cached_data=True)
            else:
                raise FileNotFoundError(
                    "Cannot initialize tokenizer: No artifact or metadata found."
                )

        # Cache special token indices for decoding
        self.vocab_size = len(self.tokenizer)
        self.pad_idx = self.tokenizer.token2id[self.tokenizer.pad_token]
        self.sos_idx = self.tokenizer.token2id[self.tokenizer.sos_token]
        self.eos_idx = self.tokenizer.token2id[self.tokenizer.eos_token]

        # 2. Load Model
        self.model = AMViT(vocab_size=self.vocab_size, pad_idx=self.pad_idx)
        self._load_weights()
        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self):
        """Loads state dictionary into the model."""
        if os.path.exists(self.model_path):
            print(f"Loading model weights from {self.model_path}")
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(
                f"Warning: Model path {self.model_path} does not exist. Using random initialization."
            )

    def greedy_decode(self, images, max_len=None):
        """
        Performs greedy decoding for a batch of images.

        This method executes the inference pipeline:
        1. Encodes images to obtain spatial and global features.
        2. Predicts chemical attributes from global features.
        3. Autoregressively generates tokens using the Transformer decoder,
           conditioned on the predicted attributes via AdaLN.

        Args:
            images (torch.Tensor): Batch of input images, shape (B, 3, H, W).
            max_len (int, optional): Maximum generation length. Defaults to Config.MAX_LEN.

        Returns:
            list[str]: A list of decoded InChI strings.
        """
        if max_len is None:
            max_len = Config.MAX_LEN

        images = images.to(self.device)

        with torch.no_grad():
            # The model.generate method encapsulates the encoder forward pass,
            # attribute prediction, and the autoregressive decoder loop.
            generated_ids = self.model.generate(
                images, max_len=max_len, sos_idx=self.sos_idx, eos_idx=self.eos_idx
            )

        # Decode the generated integer sequences into strings
        decoded_texts = []
        generated_ids_np = generated_ids.cpu().numpy()

        for i in range(generated_ids_np.shape[0]):
            text = self.tokenizer.decode(generated_ids_np[i])
            decoded_texts.append(text)

        return decoded_texts

    def generate_submission(self, test_loader, submission_path=None):
        """
        Runs inference on the entire test set and saves the results to a CSV file.

        Args:
            test_loader (DataLoader): DataLoader providing the test dataset.
            submission_path (str, optional): Destination path for the CSV.
                                             Defaults to Config.SUBMISSION_FILE.
        """
        if submission_path is None:
            submission_path = Config.SUBMISSION_FILE

        print(f"Starting inference on {len(test_loader.dataset)} test images...")

        all_image_ids = []
        all_predictions = []

        # Iterate over the test loader
        for batch in test_loader:
            images = batch["image"]
            image_ids = batch["image_id"]

            # Run greedy decoding
            batch_predictions = self.greedy_decode(images)

            all_image_ids.extend(image_ids)
            all_predictions.extend(batch_predictions)

        # Create submission DataFrame
        df_submission = pd.DataFrame(
            {"image_id": all_image_ids, "InChI": all_predictions}
        )

        # Ensure the directory exists
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)

        # Save to CSV
        df_submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
        print("First 5 predictions:")
        print(df_submission.head())
