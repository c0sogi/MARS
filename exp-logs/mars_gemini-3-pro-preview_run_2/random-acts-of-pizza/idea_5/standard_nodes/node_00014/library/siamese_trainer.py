import os
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
from library import config, data_handler, preprocessor, utils

logger = utils.setup_logger("siamese_trainer")


class FineTuner:
    """
    Encapsulates the fine-tuning logic for the representation learning stage.
    Uses a Siamese Network approach with BatchHardTripletLoss to optimize
    embeddings for the binary classification task.
    """

    def __init__(self):
        self.model_name = config.TRANSFORMER_MODEL_NAME
        self.save_path = config.FINE_TUNED_MODEL_PATH
        self.device = config.DEVICE
        self.model = None

    def train(self, load_cached_data: bool = True):
        """
        Executes the fine-tuning pipeline.

        Args:
            load_cached_data (bool): If True, skips training if the model
                                     already exists on disk.
        """
        # 1. Check Caching
        if load_cached_data and os.path.exists(self.save_path):
            logger.info(
                f"Fine-tuned model found at {self.save_path}. Skipping training."
            )
            self.load_model()
            return

        logger.info("Starting fine-tuning process...")
        utils.set_seed(config.SEED)

        # 2. Load Data
        # We only need the training set for fine-tuning
        df_train, _, _ = data_handler.load_datasets(load_cached_data=load_cached_data)

        # 3. Prepare Dataset
        logger.info("Preparing InputExamples for Siamese training...")
        dataset_builder = preprocessor.SiameseDatasetBuilder()
        train_examples = dataset_builder.create_examples(df_train)

        if not train_examples:
            logger.warning("No training examples created. Skipping fine-tuning.")
            return

        # 4. Create DataLoader
        train_dataloader = DataLoader(
            train_examples, shuffle=True, batch_size=config.FINE_TUNE_BATCH_SIZE
        )

        # 5. Initialize Model
        logger.info(f"Initializing base model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)

        # 6. Define Loss
        # BatchHardTripletLoss mines the hardest triplets within a batch
        train_loss = losses.BatchHardTripletLoss(
            model=self.model,
            margin=config.TRIPLET_MARGIN,
        )

        # 7. Train
        logger.info(f"Training for {config.FINE_TUNE_EPOCHS} epochs...")
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=config.FINE_TUNE_EPOCHS,
            optimizer_params={"lr": config.FINE_TUNE_LR},
            show_progress_bar=False,
            output_path=self.save_path,
        )

        logger.info(f"Fine-tuning complete. Model saved to {self.save_path}")

    def load_model(self):
        """
        Loads the model from disk. Falls back to base model if fine-tuned version is missing.
        """
        if os.path.exists(self.save_path):
            logger.info(f"Loading fine-tuned model from {self.save_path}...")
            self.model = SentenceTransformer(self.save_path, device=self.device)
        else:
            logger.warning(
                f"Fine-tuned model not found at {self.save_path}. Loading base model {self.model_name}."
            )
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: list) -> object:
        """
        Generates embeddings for a list of texts using the current model.

        Args:
            texts (list): List of strings to encode.

        Returns:
            numpy.ndarray: Array of embeddings.
        """
        if self.model is None:
            self.load_model()

        logger.info(f"Encoding {len(texts)} texts...")
        return self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            device=self.device,
            convert_to_numpy=True,
        )
