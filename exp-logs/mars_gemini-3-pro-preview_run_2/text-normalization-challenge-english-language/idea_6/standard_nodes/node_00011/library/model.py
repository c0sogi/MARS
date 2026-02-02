import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoConfig
from library.config import Config
from library.label_manager import LabelEngineer
from library.utils import get_logger

logger = get_logger("model")


class TransformerTokenClassifier(nn.Module):
    """
    Transformer-based Token Classifier for Fine-Grained Text Normalization.

    This model wraps a HuggingFace AutoModelForTokenClassification. It dynamically
    determines the number of output classes (transformation IDs) by consulting the
    LabelEngineer, ensuring the classification head matches the deterministic
    transformation registry.
    """

    def __init__(self, pretrained_model_name: str = Config.MODEL_NAME):
        """
        Initializes the model architecture.

        Args:
            pretrained_model_name (str): The name of the pre-trained model (e.g., 'roberta-base')
                                         or a path to a directory containing a saved model.
                                         Defaults to Config.MODEL_NAME.
        """
        super().__init__()

        # 1. Determine Label Space
        # We instantiate LabelEngineer to access the label mapping.
        # This ensures we know exactly how many fine-grained transformations exist.
        self.label_engineer = LabelEngineer()

        # Ensure the encoder is loaded/created so we can get the correct count.
        # This relies on the static TransformationRegistry, so it is consistent across runs.
        self.label_engineer._load_or_create_label_encoder()

        self.label_names = self.label_engineer.label_names
        self.num_labels = len(self.label_names)

        logger.info(
            f"Initializing TransformerTokenClassifier with {self.num_labels} classes."
        )

        # 2. Configure Model
        # We create a mapping of ID <-> Label for the config, which helps with debugging and inference.
        id2label = {i: name for i, name in enumerate(self.label_names)}
        label2id = {name: i for i, name in enumerate(self.label_names)}

        # Load configuration, overriding the head dimensions and dropout
        config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=self.num_labels,
            id2label=id2label,
            label2id=label2id,
            hidden_dropout_prob=Config.DROPOUT,
            attention_probs_dropout_prob=Config.DROPOUT,
        )

        # 3. Load Pre-trained Weights
        # If pretrained_model_name is a local path, this loads the saved weights.
        # If it is a hub ID, it loads the base pre-trained weights and initializes the head randomly.
        self.model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name, config=config
        )

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        """
        Performs a forward pass through the network.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs (Batch, Seq_Len).
            attention_mask (torch.Tensor): Tensor indicating valid tokens (Batch, Seq_Len).
            labels (torch.Tensor, optional): Ground truth label IDs for loss calculation.

        Returns:
            transformers.modeling_outputs.TokenClassifierOutput:
                Object containing:
                - loss (torch.Tensor, optional): Classification loss if labels are provided.
                - logits (torch.Tensor): Raw prediction scores (Batch, Seq_Len, Num_Labels).
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            **kwargs,
        )

    def save_pretrained(self, save_directory: str):
        """
        Saves the model weights and configuration to the specified directory.

        Args:
            save_directory (str): The directory path to save the model artifacts.
        """
        self.model.save_pretrained(save_directory)
        logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory: str):
        """
        Factory method to load a saved model from a directory.

        Args:
            load_directory (str): The directory containing the saved model files.

        Returns:
            TransformerTokenClassifier: The loaded model instance.
        """
        logger.info(f"Loading model from {load_directory}")
        # Initialize the class using the directory as the model name.
        # AutoModelForTokenClassification will detect the local files and load them.
        return cls(pretrained_model_name=load_directory)
