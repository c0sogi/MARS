import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class XLMRobertaForQA(nn.Module):
    """
    XLM-Roberta based model for Question Answering with Multi-Task Heads.
    Includes a Span Head for start/end logits and a Relevance Head for answer presence.
    """

    def __init__(self, model_name):
        super(XLMRobertaForQA, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        # Load the pretrained backbone
        # We name it 'roberta' so that utils.get_optimizer_params can find it easily
        self.roberta = AutoModel.from_pretrained(model_name, config=self.config)

        # Span Head: Predicts start and end scores for each token
        # Input: hidden_size, Output: 2 (start_logit, end_logit)
        self.qa_outputs = nn.Linear(self.config.hidden_size, 2)

        # Relevance Head: Predicts whether the context contains the answer
        # Input: hidden_size (CLS token), Output: 1 (logit)
        self.relevance_classifier = nn.Linear(self.config.hidden_size, 1)

        # Initialize weights for the new heads
        self._init_weights(self.qa_outputs)
        self._init_weights(self.relevance_classifier)

    def _init_weights(self, module):
        """Initialize the weights of the specific module."""
        if isinstance(module, nn.Linear):
            # Using the initializer range from the config (usually 0.02)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.

        Returns:
            start_logits (torch.Tensor): Logits for the start position (Batch, Seq_Len)
            end_logits (torch.Tensor): Logits for the end position (Batch, Seq_Len)
            relevance_logits (torch.Tensor): Logits for answer relevance (Batch,)
        """
        # Pass through the backbone
        outputs = self.roberta(input_ids, attention_mask=attention_mask)

        # Sequence output: (Batch, Seq_Len, Hidden)
        sequence_output = outputs.last_hidden_state

        # --- Span Prediction ---
        logits = self.qa_outputs(sequence_output)  # (Batch, Seq_Len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (Batch, Seq_Len)
        end_logits = end_logits.squeeze(-1)  # (Batch, Seq_Len)

        # --- Relevance Prediction ---
        # Use the [CLS] token (index 0) for classification
        cls_token = sequence_output[:, 0, :]  # (Batch, Hidden)
        relevance_logits = self.relevance_classifier(cls_token).squeeze(-1)  # (Batch,)

        return start_logits, end_logits, relevance_logits


class FGM:
    """
    Fast Gradient Method (FGM) for Adversarial Training.
    Perturbs the embeddings to smooth the loss landscape and improve generalization.
    """

    def __init__(self, model, epsilon=1.0, param_name="word_embeddings"):
        """
        Args:
            model (nn.Module): The model to attack.
            epsilon (float): The magnitude of the perturbation.
            param_name (str): The name of the parameter to perturb (usually embeddings).
        """
        self.model = model
        self.epsilon = epsilon
        self.param_name = param_name
        self.backup = {}

    def attack(self):
        """
        Generates the adversarial perturbation and applies it to the model parameters.
        Should be called after backward() and before optimizer.step().
        """
        for name, param in self.model.named_parameters():
            # Apply perturbation only to the target parameter (embeddings) if it has gradients
            if (
                param.requires_grad
                and self.param_name in name
                and param.grad is not None
            ):
                # Save the original data
                self.backup[name] = param.data.clone()

                # Calculate the norm of the gradient
                norm = torch.norm(param.grad)

                # Apply perturbation if norm is valid
                if norm != 0 and not torch.isnan(norm):
                    # r_at = epsilon * g / ||g||
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        """
        Restores the original model parameters.
        Should be called after optimizer.step() or before the next forward pass.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
