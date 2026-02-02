import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from library.config import Config


class CustomXLMRoberta(nn.Module):
    """
    Custom XLM-Roberta-Large model for Question Answering.

    Architectural Features:
    1. Backbone: XLM-Roberta-Large.
    2. Layer Re-initialization: Resets the top N encoder layers to mitigate pre-training bias.
    3. Multi-Sample Dropout: Applies dropout multiple times per forward pass to smooth loss and improve generalization.
    4. Dual Heads:
       - QA Head (Span Predictor): Predicts start and end token logits.
       - Relevance Head (Classifier): Predicts whether the answer exists in the window.
    """

    def __init__(self):
        super(CustomXLMRoberta, self).__init__()

        # 1. Load Configuration and Backbone
        self.config = AutoConfig.from_pretrained(Config.model_name)
        self.backbone = AutoModel.from_pretrained(Config.model_name, config=self.config)

        # 2. Structural Innovation: Layer Re-initialization
        # Reset weights of the top encoder layers to learn task-specific features from scratch
        if Config.reinit_layers > 0:
            self._reinit_layers()

        # 3. Structural Innovation: Multi-Sample Dropout
        # We define a single dropout module but will apply it multiple times in the forward pass
        self.dropout = nn.Dropout(Config.dropout_rate)

        # 4. Classification Heads
        # Span Predictor: Output size 2 (Start Logit, End Logit)
        self.qa_outputs = nn.Linear(Config.hidden_size, Config.num_labels)

        # Relevance Classifier: Output size 1 (Binary Logit)
        self.relevance_classifier = nn.Linear(Config.hidden_size, 1)

        # Initialize the new heads
        self._init_weights(self.qa_outputs)
        self._init_weights(self.relevance_classifier)

    def _init_weights(self, module):
        """
        Standard weight initialization utility for Linear and LayerNorm layers.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _reinit_layers(self):
        """
        Re-initializes the parameters of the top N layers of the backbone encoder.
        """
        if hasattr(self.backbone, "encoder") and hasattr(
            self.backbone.encoder, "layer"
        ):
            layers = self.backbone.encoder.layer
            num_layers = len(layers)
            start_layer = num_layers - Config.reinit_layers

            print(
                f"Re-initializing top {Config.reinit_layers} encoder layers (Layers {start_layer} to {num_layers-1})..."
            )

            for i in range(start_layer, num_layers):
                for module in layers[i].modules():
                    self._init_weights(module)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
        relevance=None,
        **kwargs,
    ):
        """
        Forward pass implementing Multi-Sample Dropout.

        Args:
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor): Attention mask.
            start_positions (Tensor, optional): Ground truth start indices.
            end_positions (Tensor, optional): Ground truth end indices.
            relevance (Tensor, optional): Ground truth relevance labels (0 or 1).

        Returns:
            dict: Contains 'loss' (if targets provided), 'start_logits', 'end_logits', 'relevance_logits'.
        """
        # --- Backbone Forward ---
        outputs = self.backbone(input_ids, attention_mask=attention_mask)

        # Sequence output for Span Prediction: (Batch, Seq_Len, Hidden)
        sequence_output = outputs.last_hidden_state

        # Pooled output for Relevance Classification: (Batch, Hidden)
        # Use pooler_output if available (XLM-R usually has it), otherwise use CLS token (index 0)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = sequence_output[:, 0, :]

        # --- Multi-Sample Dropout & Loss Calculation ---
        total_loss = 0.0
        mean_start_logits = 0.0
        mean_end_logits = 0.0
        mean_relevance_logits = 0.0

        # Check if we are in training mode (targets provided)
        calc_loss = (
            (start_positions is not None)
            and (end_positions is not None)
            and (relevance is not None)
        )

        # Loop K times for Multi-Sample Dropout
        for i in range(Config.multi_sample_dropout_num):
            # 1. Apply Dropout (Generates a unique mask each iteration)
            seq_drop = self.dropout(sequence_output)
            pool_drop = self.dropout(pooled_output)

            # 2. Compute Logits
            qa_logits = self.qa_outputs(seq_drop)  # (Batch, Seq_Len, 2)
            start_logits = qa_logits[:, :, 0]
            end_logits = qa_logits[:, :, 1]

            rel_logits = self.relevance_classifier(pool_drop)  # (Batch, 1)

            # 3. Accumulate Logits (for Inference averaging)
            mean_start_logits += start_logits / Config.multi_sample_dropout_num
            mean_end_logits += end_logits / Config.multi_sample_dropout_num
            mean_relevance_logits += rel_logits / Config.multi_sample_dropout_num

            # 4. Calculate Loss (if training)
            if calc_loss:
                # Span Loss (Cross Entropy)
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                span_loss = (start_loss + end_loss) / 2.0

                # Relevance Loss (BCEWithLogits)
                # Ensure targets are float for BCE
                rel_loss_fct = nn.BCEWithLogitsLoss()
                rel_loss = rel_loss_fct(rel_logits.view(-1), relevance.view(-1))

                # Weighted Combined Loss
                loss = span_loss + Config.relevance_loss_weight * rel_loss

                # Accumulate Loss (Average)
                total_loss += loss / Config.multi_sample_dropout_num

        # --- Prepare Output ---
        output_dict = {
            "start_logits": mean_start_logits,
            "end_logits": mean_end_logits,
            "relevance_logits": mean_relevance_logits,
        }

        if calc_loss:
            output_dict["loss"] = total_loss

        return output_dict
