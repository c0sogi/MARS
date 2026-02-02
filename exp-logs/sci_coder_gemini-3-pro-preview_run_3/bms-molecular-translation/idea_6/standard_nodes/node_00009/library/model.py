import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.utils import ATOM_LIST


class ResNetEncoder(nn.Module):
    """
    Encodes images into a fixed-size visual vector using a pre-trained ResNet-34.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        # Load ResNet34
        resnet = models.resnet34(pretrained=pretrained)

        # We want the output after the Global Average Pooling (avgpool) layer.
        # The standard ResNet structure in torchvision is:
        # conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
        # We take everything up to avgpool (excluding fc).
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # ResNet34 outputs 512 channels at the final layer
        self.output_dim = 512

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W)
        Returns:
            torch.Tensor: Visual features of shape (B, 512)
        """
        x = self.backbone(x)  # Output: (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # Flatten to (B, 512)
        return x


class FormulaPredictor(nn.Module):
    """
    Auxiliary head to predict atom counts from visual features.
    Acts as a 'Formula Estimator' to guide the decoder.
    """

    def __init__(self, input_dim, atom_count):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, atom_count),
            # No activation at output; we'll use MSELoss which expects raw values (or we can assume counts >= 0)
            # Usually raw logits are fine for regression, but since counts are positive, ReLU at end is optional.
            # We keep it linear to allow the loss function to handle the range.
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Visual features (B, input_dim)
        Returns:
            torch.Tensor: Predicted atom counts (B, atom_count)
        """
        return self.net(x)


class GRUDecoder(nn.Module):
    """
    GRU-based decoder for generating InChI strings.
    Initialized with a combination of visual features and predicted molecular formula.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, visual_dim, atom_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        # Project concatenated visual + atom features to initialize hidden state
        # Input size is 512 (visual) + 12 (atoms) -> Hidden Dim
        self.init_projection = nn.Linear(visual_dim + atom_dim, hidden_dim)
        self.bn_init = nn.BatchNorm1d(hidden_dim)

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        """
        Args:
            x (torch.Tensor): Input token indices (B, SeqLen)
            hidden (torch.Tensor): Hidden state (1, B, HiddenDim)
        Returns:
            logits (torch.Tensor): (B, SeqLen, VocabSize)
            hidden (torch.Tensor): Updated hidden state
        """
        embedded = self.embed(x)  # (B, SeqLen, EmbedDim)
        output, hidden = self.gru(embedded, hidden)  # output: (B, SeqLen, HiddenDim)
        logits = self.fc(output)  # (B, SeqLen, VocabSize)
        return logits, hidden

    def get_init_hidden(self, visual_features, atom_counts):
        """
        Computes the initial hidden state for the GRU.

        Args:
            visual_features (torch.Tensor): (B, 512)
            atom_counts (torch.Tensor): (B, NumAtoms)
        Returns:
            torch.Tensor: Initial hidden state (1, B, HiddenDim)
        """
        # Concatenate visual context and formula guidance
        combined = torch.cat([visual_features, atom_counts], dim=1)

        # Project to hidden dimension
        hidden = self.init_projection(combined)
        hidden = self.bn_init(hidden)
        hidden = torch.tanh(hidden)  # Tanh is standard for RNN state initialization

        # Unsqueeze to match GRU expectation: (num_layers, batch, hidden_size)
        return hidden.unsqueeze(0)


class FormulaConditionedModel(nn.Module):
    """
    End-to-End model combining ResNet Encoder, Formula Predictor, and GRU Decoder.
    """

    def __init__(
        self, vocab_size, embed_dim=256, hidden_dim=512, pretrained_encoder=True
    ):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained_encoder)

        self.atom_dim = len(ATOM_LIST)
        self.encoder_dim = self.encoder.output_dim

        self.predictor = FormulaPredictor(self.encoder_dim, self.atom_dim)

        self.decoder = GRUDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            visual_dim=self.encoder_dim,
            atom_dim=self.atom_dim,
        )
        self.vocab_size = vocab_size

    def forward(self, images, text_sequences=None):
        """
        Forward pass for training.

        Args:
            images (torch.Tensor): Images (B, 3, H, W)
            text_sequences (torch.Tensor, optional): Target sequences indices (B, L) including <SOS> and <EOS>.

        Returns:
            logits (torch.Tensor): Sequence predictions (B, L-1, VocabSize)
            pred_atoms (torch.Tensor): Formula predictions (B, AtomDim)
        """
        # 1. Extract Visual Features
        visual_features = self.encoder(images)  # (B, 512)

        # 2. Predict Formula (Auxiliary Task)
        pred_atoms = self.predictor(visual_features)  # (B, AtomDim)

        if text_sequences is not None:
            # 3. Decode with Teacher Forcing
            # Input to decoder: <SOS> ... <LastChar> (exclude <EOS> at end for input)
            # text_sequences[:, :-1] corresponds to inputs
            # text_sequences[:, 1:] corresponds to targets (handled by loss function outside)
            decoder_input = text_sequences[:, :-1]

            # Initialize hidden state with visual + formula context
            init_hidden = self.decoder.get_init_hidden(visual_features, pred_atoms)

            # Forward pass through decoder
            logits, _ = self.decoder(decoder_input, init_hidden)

            return logits, pred_atoms
        else:
            # Inference mode without teacher forcing usually uses .predict(),
            # but if forward is called without text, we return features.
            return None, pred_atoms

    def predict(self, images, tokenizer, max_len=275, device="cuda"):
        """
        Performs greedy decoding inference for a batch of images.

        Args:
            images (torch.Tensor): Batch of images (B, 3, H, W)
            tokenizer: Tokenizer object containing .sos_token_id and .eos_token_id
            max_len (int): Maximum sequence length to generate
            device (str): Device to run tensor operations on

        Returns:
            list[str]: List of predicted InChI strings
        """
        self.eval()
        batch_size = images.size(0)

        with torch.no_grad():
            # 1. Encode
            visual_features = self.encoder(images)
            pred_atoms = self.predictor(visual_features)

            # 2. Init Decoder
            hidden = self.decoder.get_init_hidden(visual_features, pred_atoms)

            # 3. Greedy Decoding Loop
            # Start with <SOS> token
            current_input = torch.full(
                (batch_size, 1), tokenizer.sos_token_id, dtype=torch.long, device=device
            )

            # Tensor to store predictions
            predictions_indices = torch.zeros(
                (batch_size, max_len), dtype=torch.long, device=device
            )

            # Mask to track which sequences have finished (hit <EOS>)
            active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            eos_id = tokenizer.eos_token_id

            for t in range(max_len):
                logits, hidden = self.decoder(current_input, hidden)
                # logits: (B, 1, VocabSize)

                # Greedy selection: argmax
                probs = F.softmax(logits[:, -1, :], dim=-1)
                predicted_id = torch.argmax(probs, dim=-1)  # (B,)

                # Store prediction
                predictions_indices[:, t] = predicted_id

                # Update input for next step
                current_input = predicted_id.unsqueeze(1)

                # Update active mask: if we predicted EOS, mark as inactive
                is_eos = predicted_id == eos_id
                active_mask = active_mask & (~is_eos)

                # Optimization: Stop if all sequences are finished
                if not active_mask.any():
                    break

            # 4. Convert indices to strings
            result_strings = []
            for i in range(batch_size):
                indices = predictions_indices[i].cpu().numpy()
                text = tokenizer.sequence_to_text(indices)
                result_strings.append(text)

            return result_strings
