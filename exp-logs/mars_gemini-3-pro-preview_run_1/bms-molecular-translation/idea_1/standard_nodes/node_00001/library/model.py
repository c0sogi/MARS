import torch
import torch.nn as nn
import torchvision.models as models
import random
from library.config import Config


class EncoderCNN(nn.Module):
    """
    Encoder using a ResNet-18 backbone to extract global image features.
    Maps the image to the initial hidden state of the RNN.
    """

    def __init__(self, hidden_size):
        super(EncoderCNN, self).__init__()
        # Load pretrained ResNet
        resnet = models.resnet18(pretrained=True)

        # Remove the last fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # Linear layer to map ResNet output (512) to RNN hidden size
        self.linear = nn.Linear(resnet.fc.in_features, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size, momentum=0.01)

    def forward(self, images):
        """
        Args:
            images: Tensor of shape (batch_size, 3, 256, 256)
        Returns:
            features: Tensor of shape (batch_size, hidden_size) to init RNN state
        """
        with torch.no_grad():
            features = self.resnet(images)

        # ResNet output is (batch, 512, 1, 1), flatten to (batch, 512)
        features = features.reshape(features.size(0), -1)

        # Project to hidden size
        features = self.linear(features)
        features = self.bn(features)

        return features


class DecoderRNN(nn.Module):
    """
    GRU-based Decoder to generate InChI string from hidden state.
    """

    def __init__(self, embed_dim, hidden_size, vocab_size, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h):
        """
        Args:
            x: Input token indices (batch_size, 1)
            h: Hidden state (1, batch_size, hidden_size)
        Returns:
            outputs: Logits (batch_size, 1, vocab_size)
            h: New hidden state
        """
        # Embed the input token
        # x shape: (batch, 1) -> (batch, 1, embed_dim)
        x = self.embed(x)
        x = self.dropout(x)

        # GRU forward
        # output shape: (batch, 1, hidden_size)
        # h shape: (1, batch, hidden_size)
        output, h = self.gru(x, h)

        # Project to vocab size
        outputs = self.linear(output)
        return outputs, h


class ShowAndTell(nn.Module):
    """
    Encoder-Decoder architecture for Image Captioning (InChI prediction).
    """

    def __init__(self, vocab_size, sos_idx, eos_idx, pad_idx, max_len=Config.MAX_LEN):
        super(ShowAndTell, self).__init__()
        self.vocab_size = vocab_size
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.device = Config.DEVICE

        self.encoder = EncoderCNN(Config.HIDDEN_SIZE)
        self.decoder = DecoderRNN(
            Config.EMBED_DIM, Config.HIDDEN_SIZE, vocab_size, Config.DROPOUT
        )

    def forward(self, images, captions):
        """
        Forward pass for training with Teacher Forcing.

        Args:
            images: (batch_size, 3, H, W)
            captions: (batch_size, max_len) - Ground truth indices including <sos> and <eos>

        Returns:
            outputs: (batch_size, max_len, vocab_size) - Logits for each step
        """
        batch_size = images.size(0)
        target_len = captions.size(1)

        # Encode images to get initial hidden state
        # Encoder output: (batch, hidden_size)
        # GRU expects hidden state: (num_layers, batch, hidden_size)
        features = self.encoder(images)
        hidden = features.unsqueeze(0)

        # Prepare tensor to hold outputs
        outputs = torch.zeros(batch_size, target_len, self.vocab_size).to(self.device)

        # First input is <sos> token
        decoder_input = captions[:, 0].unsqueeze(1)  # (batch, 1)

        # Iterate through the sequence
        # Note: We run up to target_len - 1 because the last token predicted corresponds to the last target
        for t in range(1, target_len):
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t, :] = output.squeeze(1)

            # Determine next input
            teacher_force = random.random() < Config.TEACHER_FORCING_RATIO

            # Get the highest predicted token from current step
            top1 = output.argmax(2)  # (batch, 1)

            # If teacher forcing, use actual next token as input
            # else use predicted token
            if teacher_force:
                decoder_input = captions[:, t].unsqueeze(1)
            else:
                decoder_input = top1

        return outputs

    def sample(self, images):
        """
        Inference method using Greedy Decoding.

        Args:
            images: (batch_size, 3, H, W)

        Returns:
            sampled_ids: (batch_size, max_len) - Predicted indices
        """
        with torch.no_grad():
            batch_size = images.size(0)

            # Encode images
            features = self.encoder(images)
            hidden = features.unsqueeze(0)

            sampled_ids = []

            # Start with <sos>
            decoder_input = torch.full(
                (batch_size, 1), self.sos_idx, dtype=torch.long
            ).to(self.device)

            # Loop up to max_len
            for t in range(self.max_len):
                output, hidden = self.decoder(decoder_input, hidden)

                # Greedy search: select max probability
                # output: (batch, 1, vocab) -> top1: (batch, 1)
                top1 = output.argmax(2)

                sampled_ids.append(top1)

                # Next input is current prediction
                decoder_input = top1

                # Check if all sequences have hit EOS?
                # Optimization: We could stop early if all batch items predicted EOS,
                # but for simplicity/batching speed we often run fixed length or handle post-process.

            # Stack along sequence dimension: (batch, max_len)
            sampled_ids = torch.cat(sampled_ids, dim=1)

            return sampled_ids
