import torch
import torch.nn as nn


class LSTMDecoder(nn.Module):
    """
    LSTM-based decoder for image caption generation.

    The decoder receives:
        1. Image features extracted by the CNN encoder
        2. Caption token IDs

    During training, teacher forcing is used:
    the actual previous word is given to the LSTM.

    During inference, the decoder can generate a caption
    one token at a time.
    """

    def __init__(
        self,
        feature_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # --------------------------------------------------
        # Convert image features into the initial hidden state
        # --------------------------------------------------

        self.feature_to_hidden = nn.Linear(
            feature_dim,
            hidden_dim
        )

        # --------------------------------------------------
        # Convert image features into the initial cell state
        # --------------------------------------------------

        self.feature_to_cell = nn.Linear(
            feature_dim,
            hidden_dim
        )

        # --------------------------------------------------
        # Convert word IDs into dense word embeddings
        # --------------------------------------------------

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        # --------------------------------------------------
        # LSTM
        # --------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # --------------------------------------------------
        # Convert LSTM hidden states into vocabulary scores
        # --------------------------------------------------

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def initialize_hidden_state(self, features):
        """
        Convert image features into the initial LSTM
        hidden state and cell state.

        Parameters
        ----------
        features : torch.Tensor
            Shape:
            [batch_size, feature_dim]

        Returns
        -------
        hidden : torch.Tensor
            Shape:
            [num_layers, batch_size, hidden_dim]

        cell : torch.Tensor
            Shape:
            [num_layers, batch_size, hidden_dim]
        """

        hidden = self.feature_to_hidden(features)
        cell = self.feature_to_cell(features)

        # Current output has shape:
        # [batch_size, hidden_dim]

        # LSTM expects:
        # [num_layers, batch_size, hidden_dim]

        hidden = hidden.unsqueeze(0)
        cell = cell.unsqueeze(0)

        # If using more than one LSTM layer,
        # repeat the initial state for every layer.

        if self.num_layers > 1:
            hidden = hidden.repeat(
                self.num_layers,
                1,
                1
            )

            cell = cell.repeat(
                self.num_layers,
                1,
                1
            )

        return hidden, cell

    def forward(
        self,
        features: torch.Tensor,
        captions: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate vocabulary predictions during training.

        Teacher forcing is used because the complete caption
        is provided to the decoder.

        Parameters
        ----------
        features : torch.Tensor
            Image features.

            Shape:
            [batch_size, feature_dim]

        captions : torch.Tensor
            Input caption token IDs.

            Shape:
            [batch_size, sequence_length]

        Returns
        -------
        torch.Tensor
            Vocabulary scores.

            Shape:
            [batch_size, sequence_length, vocab_size]
        """

        # --------------------------------------------------
        # Get initial hidden and cell states from image
        # --------------------------------------------------

        hidden, cell = self.initialize_hidden_state(
            features
        )

        # --------------------------------------------------
        # Convert caption token IDs into embeddings
        # --------------------------------------------------

        embeddings = self.embedding(captions)

        # Shape:
        # [batch_size, sequence_length, embedding_dim]

        # --------------------------------------------------
        # Run caption through LSTM
        # --------------------------------------------------

        outputs, _ = self.lstm(
            embeddings,
            (hidden, cell)
        )

        # Shape:
        # [batch_size, sequence_length, hidden_dim]

        # --------------------------------------------------
        # Convert LSTM outputs into vocabulary scores
        # --------------------------------------------------

        outputs = self.fc(outputs)

        # Shape:
        # [batch_size, sequence_length, vocab_size]

        return outputs

    def generate(
        self,
        features: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_length: int = 30
    ) -> list[int]:
        """
        Generate a caption one token at a time.

        This method is used during validation/inference,
        not normal teacher-forced training.

        Greedy decoding is used: at every step, the word
        with the highest predicted score is selected.

        Parameters
        ----------
        features : torch.Tensor
            Image features.

            Shape:
            [1, feature_dim]

        start_token_id : int
            ID of <start> token.

        end_token_id : int
            ID of <end> token.

        max_length : int
            Maximum number of tokens to generate.

        Returns
        -------
        list[int]
            Generated token IDs.
        """

        self.eval()

        generated_tokens = [start_token_id]

        hidden, cell = self.initialize_hidden_state(
            features
        )

        current_token = torch.tensor(
            [[start_token_id]],
            dtype=torch.long,
            device=features.device
        )

        with torch.no_grad():

            for _ in range(max_length):

                # Convert current word ID into embedding
                embedding = self.embedding(
                    current_token
                )

                # embedding shape:
                # [1, 1, embedding_dim]

                # Generate next hidden state
                output, (hidden, cell) = self.lstm(
                    embedding,
                    (hidden, cell)
                )

                # Convert LSTM output to vocabulary scores
                scores = self.fc(output)

                # Get the word with highest score
                next_token = scores.argmax(
                    dim=-1
                ).item()

                # Add predicted word
                generated_tokens.append(
                    next_token
                )

                # Stop when <end> is generated
                if next_token == end_token_id:
                    break

                # Use predicted word as next input
                current_token = torch.tensor(
                    [[next_token]],
                    dtype=torch.long,
                    device=features.device
                )

        return generated_tokens


if __name__ == "__main__":

    # --------------------------------------------------
    # Simple test
    # --------------------------------------------------

    feature_dim = 2048
    embedding_dim = 256
    hidden_dim = 512
    vocab_size = 5000

    decoder = LSTMDecoder(
        feature_dim=feature_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size
    )

    # Fake image features for testing
    features = torch.randn(
        4,
        feature_dim
    )

    # Fake caption token IDs
    captions = torch.randint(
        0,
        vocab_size,
        (4, 20)
    )

    outputs = decoder(
        features,
        captions
    )

    print("Feature shape:", features.shape)
    print("Caption shape:", captions.shape)
    print("Output shape:", outputs.shape)