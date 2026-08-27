"""
Inference-time logic for the image captioning API.

Keeps image preprocessing and caption generation separate from the
FastAPI routing code (app/main.py), per Section 8.3 of the brief.

Critically, this module reuses the EXACT SAME ResNet-50 preprocessing
transform used during training (encoder.py) and the same Vocabulary /
LSTMDecoder classes used during training/evaluation, so there is no
train/serve mismatch.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# decoder.py and preprocessing.py live in src/ alongside train.py/evaluate.py,
# not inside app/. Add src/ to sys.path so we reuse those exact same classes
# instead of duplicating them here (duplication is how train/serve code
# quietly drifts apart).
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import Vocabulary  # noqa: E402
from decoder import LSTMDecoder  # noqa: E402


def load_vocabulary(vocab_path: Path) -> Vocabulary:
    """
    Rebuild a Vocabulary object from the saved word_to_index mapping.

    NOTE: this is the fixed version of the bug in evaluate.py, where
    load_vocabulary() built an empty Vocabulary and never actually read
    the saved JSON file (so decode() only ever produced <unk>).
    """
    with vocab_path.open("r", encoding="utf-8") as f:
        saved = json.load(f)

    vocabulary = Vocabulary(min_frequency=saved["min_frequency"])
    vocabulary.word_to_index = saved["word_to_index"]
    # Rebuild the reverse mapping; JSON keys are always strings, so
    # index_to_word keys must be cast back to int.
    vocabulary.index_to_word = {
        int(index): word for word, index in saved["word_to_index"].items()
    }
    return vocabulary


def load_config(config_path: Path) -> dict:
    """Load the hyperparameters needed to rebuild the exact decoder architecture."""
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_decoder(config: dict, weights_path: Path, device: torch.device) -> LSTMDecoder:
    """Rebuild the decoder architecture from config.json, then load trained weights."""
    decoder = LSTMDecoder(
        feature_dim=config["feature_dim"],
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        vocab_size=config["vocab_size"],
        num_layers=config["num_layers"],
    )
    decoder.load_state_dict(torch.load(weights_path, map_location=device))
    decoder.to(device)
    decoder.eval()
    return decoder


class CNNFeatureExtractor:
    """
    Thin wrapper around a frozen ResNet-50 for extracting a single
    image's feature vector at request time.

    This mirrors CNNEncoder in encoder.py, but only exposes what the
    API needs (single-image extraction, not batch caching to disk).
    Using weights.transforms() again here (instead of re-implementing
    resize/normalize by hand) guarantees the preprocessing matches
    training exactly.
    """

    def __init__(self, device: torch.device):
        self.device = device

        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        self.model.fc = nn.Identity()  # drop the ImageNet classification head

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = weights.transforms()

    def extract(self, image: Image.Image) -> torch.Tensor:
        """Extract a [1, 2048] feature vector for a single PIL image."""
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image_tensor)
        return features


class CaptionModel:
    """
    Bundles everything needed to go from a raw uploaded image to a
    generated caption string: CNN encoder, LSTM decoder, vocabulary,
    and config. Loaded ONCE at app startup (see main.py), not per request.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = load_config(models_dir / "config.json")
        self.vocabulary = load_vocabulary(models_dir / "vocabulary.json")
        self.decoder = load_decoder(
            self.config, models_dir / "decoder_best.pt", self.device
        )
        self.encoder = CNNFeatureExtractor(self.device)

    def caption_image(self, image: Image.Image) -> str:
        """
        Run the full inference pipeline on one PIL image:
        CNN features -> greedy decoding -> decoded caption string.
        """
        features = self.encoder.extract(image)

        token_ids = self.decoder.generate(
            features=features,
            start_token_id=self.config["start_token_id"],
            end_token_id=self.config["end_token_id"],
            max_length=self.config["max_length"],
        )

        return self.vocabulary.decode(token_ids)