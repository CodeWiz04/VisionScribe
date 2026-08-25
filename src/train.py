"""
Step 5: Train the LSTM decoder on cached CNN features and
tokenized captions. Run from inside src/: `python train.py`
"""
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from preprocessing import Vocabulary
from dataset import CaptionDataset
from decoder import LSTMDecoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_DIR = PROJECT_ROOT / "features"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_FILE = PROCESSED_DIR / "train_captions.csv"
VAL_FILE = PROCESSED_DIR / "val_captions.csv"

FEATURE_DIM = 2048
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 1
DROPOUT = 0.0
MIN_WORD_FREQUENCY = 2
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
NUM_EPOCHS = 20


def build_vocabulary(train_df: pd.DataFrame) -> Vocabulary:
    """Build the vocabulary from the TRAINING captions only."""
    vocabulary = Vocabulary(min_frequency=MIN_WORD_FREQUENCY)
    vocabulary.build(train_df["caption"].tolist())
    return vocabulary


def get_max_caption_length(captions: list[str], vocabulary: Vocabulary) -> int:
    return max(len(vocabulary.numericalize(c)) for c in captions)


def load_feature(image_name: str) -> torch.Tensor:
    """Load a single cached ResNet-50 feature vector from disk."""
    stem = Path(image_name).stem
    feature_path = FEATURE_DIR / f"{stem}.pt"
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Missing cached feature for image: {image_name}. "
            f"Run feature extraction first."
        )
    return torch.load(feature_path)


def collate_batch(batch: list[dict]) -> dict:
    """Look up and stack the cached CNN feature for every image in the batch."""
    features = torch.stack([load_feature(item["image_name"]) for item in batch])
    captions = torch.stack([item["caption"] for item in batch])
    return {"features": features, "captions": captions}


def compute_masked_loss(
    outputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module
) -> torch.Tensor:
    """Cross-entropy loss over the vocabulary, ignoring <pad> positions
    (criterion must be built with ignore_index=<pad_token_id>)."""
    vocab_size = outputs.shape[-1]
    outputs = outputs.reshape(-1, vocab_size)
    targets = targets.reshape(-1)
    return criterion(outputs, targets)


def run_epoch(
    decoder: LSTMDecoder,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """optimizer=None runs in eval mode (validation), no gradient updates."""
    is_training = optimizer is not None
    decoder.train() if is_training else decoder.eval()

    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        features = batch["features"].to(device)
        captions = batch["captions"].to(device)

        # Teacher forcing: shift by one so the model predicts the
        # NEXT word given every previous true word.
        decoder_input = captions[:, :-1]
        targets = captions[:, 1:]

        with torch.set_grad_enabled(is_training):
            outputs = decoder(features, decoder_input)
            loss = compute_masked_loss(outputs, targets, criterion)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / total_batches


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)

    vocabulary = build_vocabulary(train_df)
    max_length = get_max_caption_length(train_df["caption"].tolist(), vocabulary)

    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Max caption length: {max_length}")

    train_dataset = CaptionDataset(TRAIN_FILE, vocabulary, max_length)
    val_dataset = CaptionDataset(VAL_FILE, vocabulary, max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch
    )

    decoder = LSTMDecoder(
        feature_dim=FEATURE_DIM,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        vocab_size=len(vocabulary),
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    pad_token_id = vocabulary.word_to_index[vocabulary.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = run_epoch(decoder, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(decoder, val_loader, criterion, device, optimizer=None)

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} — "
            f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(decoder.state_dict(), MODELS_DIR / "decoder_best.pt")

            with (MODELS_DIR / "vocabulary.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "word_to_index": vocabulary.word_to_index,
                        "min_frequency": vocabulary.min_frequency,
                    },
                    f,
                )

            config = {
                "feature_dim": FEATURE_DIM,
                "embedding_dim": EMBEDDING_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "vocab_size": len(vocabulary),
                "max_length": max_length,
                "pad_token_id": pad_token_id,
                "start_token_id": vocabulary.word_to_index[vocabulary.start_token],
                "end_token_id": vocabulary.word_to_index[vocabulary.end_token],
            }
            with (MODELS_DIR / "config.json").open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

            print(f"  \u21b3 New best model saved (val_loss: {val_loss:.4f})")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()