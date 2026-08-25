"""
Step 6: Evaluate the trained decoder on the held-out test set.

For each test image:
    - load its cached CNN feature
    - generate a caption via greedy decoding (decoder.generate)
    - compare it against all reference captions for that image

Reports corpus-level BLEU-1 through BLEU-4 and prints sample
generated captions alongside their references for the report.

Requires: pip install nltk
"""
import json
from pathlib import Path

import pandas as pd
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from preprocessing import Vocabulary
from decoder import LSTMDecoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_DIR = PROJECT_ROOT / "features"
MODELS_DIR = PROJECT_ROOT / "models"

TEST_FILE = PROCESSED_DIR / "test_captions.csv"


def load_vocabulary(vocab_path: Path) -> Vocabulary:
    """Rebuild a Vocabulary object from the saved word_to_index mapping."""
    with vocab_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    vocabulary = Vocabulary(min_frequency=data["min_frequency"])
    vocabulary.word_to_index = {w: int(i) for w, i in data["word_to_index"].items()}
    vocabulary.index_to_word = {i: w for w, i in vocabulary.word_to_index.items()}
    return vocabulary


def load_decoder(config: dict, weights_path: Path, device: torch.device) -> LSTMDecoder:
    """Rebuild the exact architecture from config.json, then load weights."""
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


def load_feature(image_name: str) -> torch.Tensor:
    """Load a single cached ResNet-50 feature vector from disk."""
    stem = Path(image_name).stem
    feature_path = FEATURE_DIR / f"{stem}.pt"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing cached feature for image: {image_name}")
    return torch.load(feature_path)


def build_references_and_candidates(
    test_df: pd.DataFrame,
    vocabulary: Vocabulary,
    decoder: LSTMDecoder,
    config: dict,
    device: torch.device,
) -> tuple[list, list, list]:
    """
    Group the test set by image (multiple references each), generate
    one candidate caption per image via greedy decoding, and return
    tokenized references/candidates in the shape nltk's corpus_bleu
    expects: references = list[list[list[str]]], candidates = list[list[str]].
    """
    references = []
    candidates = []
    examples = []  # (image_name, generated_caption, reference_captions) for reporting

    grouped = test_df.groupby("image")["caption"].apply(list)

    skipped = 0
    for image_name, reference_captions in grouped.items():
        try:
            features = load_feature(image_name).unsqueeze(0).to(device)
        except FileNotFoundError:
            # Skip images without a cached feature (e.g. partial dataset).
            skipped += 1
            continue

        token_ids = decoder.generate(
            features=features,
            start_token_id=config["start_token_id"],
            end_token_id=config["end_token_id"],
            max_length=config["max_length"],
        )
        generated_caption = vocabulary.decode(token_ids)

        references.append([vocabulary.tokenize(ref) for ref in reference_captions])
        candidates.append(generated_caption.split())
        examples.append((image_name, generated_caption, reference_captions))

    if skipped:
        print(f"Skipped {skipped} test images with no cached feature.")

    return references, candidates, examples


def compute_bleu_scores(references: list, candidates: list) -> dict:
    """Corpus-level BLEU-1 through BLEU-4, with smoothing since short
    generated captions otherwise often score exactly 0 on higher-order
    n-grams (see Section 5 Step 6 guidance)."""
    smoothing = SmoothingFunction().method1

    scores = {}
    for n in range(1, 5):
        weights = tuple(1.0 / n for _ in range(n)) + tuple(0.0 for _ in range(4 - n))
        scores[f"BLEU-{n}"] = corpus_bleu(
            references, candidates, weights=weights, smoothing_function=smoothing
        )
    return scores


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with (MODELS_DIR / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)

    vocabulary = load_vocabulary(MODELS_DIR / "vocabulary.json")
    decoder = load_decoder(config, MODELS_DIR / "decoder_best.pt", device)

    test_df = pd.read_csv(TEST_FILE)

    references, candidates, examples = build_references_and_candidates(
        test_df, vocabulary, decoder, config, device
    )

    print(f"\nEvaluated {len(candidates)} test images.")

    scores = compute_bleu_scores(references, candidates)
    print("\n--- BLEU Scores ---")
    for name, value in scores.items():
        print(f"{name}: {value:.4f}")

    print("\n--- Sample generated captions (first 10) ---")
    for image_name, generated, reference_captions in examples[:10]:
        print(f"\nImage: {image_name}")
        print(f"Generated: {generated}")
        for i, ref in enumerate(reference_captions, start=1):
            print(f"  Reference {i}: {ref}")


if __name__ == "__main__":
    main()