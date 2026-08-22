from collections import Counter
import string
from pathlib import Path
import pandas as pd

class Vocabulary:
    def __init__(self, min_frequency: int = 2):

        self.min_frequency = min_frequency

        # Special tokens
        self.pad_token = "<pad>"
        self.start_token = "<start>"
        self.end_token = "<end>"
        self.unk_token = "<unk>"

        # Word → integer ID
        self.word_to_index = {
            self.pad_token: 0,
            self.start_token: 1,
            self.end_token: 2,
            self.unk_token: 3,
        }

        # Integer ID → word
        self.index_to_word = {
            0: self.pad_token,
            1: self.start_token,
            2: self.end_token,
            3: self.unk_token,
        }

    def tokenize(self, caption: str) -> list[str]:
        caption = caption.lower()

        # Remove punctuation.
        for symbol in string.punctuation:
            caption = caption.replace(symbol, "")
        return caption.split()

    def build(self, captions: list[str]) -> None:
        word_counts = Counter()

        # Count every word in the training captions.
        for caption in captions:
            words = self.tokenize(caption)
            word_counts.update(words)

        # Add sufficiently frequent words to vocabulary.
        for word, frequency in word_counts.items():

            if frequency >= self.min_frequency:

                index = len(self.word_to_index)

                self.word_to_index[word] = index
                self.index_to_word[index] = word

    def numericalize(self, caption: str) -> list[int]:
        tokens = self.tokenize(caption)
        tokens.insert(0, self.start_token)
        tokens.append(self.end_token)

        token_ids = []

        for token in tokens:

            if token in self.word_to_index:
                token_ids.append(
                    self.word_to_index[token]
                )
            else:
                token_ids.append(
                    self.word_to_index[self.unk_token]
                )

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        words = []

        for token_id in token_ids:

            word = self.index_to_word.get(
                token_id,
                self.unk_token
            )

            # Stop generation at <end>.
            if word == self.end_token:
                break

            # Don't include special tokens in the final caption.
            if word != self.pad_token and word != self.start_token:
                words.append(word)

        return " ".join(words)

    def __len__(self) -> int:
        return len(self.word_to_index)
    
    def pad_sequence(
        self,
        token_ids:list[int],
        max_length:int,
    )->list[int]:
        pad_id = self.word_to_index[self.pad_token]
        # If the caption is shorter than max_length,
        # add <pad> tokens at the end.
        if len(token_ids) < max_length:
            padding_length = max_length - len(token_ids)
            for _ in range(padding_length):
                token_ids.append(pad_id)
                
        # If the caption is longer than max_length,
        # truncate it.
        else:
            token_ids = token_ids[:max_length]
            
            #Place end token at the end
            token_ids[-1] = self.word_to_index[
            self.end_token
        ]
        return token_ids
    
def get_max_caption_length(
    captions: list[str],
    vocabulary: Vocabulary,
) -> int:
    lengths = []

    for caption in captions:
        token_ids = vocabulary.numericalize(caption)
        lengths.append(len(token_ids))

    return max(lengths)

def preprocess_captions(
    captions: list[str],
    vocabulary: Vocabulary,
    max_length: int,
) -> list[list[int]]:
    processed_captions = []

    for caption in captions:

        token_ids = vocabulary.numericalize(
            caption
        )
        token_ids = vocabulary.pad_sequence(
            token_ids,
            max_length,
        )
        processed_captions.append(token_ids)
    return processed_captions

if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    train_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "train_captions.csv"
    )

    train_df = pd.read_csv(train_file)

    captions = train_df["caption"].tolist()

    print(f"Training captions: {len(captions)}")

    # Build vocabulary ONLY from training captions.
    vocabulary = Vocabulary(min_frequency=2)

    vocabulary.build(captions)

    print(f"Vocabulary size: {len(vocabulary)}")

    max_length = get_max_caption_length(
        captions,
        vocabulary,
    )

    print(f"Maximum caption length: {max_length}")

    # Calculate some useful statistics.
    lengths = [
        len(vocabulary.numericalize(caption))
        for caption in captions
    ]

    print(
        f"Minimum caption length: {min(lengths)}"
    )

    print(
        f"Average caption length: "
        f"{sum(lengths) / len(lengths):.2f}"
    )

    print(
        f"95th percentile: "
        f"{pd.Series(lengths).quantile(0.95):.0f}"
    )

    print(
        f"99th percentile: "
        f"{pd.Series(lengths).quantile(0.99):.0f}"
    )
