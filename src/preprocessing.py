from collections import Counter
import string


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
        """
        Clean and tokenize a caption.

        The caption is converted to lowercase, punctuation is
        removed, and the remaining text is split into words.

        Parameters
        ----------
        caption : str
            Raw caption text.

        Returns
        -------
        list[str]
            List of tokens.
        """

        caption = caption.lower()

        # Remove punctuation.
        for symbol in string.punctuation:
            caption = caption.replace(symbol, "")

        # Split into individual words.
        return caption.split()

    def build(self, captions: list[str]) -> None:
        """
        Build the vocabulary using training captions.

        Only words appearing at least min_frequency times
        are added to the vocabulary.

        Parameters
        ----------
        captions : list[str]
            Training captions only.
        """

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

        # Add special tokens.
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