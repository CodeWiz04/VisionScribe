from collections import Counter
from pathlib import Path
import string
import re

import pandas as pd

class Vocabulary:
    def __init__(self,min_frequency:int=2):
        self.min_frequency=min_frequency
        self.pad_token="<pad>"
        self.end_token="<end>"
        self.unk_token = "<unk>"
        
        self.word_to_index = {
            self.pad_token: 0,
            self.start_token: 1,
            self.end_token: 2,
            self.unk_token: 3,
        }

        self.index_to_word = {
            0: self.pad_token,
            1: self.start_token,
            2: self.end_token,
            3: self.unk_token,
        }
        
    def tokenize(self, caption: str) -> list[str]:
       caption = caption.lower()

       for symbol in string.punctuation:
         caption = caption.replace(symbol, "")

       return caption.split()
    def build(self, captions: list[str]) -> None:
        word_counts = Counter()
        for caption in captions:
           words = self.tokenize(caption)
           word_counts.update(words)
        counter = Counter()

        for word, frequency in counter.items():

            if frequency >= self.min_frequency:
                index = len(self.word_to_index)
                self.word_to_index[word] = index
                self.index_to_word[index] = word
                