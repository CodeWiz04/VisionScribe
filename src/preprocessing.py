from collections import Counter
from pathlib import Path
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
        