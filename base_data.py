import re
import string

from tqdm.auto import tqdm
from nltk import wordpunct_tokenize

class BaseData:

    def __init__(self):
        self.tokenize_texts_list: list = []

    @staticmethod
    def clean_text(text):
        text = "".join([ch if ch not in string.punctuation else ' ' for ch in text])
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        return text

    def tokenize_text_list(self, texts):
        for text in tqdm(texts):
            tokens = wordpunct_tokenize(text)
            text = " ".join(tokens)
            self.tokenize_texts_list.append(text)
