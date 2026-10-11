import re


class SimpleTokenizer:
    """A tiny tokenizer that keeps the implementation readable and reliable."""

    def __init__(self):
        self.pattern = re.compile(r"\s+|[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]")
        self.vocab = {}
        self.inv_vocab = {}
        self._built = False

    def fit(self, text: str):
        self.vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        seen = set(self.vocab)
        for token in self.pattern.findall(text):
            if token not in seen:
                self.vocab[token] = len(self.vocab)
                seen.add(token)
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}
        self._built = True
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab.get("<pad>", 0)

    @property
    def eos_id(self) -> int:
        return self.vocab.get("<eos>", 3)

    def encode(self, text: str):
        if not self._built:
            self.fit(text)
        ids = []
        for token in self.pattern.findall(text):
            ids.append(self.vocab.get(token, self.vocab["<unk>"]))
        return ids

    def decode(self, ids):
        tokens = []
        for idx in ids:
            tokens.append(self.inv_vocab.get(idx, "<unk>"))
        return "".join(tokens)

    def __len__(self):
        return self.vocab_size
