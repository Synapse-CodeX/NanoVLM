from collections import Counter


class SimpleTokenizer:
    def __init__(self):
        self.special_tokens = {
            "<PAD>": 0,
            "<BOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
        }

        self.vocab = self.special_tokens.copy()
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def build_vocab(self, texts):
        counter = Counter()

        for text in texts:
            words = text.lower().split()
            counter.update(words)

        for word in counter:
            if word not in self.vocab:
                idx = len(self.vocab)
                self.vocab[word] = idx
                self.reverse_vocab[idx] = word

        self.vocab_size = len(self.vocab)

    def encode(self, text):
        words = text.lower().split()

        token_ids = [self.vocab["<BOS>"]]

        for word in words:
            token_ids.append(
                self.vocab.get(word, self.vocab["<UNK>"])
            )

        token_ids.append(self.vocab["<EOS>"])

        return token_ids

    def decode(self, token_ids):
        words = []

        for idx in token_ids:
            token = self.reverse_vocab.get(idx, "<UNK>")

            if token in ["<PAD>", "<BOS>", "<EOS>"]:
                continue

            words.append(token)

        return " ".join(words)

    def pad_sequences(self, sequences):
        max_length = max(len(seq) for seq in sequences)

        padded = []

        for seq in sequences:
            padding = [self.vocab["<PAD>"]] * (max_length - len(seq))
            padded.append(seq + padding)

        return padded