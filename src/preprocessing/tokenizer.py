from typing import ClassVar

from src.tensor import Tensor


class Tokenizer:
    __slots__ = "vocab", "word2idx"

    PAD: ClassVar[str] = "<pad>"
    SOS: ClassVar[str] = "<sos>"
    EOS: ClassVar[str] = "<eos>"
    UNK: ClassVar[str] = "<unk>"

    def __init__(self, vocab: list[str]) -> None:
        self.vocab = [self.PAD, self.SOS, self.EOS, self.UNK] + vocab
        self.word2idx = {word: i for i, word in enumerate(self.vocab)}

    def __call__(
        self, words: list[str], max_size: int = -1, add_sos: bool = False, add_eos: bool = True
    ) -> Tensor[int]:
        if len(words) > max_size:
            raise ValueError(f"The number of words exceeds the maximum size. Got {len(words)=}, {max_size=}")

        max_size += 1
        tokens = []

        if add_sos:
            tokens.append(self.word2idx[self.SOS])

        unk_token = self.word2idx[self.UNK]
        tokens.extend(self.word2idx.get(w, unk_token) for w in words)

        if add_eos:
            tokens.append(self.word2idx[self.EOS])

        if len(tokens) < max_size:
            tokens += [self.word2idx[self.PAD]] * (max_size - len(tokens))

        return Tensor(tokens)

    def decode(self, indices: Tensor[int]) -> list[str]:
        decoded = []

        for index in indices:
            if 0 <= (i := index.item()) < len(self.vocab):
                word = self.vocab[i]
            else:
                word = self.UNK

            decoded.append(word)

        return decoded
