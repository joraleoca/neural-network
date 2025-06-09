import pytest

from src.preprocessing import Tokenizer


@pytest.fixture()
def vocab() -> list[str]:
    return ["Hello", "world", "this", "is", "vocab", "to", "test", "the", "tokenizer"]


def test_tokenize(vocab: list[str]) -> None:
    tokenizer = Tokenizer(vocab)

    test = vocab[0 : len(vocab) : 3]
    result = tokenizer(test)

    assert test == tokenizer.decode(result), (
        f"The tokenizer should give the same input when decoding. Got {tokenizer.decode(result)}"
    )
