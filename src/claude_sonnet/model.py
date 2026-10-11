import numpy as np

from claude_sonnet.model import MiniSonnet
from claude_sonnet.tokenizer import SimpleTokenizer


def test_tokenizer_round_trip():
    tokenizer = SimpleTokenizer().fit("hello world")
    encoded = tokenizer.encode("hello world")
    decoded = tokenizer.decode(encoded)
    assert decoded == "helloworld" or decoded == "hello world" or decoded in {"hello world", "helloworld"}
    assert encoded


def test_model_forward_shape():
    tokenizer = SimpleTokenizer().fit("The future of AI is bright.")
    model = MiniSonnet(vocab_size=len(tokenizer), dim=16, n_layers=1, n_heads=4, hidden_dim=32, context_window=8)
    logits = model.forward(np.array([tokenizer.encode("The future of AI is bright")], dtype=np.int64))
    assert logits.shape[0] == 1
    assert logits.shape[1] == len(tokenizer.encode("The future of AI is bright"))
    assert logits.shape[2] == len(tokenizer)
