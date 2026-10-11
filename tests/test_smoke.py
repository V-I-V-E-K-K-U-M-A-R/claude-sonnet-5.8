import argparse

import numpy as np

from claude_sonnet.generate import generate_text
from claude_sonnet.model import MiniSonnet
from claude_sonnet.tokenizer import SimpleTokenizer


DEFAULT_TEXT = """The future of AI is exciting and full of possibility. We build tools that help people think, create, and solve problems. Every new model is a step toward better reasoning and more helpful systems."""


def build_demo_model():
    tokenizer = SimpleTokenizer().fit(DEFAULT_TEXT)
    model = MiniSonnet(vocab_size=len(tokenizer), dim=32, n_layers=2, n_heads=4, hidden_dim=64, context_window=32)
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser(description="Generate text from a tiny transformer model.")
    parser.add_argument("--prompt", type=str, default="The future of AI is", help="Prompt to begin generation from.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    tokenizer, model = build_demo_model()
    output = generate_text(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    print(output)


if __name__ == "__main__":
    main()
