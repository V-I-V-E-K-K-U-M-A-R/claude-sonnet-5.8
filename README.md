# Claude Sonnet 5.8

This repository contains a compact, from-scratch transformer implementation inspired by modern decoder-only language models. It is intentionally small, readable, and educational.

Important: this is not an official Anthropic model, training checkpoint, or proprietary codebase. It is a clean local implementation designed to demonstrate how a miniature LLM can be built from the ground up.

## Features

- Simple tokenizer
- Decoder-only transformer block
- Causal self-attention
- Feed-forward network
- Text generation loop
- CLI entry point
- Smoke tests

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
claude-sonnet-5.8 --prompt "Once upon a time" --max-new-tokens 32
```

## Example

```bash
claude-sonnet-5.8 --prompt "The future of AI is" --max-new-tokens 40
```

## Notes

This project focuses on architecture and code clarity rather than giant-scale training or model weights. It is a good starting point for experimentation, learning, and extension.
