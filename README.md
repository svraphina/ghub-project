## NumPy Transformer Decoder

This project contains a small decoder-only Transformer language model written
with NumPy only. It trains on `corpus.txt`, saves weights to
`transformer_decoder_weights.npz` in the project root, and uses `infer.py` for
text generation.

The current small-model setup uses:

- a corpus-built word/whitespace/punctuation tokenizer saved inside the `.npz`
- causal multi-head self-attention
- pre-normalized decoder blocks
- SwiGLU feed-forward layers
- tied token embeddings and output projection
- dropout during training
- AdamW with gradient clipping and cosine learning-rate decay
- optional repetition penalty during generation

Edit training and inference settings in `config.py`.

Train:

```bash
python main.py
```

Generate after training:

```bash
python infer.py
```

There is no `argparse`; the configuration class is the single place to change
tokenizer type, model size, training steps, prompt, sampling temperature, output
length, and checkpoint behavior.
