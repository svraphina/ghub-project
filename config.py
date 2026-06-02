from pathlib import Path


class Config:
    root_dir = Path(__file__).resolve().parent
    corpus_path = root_dir / "corpus.txt"
    weights_path = root_dir / "transformer_decoder_weights.npz"

    seed = 1337
    tokenizer_type = "word"
    vocab_size = 0
    block_size = 48
    batch_size = 8

    n_layers = 2
    n_heads = 4
    d_model = 64
    ff_hidden = 128
    layer_norm_eps = 1e-5
    dropout = 0.1

    max_iters = 1500
    eval_interval = 150
    eval_batches = 4
    learning_rate = 6e-4
    min_learning_rate = 1e-4
    warmup_iters = 50
    weight_decay = 1e-2
    beta1 = 0.9
    beta2 = 0.999
    adam_eps = 1e-8
    grad_clip = 1.0
    train_split = 0.9
    save_best_checkpoint = False

    inference_prompt = "The little transformer"
    max_new_tokens = 80
    temperature = 0.6
    top_k = 12
    repetition_penalty = 0.8
    repetition_window = 16
