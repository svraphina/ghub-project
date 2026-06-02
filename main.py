import math
import time

import numpy as np

from config import Config
from numpy_transformer import AdamW, NumpyTransformerLM, estimate_loss, get_batch, load_corpus, split_data


def count_parameters(params):
    return sum(param.size for param in params.values())


def learning_rate_for_step(step, cfg):
    if step <= cfg.warmup_iters:
        return cfg.learning_rate * step / max(1, cfg.warmup_iters)

    decay_steps = max(1, cfg.max_iters - cfg.warmup_iters)
    progress = min(1.0, (step - cfg.warmup_iters) / decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_learning_rate + cosine * (cfg.learning_rate - cfg.min_learning_rate)


def main():
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)

    data, tokenizer = load_corpus(cfg.corpus_path, cfg)
    cfg.vocab_size = tokenizer.vocab_size
    train_data, val_data = split_data(data, cfg.train_split)

    model = NumpyTransformerLM(cfg)
    optimizer = AdamW(
        model.params,
        learning_rate=cfg.learning_rate,
        beta1=cfg.beta1,
        beta2=cfg.beta2,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip,
    )

    print("Training NumPy decoder-only Transformer")
    print(
        f"corpus tokens: {len(data)} | train: {len(train_data)} | "
        f"val: {len(val_data)} | vocab: {cfg.vocab_size}"
    )
    print(f"parameters: {count_parameters(model.params):,}")
    print(f"weights will save to: {cfg.weights_path}")

    started = time.time()
    best_loss = float("inf")
    best_step = 0
    best_params = None
    for step in range(1, cfg.max_iters + 1):
        optimizer.learning_rate = learning_rate_for_step(step, cfg)
        xb, yb = get_batch(train_data, cfg.batch_size, cfg.block_size, rng)
        loss, grads = model.loss_and_grads(xb, yb)
        grad_norm = optimizer.step(grads)

        if step == 1 or step % cfg.eval_interval == 0 or step == cfg.max_iters:
            losses = estimate_loss(model, train_data, val_data, cfg, rng)
            val_text = f" | val loss {losses['val']:.4f}" if "val" in losses else ""
            monitor_loss = losses.get("val", losses["train"])
            if monitor_loss < best_loss:
                best_loss = monitor_loss
                best_step = step
                best_params = {name: param.copy() for name, param in model.params.items()}
            elapsed = time.time() - started
            print(
                f"step {step:4d}/{cfg.max_iters} | "
                f"batch loss {loss:.4f} | train loss {losses['train']:.4f}"
                f"{val_text} | lr {optimizer.learning_rate:.6f} | "
                f"grad norm {grad_norm:.3f} | {elapsed:.1f}s"
            )

    if best_params is not None and cfg.save_best_checkpoint:
        for name, param in model.params.items():
            param[...] = best_params[name]
        print(f"restored best checkpoint from step {best_step} with loss {best_loss:.4f}")
    elif best_step:
        print(f"best validation checkpoint was step {best_step} with loss {best_loss:.4f}")

    model.save(cfg.weights_path, tokenizer)
    print(f"saved weights: {cfg.weights_path}")

    prompt_ids = tokenizer.encode(cfg.inference_prompt)
    sample_ids = model.generate(
        prompt_ids,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        seed=cfg.seed + 1,
        repetition_penalty=cfg.repetition_penalty,
        repetition_window=cfg.repetition_window,
    )
    print("\nSample:")
    print(tokenizer.decode(sample_ids))


if __name__ == "__main__":
    main()
