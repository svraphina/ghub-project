from config import Config
from numpy_transformer import NumpyTransformerLM, load_tokenizer


def main():
    cfg = Config()
    model = NumpyTransformerLM.load(cfg.weights_path, cfg)
    tokenizer = load_tokenizer(cfg.weights_path)

    prompt = input(f"Prompt [{cfg.inference_prompt}]: ").strip()
    if not prompt:
        prompt = cfg.inference_prompt

    start_ids = tokenizer.encode(prompt)
    generated = model.generate(
        start_ids,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        seed=cfg.seed + 2,
        repetition_penalty=cfg.repetition_penalty,
        repetition_window=cfg.repetition_window,
    )
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()
