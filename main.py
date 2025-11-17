from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.gsm8k import GSM8KInference


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_CONFIG_PATH = Path(__file__).resolve().parent / "inference" / "confing.yaml"
config = yaml.load(open(_CONFIG_PATH, "r"), Loader=yaml.FullLoader)

DEVICE = get_device()

tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

target_model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    torch_dtype=torch.float16
).to(DEVICE)
target_model.eval()

draft_model_name = config.get("speculative_params", {}).get("draft_model_name", config["model_name"])
if draft_model_name == config["model_name"]:
    draft_model = target_model
else:
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name,
        torch_dtype=torch.float16
    ).to(DEVICE)
    draft_model.eval()

gsm8k_inference = GSM8KInference(
    tokenizer=tokenizer,
    target_model=target_model,
    draft_model=draft_model,
    device=DEVICE
)

dataset = list(gsm8k_inference.load_gsm8k())

BATCH_SIZE = config["batch_params"]["batch_size"]
total_batches = (len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE


def chunked(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def _summarize_metrics(metrics_log, label: str):
    if not metrics_log:
        print(f"\nno datapoints processed ({label})")
        return
    print(f"\nall {len(metrics_log)} datapoints done ({label})")
    print(f"avg ttft: {np.mean([m['ttft'] for m in metrics_log]):.3f}s")
    print(f"avg latency: {np.mean([m['total_latency'] for m in metrics_log]):.3f}s")
    print(f"avg throughput: {np.mean([m['throughput'] for m in metrics_log]):.2f} tok/s")


def base_decoding_benchmark():
    results = []
    metrics_log = []

    for batch in tqdm(chunked(dataset, BATCH_SIZE), total=total_batches, desc="Base decoding"):
        prompts = [gsm8k_inference.format_prompt(example["question"]) for example in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True
        )

        input_ids = encoded["input_ids"].to(DEVICE, non_blocking=True)
        attn_mask = encoded["attention_mask"].to(DEVICE, non_blocking=True)

        output_ids, metrics = gsm8k_inference.generate_response_with_base_decoding(input_ids, attn_mask)
        responses = tokenizer.batch_decode(output_ids.cpu(), skip_special_tokens=True)

        for example, response in zip(batch, responses):
            results.append({
                "question": example["question"],
                "answer": example["answer"],
                "generated": response
            })

        metrics_log.append({
            "ttft": metrics.ttft,
            "total_latency": metrics.total_latency,
            "tokens_generated": metrics.tokens_generated,
            "throughput": metrics.throughput
        })

    _summarize_metrics(metrics_log, "base decoding")
    return results, metrics_log


def speculative_decoding_benchmark():
    results = []
    metrics_log = []

    for batch in tqdm(chunked(dataset, BATCH_SIZE), total=total_batches, desc="Speculative decoding"):
        prompts = [gsm8k_inference.format_prompt(example["question"]) for example in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True
        )

        input_ids = encoded["input_ids"].to(DEVICE, non_blocking=True)
        attn_mask = encoded["attention_mask"].to(DEVICE, non_blocking=True)

        output_ids, metrics = gsm8k_inference.generate_response_with_speculative_decoding(input_ids, attn_mask)
        responses = tokenizer.batch_decode(output_ids.cpu(), skip_special_tokens=True)

        for example, response in zip(batch, responses):
            results.append({
                "question": example["question"],
                "answer": example["answer"],
                "generated": response
            })

        metrics_log.append({
            "ttft": metrics.ttft,
            "total_latency": metrics.total_latency,
            "tokens_generated": metrics.tokens_generated,
            "throughput": metrics.throughput
        })

    _summarize_metrics(metrics_log, "speculative decoding")
    return results, metrics_log
