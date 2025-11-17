from typing import Any

import torch
import numpy as np
import yaml
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

from benchmark.data import GSM8KInference

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

config = yaml.load(open("/Users/ishaankumar/Documents/engine/inference/confing.yaml"), Loader=yaml.FullLoader)

model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    torch_dtype=torch.float16
).to(get_device())

tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

gsm8k_inference = GSM8KInference(model_name=config["model_name"])

dataset = list(gsm8k_inference.load_gsm8k())

BATCH_SIZE = config["batch_params"]["batch_size"]
NUM_WORKERS = config["batch_params"]["num_workers"]
PIN_MEMORY = config["batch_params"]["pin_memory"]

def chunked(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

all_results = []
all_metrics = []

total_batches = (len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE

def base_decoding_benchmark():
    for batch in tqdm(chunked(dataset, BATCH_SIZE), total=total_batches, desc="Processing batches"):
        prompts = [gsm8k_inference.format_prompt(example["question"]) for example in batch]
        encoded_prompts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)

        input_ids = encoded_prompts["input_ids"].to(get_device())
        attn_mask = encoded_prompts["attention_mask"].to(get_device())

        output_ids, metrics = gsm8k_inference.generate_response_with_base_decoding(input_ids, attn_mask)

        responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        for example, response in zip(batch, responses):
            result = {
                "question": example["question"],
                "answer": example["answer"],
                "generated": response
            }
            all_results.append(result)
        
        all_metrics.append({
            "ttft": metrics.ttft,
            "total_latency": metrics.total_latency,
            "tokens_generated": metrics.tokens_generated,
            "throughput": metrics.throughput
        })
        

    print(f"\nall {len(all_results)} datapoints done")
    print(f"avg ttft: {np.mean([m['ttft'] for m in all_metrics]):.3f}s")
    print(f"avg latency: {np.mean([m['total_latency'] for m in all_metrics]):.3f}s")
    print(f"avg throughput: {np.mean([m['throughput'] for m in all_metrics]):.2f} tok/s")

def speculative_decoding_benchmark():
    pass

