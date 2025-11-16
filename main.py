import torch
import numpy as np
import yaml

from transformers import AutoTokenizer, AutoModelForCausalLM

from benchmark.data import GSM8KInference

config = yaml.load(open("inference/config.yaml"), Loader=yaml.FullLoader)

model = AutoModelForCausalLM.from_pretrained(
    "qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="mps"
)

tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

gsm8k_inference = GSM8KInference(model_name=config["model_name"])

dataset = gsm8k_inference.load_gsm8k()

BATCH_SIZE = config["batch_params"]["batch_size"]
NUM_WORKERS = config["batch_params"]["num_workers"]
PIN_MEMORY = config["batch_params"]["pin_memory"]

def chunked(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

all_results = []
all_metrics = []

for batch in chunked(dataset, BATCH_SIZE):
    prompts = [gsm8k_inference.format_prompt(example["question"]) for example in batch]
    encoded_prompts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)

    input_ids = encoded_prompts["input_ids"].to(get_device())
    attn_mask = encoded_prompts["attention_mask"].to(get_device())

    output_ids, metrics = gsm8k_inference.generate_response(input_ids, attn_mask)

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
    

print(f"Processed {len(all_results)} examples")
print(f"Average TTFT: {np.mean(all_metrics['ttft']):.3f}s")
print(f"Average Latency: {np.mean(all_metrics['total_latency']):.3f}s")
print(f"Average Throughput: {np.mean(all_metrics['throughput']):.2f} tok/s")


