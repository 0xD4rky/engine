from dataclasses import dataclass
from pathlib import Path
from datasets import load_dataset
import torch
import re
import time
import yaml

from inference.base_decoding import (
    base_decoding_without_kv_cache,
    base_decoding_with_kv_cache,
)
from inference.speculative_decoding import (
    speculative_decoding_without_kv_cache,
    speculative_decoding_with_kv_cache,
)
from utils.processor import GreedyProcessor
from transformers import DynamicCache

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "inference" / "confing.yaml"
config = yaml.load(open(_CONFIG_PATH, "r"), Loader=yaml.FullLoader)

@dataclass
class Metrics:
    ttft: float  # Time to first token (seconds)
    total_latency: float  # Total generation time (seconds)
    tokens_generated: int  # Number of tokens generated
    throughput: float  # Tokens per second


class GSM8KInference:
    def __init__(
        self,
        tokenizer,
        target_model,
        draft_model=None,
        device=None
    ):
        self.device = torch.device(device) if device is not None else next(target_model.parameters()).device

        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token

        self.target_model = target_model.to(self.device)
        self.target_model.eval()

        self.draft_model = None
        if draft_model is not None:
            self.draft_model = draft_model.to(self.device)
            self.draft_model.eval()

        temperature = config["sampling_params"].get("temperature", 1.0)
        self.processor = GreedyProcessor(temperature=temperature)

    def load_gsm8k(self):
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        return dataset

    def format_prompt(self, question: str) -> str:
        # Use Qwen's chat template for proper generation
        messages = [
            {"role": "system", "content": "You are a reasoning large language model developed by Ishaan. Solve problems step by step, thinking thoroughly and end with #### followed by the final numerical answer."},
            {"role": "user", "content": question}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 

    def extract_answer(self, text: str) -> str:
        """
        function to extract numerical answer from generated text
        """

        match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
        if match:
            return match.group(1).replace(',', '')
        
        numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
        if numbers:
            return numbers[-1].replace(',', '')
        return ""
    
    @torch.no_grad()
    def generate_response_with_base_decoding(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        max_new_tokens: int = config["sampling_params"]["max_new_tokens"]
    ):
        input_ids = input_ids.to(self.device, non_blocking=True)
        attn_mask = attn_mask.to(self.device, non_blocking=True)

        start_time = time.time()

        # Prefill phase - compute KV cache and measure TTFT
        cache = DynamicCache()
        outputs = self.target_model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            past_key_values=cache,
            use_cache=True,
        )
        prefill_cache = outputs.past_key_values
        prefill_logits = outputs.logits[:, -1, :]

        first_token_time = time.time()
        ttft = first_token_time - start_time

        # Decode phase - pass prefill outputs to avoid redundant computation
        output_ids = base_decoding_with_kv_cache(
            self.target_model,
            self.tokenizer,
            input_ids,
            attn_mask,
            processor=self.processor,
            max_new_tokens=max_new_tokens,
            past_key_values=prefill_cache,
            prefill_logits=prefill_logits,
        )
        
        end_time = time.time()
        total_latency = end_time - start_time
        tokens_generated = output_ids.shape[1]
        throughput = tokens_generated / total_latency if total_latency > 0 else 0

        metrics = Metrics( 
            ttft=ttft,
            total_latency=total_latency,
            tokens_generated=tokens_generated,
            throughput=throughput
        )

        return output_ids, metrics
    
    @torch.no_grad()
    def generate_response_with_speculative_decoding(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        max_new_tokens: int = config["sampling_params"]["max_new_tokens"],
        gamma: int = config["speculative_params"]["max_speculative_tokens"]
    ):
        if self.draft_model is None:
            raise ValueError("Draft model is required for speculative decoding.")

        input_ids = input_ids.to(self.device, non_blocking=True)
        attn_mask = attn_mask.to(self.device, non_blocking=True)

        start_time = time.time()

        # Prefill both draft and target models, measure TTFT from draft (faster)
        draft_cache = DynamicCache()
        draft_outputs = self.draft_model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            past_key_values=draft_cache,
            use_cache=True,
        )
        draft_prefill_cache = draft_outputs.past_key_values
        draft_prefill_logits = draft_outputs.logits[:, -1, :]

        first_token_time = time.time()
        ttft = first_token_time - start_time

        target_cache = DynamicCache()
        target_outputs = self.target_model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            past_key_values=target_cache,
            use_cache=True,
        )
        target_prefill_cache = target_outputs.past_key_values
        target_prefill_logits = target_outputs.logits[:, -1, :]

        # Decode phase - pass prefill outputs to avoid redundant computation
        output_ids = speculative_decoding_with_kv_cache(
            target_model=self.target_model,
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            attn_mask=attn_mask,
            processor=self.processor,
            max_new_tokens=max_new_tokens,
            gamma=gamma,
            draft_past_key_values=draft_prefill_cache,
            draft_prefill_logits=draft_prefill_logits,
            target_past_key_values=target_prefill_cache,
            target_prefill_logits=target_prefill_logits,
        )

        end_time = time.time()
        total_latency = end_time - start_time
        tokens_generated = output_ids.shape[1]
        throughput = tokens_generated / total_latency if total_latency > 0 else 0

        metrics = Metrics(
            ttft=ttft,
            total_latency=total_latency,
            tokens_generated=tokens_generated,
            throughput=throughput
        )

        return output_ids, metrics

    
