from dataclasses import dataclass
from datasets import load_dataset
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import time
import yaml

from inference.base_decoding import base_decoding_without_kv_cache

config = yaml.load(open("/Users/ishaankumar/Documents/engine/inference/confing.yaml"), Loader=yaml.FullLoader)

@dataclass
class Metrics:
    ttft: float  # Time to first token (seconds)
    total_latency: float  # Total generation time (seconds)
    tokens_generated: int  # Number of tokens generated
    throughput: float  # Tokens per second



class GSM8KInference:

    def __init__(self, model_name: str, device: str = "mps"):

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="mps"
        )
        self.model.eval()
        self.model.to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def load_gsm8k(self):

        dataset = load_dataset("openai/gsm8k", "main", split="test")
        return dataset

    def format_prompt(self, question: str) -> str:
        return f"""Solve this math problem step by step:

Question: {question}

Answer: Let's solve this step by step:
""" 

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
        start_time = time.time()
        
        outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
        first_token_time = time.time()
        ttft = first_token_time - start_time

        output_ids = base_decoding_without_kv_cache(
            self.model,
            self.tokenizer,
            input_ids,
            attn_mask,
            max_new_tokens=max_new_tokens
        )
        
        end_time = time.time()
        total_latency = end_time - start_time
        tokens_generated = output_ids.shape[1]
        throughput = tokens_generated / total_latency if total_latency > 0 else 0

        # for each indv batch
        metrics = Metrics( 
            ttft=ttft,
            total_latency=total_latency,
            tokens_generated=tokens_generated,
            throughput=throughput
        )

        return output_ids, metrics
    
    def generate_response_with_speculative_decoding(
    ):

        

















    


