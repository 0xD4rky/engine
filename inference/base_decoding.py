from pathlib import Path
import torch
import yaml
from utils.processor import Processor, GreedyProcessor
from transformers import DynamicCache 

_CONFIG_PATH = Path(__file__).resolve().parent / "inference" / "confing.yaml"
config = yaml.load(open(_CONFIG_PATH, "r"), Loader=yaml.FullLoader)

@torch.no_grad()
def base_decoding_without_kv_cache(
    model,
    tokenizer,
    input_ids: torch.Tensor, # tokenized input ids
    attn_mask: torch.Tensor, 
    processor: Processor = GreedyProcessor(),
    max_new_tokens: int = config["sampling_params"]["max_new_tokens"]
) -> torch.Tensor:

    batch_size = input_ids.shape[0]
    eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None
    
    current_sequence = input_ids.clone()
    current_mask = attn_mask.clone()
    generated_ids = torch.empty((batch_size, 0), dtype=torch.long, device=model.device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=model.device)

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=current_sequence, 
            attention_mask=current_mask, 
            output_hidden_states=False,
            output_attentions=False
        )
        logits = outputs.logits[:, -1, :]
        probs = processor(logits)
        next_token = processor.sample(probs)

        current_sequence = torch.cat([current_sequence, next_token], dim=-1)
        current_mask = torch.cat([current_mask, torch.ones((batch_size, 1), device=model.device)], dim=-1)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        if eos_token_id is not None:
            finished |= (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break

    return generated_ids

@torch.no_grad()
def base_decoding_with_kv_cache(
    model,
    tokenizer,
    input_ids: torch.Tensor, # tokenized input ids
    attn_mask: torch.Tensor, 
    processor: Processor = GreedyProcessor(),
    max_new_tokens: int = config["sampling_params"]["max_new_tokens"]
) -> torch.Tensor:  

    cache = DynamicCache()

    batch_size = input_ids.shape[0]
    eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None

    # Prefill phase: processing the entire input seq and storing kvs
    outputs = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        past_key_values=cache,
        use_cache=True,
        output_hidden_states=False,
        output_attentions=False
    )
    cache = outputs.past_key_values

    logits = outputs.logits[:,-1,:] 
    probs = processor(logits)
    next_token = processor.sample(probs) # we need this last token only

    generated_ids = next_token.clone()
    current_mask = torch.cat([attn_mask, torch.ones((batch_size, 1), device=model.device)], dim=-1)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=model.device)

    if eos_token_id is not None:
        finished |= (next_token.squeeze(-1) == eos_token_id)

    #Decode phase
    for _ in range(max(0, max_new_tokens - 1)):
        if finished.all():
            break

        outputs = model(
            input_ids=next_token,              # Only the NEW token!
            attention_mask=current_mask,       # Full mask (tells model about all positions)
            past_key_values=cache,   # Pass the cache
            use_cache=True,                    # Continue caching
            output_hidden_states=False,
            output_attentions=False
        )
        cache = outputs.past_key_values

        logits = outputs.logits[:,-1,:] 
        probs = processor(logits)
        next_token = processor.sample(probs)
        
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        current_mask = torch.cat([current_mask, torch.ones((batch_size, 1), device=model.device)], dim=-1)

        if eos_token_id is not None:
              finished |= (next_token.squeeze(-1) == eos_token_id)
    
    return generated_ids








        




        







