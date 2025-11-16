import torch
import yaml
from utils.processor import Processor, GreedyProcessor

config = yaml.load(open("inference/config.yaml"), Loader=yaml.FullLoader)

@torch.no_grad()
def base_decoding_without_kv_cache(
    model,
    tokenizer,
    input_ids: torch.Tensor,
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
        outputs = model(input_ids=current_sequence, attention_mask=current_mask)
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







        




        







