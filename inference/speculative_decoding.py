import torch
import yaml
from utils.processor import Processor, GreedyProcessor

config = yaml.load(open("engine/inference/confing.yaml"), Loader=yaml.FullLoader)

def _true_lengths(attn_mask: torch.Tensor) -> torch.Tensor:
    return attn_mask.sum(dim=1) # true sequence lengths from an attention mask (1 for tokens, 0 for pad)

def _append_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, y], dim=1) # concatenate along sequence dimension (dim=1)

@torch.no_grad()
def speculative_decoding_without_kv_cache(
    target_model,
    draft_model,
    tokenizer,
    input_ids: torch.Tensor, # tokenized input ids
    attn_mask: torch.Tensor,
    processor: Processor = GreedyProcessor(),
    max_new_tokens: int = config["sampling_params"]["max_new_tokens"],
    gamma: int = config["speculative_params"]["max_speculative_tokens"]
) -> torch.Tensor:

    batch_size = input_ids.shape[0]
    device = next(target_model.parameters()).device

    eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id


    current_sequence = input_ids.clone().to(device)
    current_mask = attn_mask.clone().to(device)
    generated_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    if pad_token_id is None:
        pad_token_id = 0 # safe fall-back when tokenizer lacks pad/eos information

    mask_is_bool = current_mask.dtype == torch.bool

    while generated_ids.shape[1] < max_new_tokens and not finished.all():
        prev_len = _true_lengths(current_mask).long()
        active = ~finished
        if not active.any():
            break

        remaining = max_new_tokens - generated_ids.shape[1]
        if remaining <= 0:
            break

        step_budget = min(gamma, remaining)

        draft_columns = []
        draft_prob_columns = []

        speculative_ids = current_sequence
        speculative_mask = current_mask

        for _ in range(step_budget):
            if not active.any():
                break

            draft_outputs = draft_model(input_ids=speculative_ids, attention_mask=speculative_mask)
            draft_logits = draft_outputs.logits[:, -1, :]
            draft_probs = processor(draft_logits)
            next_token = processor.sample(draft_probs).squeeze(-1)
            next_prob = draft_probs.gather(-1, next_token.unsqueeze(-1)).squeeze(-1)

            if not active.all():
                next_token = next_token.clone()
                next_prob = next_prob.clone()
                next_token[~active] = pad_token_id
                next_prob[~active] = 1.0

            draft_columns.append(next_token.unsqueeze(1))
            draft_prob_columns.append(next_prob.unsqueeze(1))

            new_mask = torch.zeros((batch_size, 1), dtype=current_mask.dtype, device=device)
            if mask_is_bool:
                new_mask[active] = True
            else:
                new_mask[active] = 1

            speculative_ids = _append_rows(speculative_ids, next_token.unsqueeze(1))
            speculative_mask = _append_rows(speculative_mask, new_mask)

        if not draft_columns:
            break

        draft_block = torch.cat(draft_columns, dim=1)
        draft_block_probs = torch.cat(draft_prob_columns, dim=1).clamp_min(1e-8)
        block_len = draft_block.shape[1]

        target_outputs = target_model(input_ids=speculative_ids, attention_mask=speculative_mask)
        target_logits = target_outputs.logits

        batch_idx = torch.arange(batch_size, device=device)
        alive = active.clone()

        for step in range(block_len):
            if not alive.any() or generated_ids.shape[1] >= max_new_tokens:
                break

            positions = (prev_len + step - 1).clamp(min=0)
            logits_step = target_logits[batch_idx, positions, :]
            target_probs = processor(logits_step)

            candidate_tokens = draft_block[:, step]
            candidate_probs = target_probs.gather(-1, candidate_tokens.unsqueeze(-1)).squeeze(-1)
            draft_probs_sel = draft_block_probs[:, step]

            acceptance = torch.minimum(torch.ones_like(candidate_probs), candidate_probs / draft_probs_sel)
            acceptance = torch.where(alive, acceptance, torch.zeros_like(acceptance))

            accept_mask = (torch.rand_like(acceptance) <= acceptance) & alive
            reject_mask = alive & ~accept_mask

            token_column = torch.full((batch_size,), pad_token_id, dtype=torch.long, device=device)
            mask_column = torch.zeros((batch_size,), dtype=current_mask.dtype, device=device)

            if accept_mask.any():
                token_column[accept_mask] = candidate_tokens[accept_mask]
                if mask_is_bool:
                    mask_column[accept_mask] = True
                else:
                    mask_column[accept_mask] = 1

            if reject_mask.any():
                fallback_probs = target_probs.clone()
                reject_idx = torch.nonzero(reject_mask, as_tuple=False).squeeze(-1)
                if reject_idx.numel() > 0:
                    fallback_probs[reject_idx, candidate_tokens[reject_idx]] = 0
                    slice_probs = fallback_probs[reject_idx]
                    sums = slice_probs.sum(dim=-1, keepdim=True)
                    zero_mass = sums.squeeze(-1) <= 1e-8
                    if zero_mass.any():
                        slice_probs[zero_mass] = target_probs[reject_idx][zero_mass]
                        sums = slice_probs.sum(dim=-1, keepdim=True)
                    slice_probs = slice_probs / sums.clamp_min(1e-8)
                    fallback_probs[reject_idx] = slice_probs

                fallback_samples = processor.sample(fallback_probs).squeeze(-1)
                token_column[reject_mask] = fallback_samples[reject_mask]
                if mask_is_bool:
                    mask_column[reject_mask] = True
                else:
                    mask_column[reject_mask] = 1

            if mask_column.any():
                token_column_unsq = token_column.unsqueeze(1)
                mask_column_unsq = mask_column.unsqueeze(1)
                current_sequence = _append_rows(current_sequence, token_column_unsq)
                current_mask = _append_rows(current_mask, mask_column_unsq)
                generated_ids = _append_rows(generated_ids, token_column_unsq)

                if eos_token_id is not None:
                    mask_bool = mask_column if mask_is_bool else mask_column.bool()
                    eos_mask = (token_column == eos_token_id) & mask_bool
                    finished |= eos_mask

            alive = accept_mask.clone()
            if eos_token_id is not None:
                alive &= token_column != eos_token_id

        if generated_ids.shape[1] >= max_new_tokens or not alive.any():
            if generated_ids.shape[1] >= max_new_tokens:
                break
        else:
            positions = (prev_len + block_len - 1).clamp(min=0)
            logits_next = target_logits[batch_idx, positions, :]
            probs_next = processor(logits_next)
            next_token = processor.sample(probs_next).squeeze(-1)

            token_column = torch.full((batch_size,), pad_token_id, dtype=torch.long, device=device)
            mask_column = torch.zeros((batch_size,), dtype=current_mask.dtype, device=device)
            token_column[alive] = next_token[alive]
            if mask_is_bool:
                mask_column[alive] = True
            else:
                mask_column[alive] = 1

            token_column_unsq = token_column.unsqueeze(1)
            mask_column_unsq = mask_column.unsqueeze(1)
            current_sequence = _append_rows(current_sequence, token_column_unsq)
            current_mask = _append_rows(current_mask, mask_column_unsq)
            generated_ids = _append_rows(generated_ids, token_column_unsq)

            if eos_token_id is not None:
                mask_bool = mask_column if mask_is_bool else mask_column.bool()
                eos_mask = (token_column == eos_token_id) & mask_bool
                finished |= eos_mask

    return generated_ids
