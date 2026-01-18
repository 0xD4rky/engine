from pathlib import Path
import torch
import yaml
from utils.processor import Processor, GreedyProcessor
from transformers import DynamicCache 

_CONFIG_PATH = Path(__file__).resolve().parent / "confing.yaml"
config = yaml.load(open(_CONFIG_PATH, "r"), Loader=yaml.FullLoader)

def _true_lengths(attn_mask: torch.Tensor) -> torch.Tensor:
    return attn_mask.sum(dim=1) # true sequence lengths from an attention mask (1 for tokens, 0 for pad)

def _append_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, y], dim=1) # concatenate along sequence dimension (dim=1)

def _clone_dynamic_cache(cache: DynamicCache) -> DynamicCache:
    legacy = cache.to_legacy_cache()
    cloned_legacy = tuple((k.clone(), v.clone()) for k, v in legacy)
    return DynamicCache.from_legacy_cache(cloned_legacy)

def _truncate_cache(cache: DynamicCache, seq_len: int) -> DynamicCache:
    for layer_idx in range(len(cache.key_cache)):
        cache.key_cache[layer_idx] = cache.key_cache[layer_idx][:, :, :seq_len, :]
        cache.value_cache[layer_idx] = cache.value_cache[layer_idx][:, :, :seq_len, :]
    cache._seen_tokens = seq_len
    return cache

def _get_cache_seq_len(cache: DynamicCache) -> int:
    if len(cache.key_cache) == 0:
        return 0
    return cache.key_cache[0].shape[2]

def _multi_token_forward(model, tokens, attention_mask, cache):
    outputs = model(
        input_ids=tokens,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
        output_hidden_states=False,
        output_attentions=False,
    )
    return outputs.past_key_values, outputs.logits

def _forward_step(
    model,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: DynamicCache,
) -> tuple[DynamicCache, torch.Tensor]:
    outputs = model(
        input_ids=tokens,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
        output_hidden_states=False,
        output_attentions=False,
    )
    return outputs.past_key_values, outputs.logits[:, -1, :]

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

        draft_model.eval()
        target_model.eval()

        for _ in range(step_budget):
            if not active.any():
                break
            
            draft_outputs = draft_model(
                input_ids=speculative_ids, 
                attention_mask=speculative_mask, 
                output_hidden_states=False,
                output_attentions=False
            )
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

        target_outputs = target_model(
            input_ids=speculative_ids, 
            attention_mask=speculative_mask,
            output_hidden_states=False,
            output_attentions=False
        )
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

@torch.no_grad()
def speculative_decoding_with_kv_cache(
    target_model,
    draft_model,
    tokenizer,
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    processor: Processor = GreedyProcessor(),
    max_new_tokens: int = config["sampling_params"]["max_new_tokens"],
    gamma: int = config["speculative_params"]["max_speculative_tokens"],
    draft_past_key_values: DynamicCache = None,
    draft_prefill_logits: torch.Tensor = None,
    target_past_key_values: DynamicCache = None,
    target_prefill_logits: torch.Tensor = None,
) -> torch.Tensor:
    batch_size = input_ids.shape[0]
    device = next(target_model.parameters()).device

    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)

    draft_model.eval()
    target_model.eval()

    eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    current_sequence = input_ids.clone()
    current_mask = attn_mask.clone()
    generated_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    mask_is_bool = current_mask.dtype == torch.bool

    if draft_past_key_values is not None and draft_prefill_logits is not None:
        draft_cache = draft_past_key_values
        draft_logits_next = draft_prefill_logits
    else:
        draft_cache = DynamicCache()
        draft_outputs = draft_model(
            input_ids=current_sequence,
            attention_mask=current_mask,
            past_key_values=draft_cache,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=False,
        )
        draft_cache = draft_outputs.past_key_values
        draft_logits_next = draft_outputs.logits[:, -1, :]

    if target_past_key_values is not None and target_prefill_logits is not None:
        target_cache = target_past_key_values
        target_logits_next = target_prefill_logits
    else:
        target_cache = DynamicCache()
        target_outputs = target_model(
            input_ids=current_sequence,
            attention_mask=current_mask,
            past_key_values=target_cache,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=False,
        )
        target_cache = target_outputs.past_key_values
        target_logits_next = target_outputs.logits[:, -1, :]

    if eos_token_id is not None:
        prompt_lengths = _true_lengths(current_mask).long() - 1
        prompt_lengths = prompt_lengths.clamp(min=0)
        eos_in_prompt = current_sequence.gather(1, prompt_lengths.unsqueeze(-1)).squeeze(-1) == eos_token_id
        finished |= eos_in_prompt

    while generated_ids.shape[1] < max_new_tokens and not finished.all():
        active = ~finished
        if not active.any():
            break

        remaining = max_new_tokens - generated_ids.shape[1]
        if remaining <= 0:
            break

        step_budget = min(gamma, remaining)

        speculative_cache = _clone_dynamic_cache(draft_cache)
        speculative_mask = current_mask.clone()
        draft_tokens_list = []
        draft_probs_list = []
        draft_logits_step = draft_logits_next

        for _ in range(step_budget):
            if not active.any():
                break

            draft_probs = processor(draft_logits_step)
            next_token = processor.sample(draft_probs).squeeze(-1)
            next_prob = draft_probs.gather(-1, next_token.unsqueeze(-1)).squeeze(-1)

            if not active.all():
                next_token = next_token.clone()
                next_prob = next_prob.clone()
                next_token[~active] = pad_token_id
                next_prob[~active] = 1.0

            draft_tokens_list.append(next_token)
            draft_probs_list.append(next_prob)

            new_mask = torch.zeros((batch_size, 1), dtype=current_mask.dtype, device=device)
            if mask_is_bool:
                new_mask[active] = True
            else:
                new_mask[active] = 1
            speculative_mask = _append_rows(speculative_mask, new_mask)

            speculative_cache, draft_logits_step = _forward_step(
                draft_model,
                next_token.unsqueeze(1),
                speculative_mask,
                speculative_cache,
            )

        if not draft_tokens_list:
            break

        draft_block = torch.stack(draft_tokens_list, dim=1)
        draft_block_probs = torch.stack(draft_probs_list, dim=1).clamp_min(1e-8)
        block_len = draft_block.shape[1]

        verify_mask_extension = torch.zeros((batch_size, block_len), dtype=current_mask.dtype, device=device)
        if mask_is_bool:
            verify_mask_extension[active] = True
        else:
            verify_mask_extension[active] = 1
        verify_mask = _append_rows(current_mask, verify_mask_extension)

        verify_cache = _clone_dynamic_cache(target_cache)
        verify_cache, verify_logits = _multi_token_forward(
            target_model,
            draft_block,
            verify_mask,
            verify_cache,
        )

        all_verify_logits = torch.cat([
            target_logits_next.unsqueeze(1),
            verify_logits[:, :-1, :]
        ], dim=1)
        bonus_logits = verify_logits[:, -1, :]

        first_reject = torch.full((batch_size,), block_len, dtype=torch.long, device=device)
        fallback_tokens = torch.full((batch_size,), pad_token_id, dtype=torch.long, device=device)
        alive = active.clone()

        for step in range(block_len):
            step_logits = all_verify_logits[:, step, :]
            target_probs = processor(step_logits)

            draft_token = draft_block[:, step]
            draft_prob = draft_block_probs[:, step]

            target_prob_of_draft = target_probs.gather(-1, draft_token.unsqueeze(-1)).squeeze(-1)
            acceptance_ratio = torch.minimum(
                torch.ones_like(target_prob_of_draft),
                target_prob_of_draft / draft_prob
            )
            acceptance_ratio = torch.where(alive, acceptance_ratio, torch.zeros_like(acceptance_ratio))

            rand = torch.rand_like(acceptance_ratio)
            accept_mask = (rand <= acceptance_ratio) & alive
            reject_mask = alive & ~accept_mask

            if reject_mask.any():
                residual_probs = target_probs.clone()
                reject_idx = reject_mask.nonzero(as_tuple=False).squeeze(-1)
                if reject_idx.numel() == 1:
                    reject_idx = reject_idx.unsqueeze(0)
                if reject_idx.numel() > 0:
                    for idx in reject_idx:
                        residual_probs[idx, draft_token[idx]] = 0
                    slice_probs = residual_probs[reject_idx]
                    sums = slice_probs.sum(dim=-1, keepdim=True)
                    zero_mass = sums.squeeze(-1) <= 1e-8
                    if zero_mass.any():
                        slice_probs[zero_mass] = target_probs[reject_idx[zero_mass]]
                        sums = slice_probs.sum(dim=-1, keepdim=True)
                    slice_probs = slice_probs / sums.clamp_min(1e-8)
                    residual_probs[reject_idx] = slice_probs
                    sampled = processor.sample(residual_probs).squeeze(-1)
                    fallback_tokens[reject_mask] = sampled[reject_mask]

                newly_rejected = reject_mask & (first_reject == block_len)
                first_reject[newly_rejected] = step

            alive = accept_mask.clone()
            if eos_token_id is not None:
                alive = alive & (draft_token != eos_token_id)

        fully_accepted_mask = (first_reject == block_len) & active
        bonus_probs = processor(bonus_logits)
        bonus_token = processor.sample(bonus_probs).squeeze(-1)

        tokens_to_add = first_reject.clone()
        tokens_to_add[fully_accepted_mask] = block_len
        total_tokens_per_seq = tokens_to_add + 1

        min_common = tokens_to_add.min().item()

        target_all_logits = None
        draft_all_logits = None

        if min_common > 0:
            common_block = draft_block[:, :min_common]
            common_mask_ext = torch.zeros((batch_size, min_common), dtype=current_mask.dtype, device=device)
            if mask_is_bool:
                common_mask_ext[active] = True
            else:
                common_mask_ext[active] = 1
            current_mask = _append_rows(current_mask, common_mask_ext)
            current_sequence = _append_rows(current_sequence, common_block)
            generated_ids = _append_rows(generated_ids, common_block)

            target_cache, target_all_logits = _multi_token_forward(
                target_model,
                common_block,
                current_mask,
                target_cache,
            )
            draft_cache, draft_all_logits = _multi_token_forward(
                draft_model,
                common_block,
                current_mask,
                draft_cache,
            )

            if eos_token_id is not None:
                for pos in range(min_common):
                    tok = common_block[:, pos]
                    eos_hit = (tok == eos_token_id) & active
                    finished |= eos_hit

        max_total = total_tokens_per_seq.max().item()
        target_logits_step = target_all_logits[:, -1, :] if target_all_logits is not None else target_logits_next
        draft_logits_step = draft_all_logits[:, -1, :] if draft_all_logits is not None else draft_logits_next

        for pos in range(min_common, max_total):
            active_for_pos = (total_tokens_per_seq > pos) & (~finished)
            if not active_for_pos.any():
                break

            token_at_pos = torch.full((batch_size,), pad_token_id, dtype=torch.long, device=device)

            is_last_token = (pos == tokens_to_add)
            use_draft = (pos < tokens_to_add) & active_for_pos
            use_fallback = is_last_token & (~fully_accepted_mask) & active_for_pos
            use_bonus = is_last_token & fully_accepted_mask & active_for_pos

            if use_draft.any():
                draft_indices = use_draft.nonzero(as_tuple=False).squeeze(-1)
                if draft_indices.numel() == 1:
                    draft_indices = draft_indices.unsqueeze(0)
                for idx in draft_indices:
                    if pos < block_len:
                        token_at_pos[idx] = draft_block[idx, pos]

            if use_fallback.any():
                token_at_pos[use_fallback] = fallback_tokens[use_fallback]

            if use_bonus.any():
                token_at_pos[use_bonus] = bonus_token[use_bonus]

            mask_col = torch.zeros((batch_size,), dtype=current_mask.dtype, device=device)
            if mask_is_bool:
                mask_col[active_for_pos] = True
            else:
                mask_col[active_for_pos] = 1

            token_col_unsq = token_at_pos.unsqueeze(1)
            mask_col_unsq = mask_col.unsqueeze(1)

            current_sequence = _append_rows(current_sequence, token_col_unsq)
            current_mask = _append_rows(current_mask, mask_col_unsq)
            generated_ids = _append_rows(generated_ids, token_col_unsq)

            target_cache, target_logits_step = _forward_step(
                target_model,
                token_col_unsq,
                current_mask,
                target_cache,
            )
            draft_cache, draft_logits_step = _forward_step(
                draft_model,
                token_col_unsq,
                current_mask,
                draft_cache,
            )

            if eos_token_id is not None:
                eos_hit = (token_at_pos == eos_token_id) & active_for_pos
                finished |= eos_hit

        target_logits_next = target_logits_step
        draft_logits_next = draft_logits_step

    return generated_ids
