import argparse
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb


def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def calculate_DPO_loss(
    model_prefered_logprob,
    model_disprefered_logprob,
    ref_prefered_logprob,
    ref_disprefered_logprob,
    beta=0.5,
):
    """_summary_

    Args:
        model_prefered_logprob (tensor): [B]
        model_disprefered_logprob (tensor): [B]
        ref_prefered_logprob (tensor): [B]
        ref_disprefered_logprob (tensor): [B]
        beta (float): defaults to 0.5.

    Returns:
        _type_: _description_
    """
    # [B]
    prefered_relative_logprob = model_prefered_logprob - ref_prefered_logprob
    # [B]
    disprefered_relative_logprob = model_disprefered_logprob - ref_disprefered_logprob
    # float
    reward_accuracies = (
        (prefered_relative_logprob > disprefered_relative_logprob).float().mean(dim=-1)
    )
    # float
    reward_margins = (prefered_relative_logprob - disprefered_relative_logprob).mean(
        dim=-1
    )

    # float
    loss = -F.logsigmoid(
        beta * (prefered_relative_logprob - disprefered_relative_logprob)
    ).mean(dim=-1)

    return (
        loss,
        prefered_relative_logprob.mean(dim=-1),
        disprefered_relative_logprob.mean(dim=-1),
        reward_accuracies,
        reward_margins,
    )


def get_log_prob(logits, labels, prompt_lengths):
    """_summary_

    Args:
        logits (tensor):         [B, 2*max_length, vocab_size]
        labels (tensor):         [B, 2*max_length]
        prompt_lengths (tensor): [B]
            Note: seq_len = 2*max_length due to the concatenation of prompt and response

    Returns:
        (tensor): [B] The summation of log probs of all responsive tokens for each batch 
    """
    # [B, 2*max_length, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)
    # [B, 2*max_length]
    token_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
    
    batch_size, seq_len = labels.shape
    # Creates a mask for the non-prompt part of the sequence, [B, seq_len]
    response_mask = torch.arange(seq_len, device=labels.device).unsqueeze(0) >= prompt_lengths.unsqueeze(1)
    response_mask = response_mask.float()
    # [B, seq_len]
    response_log_probs = (token_log_probs * response_mask).sum(dim=-1)
    # Calculates the total number of tokens in the response, [B]
    response_lengths = response_mask.sum(dim=-1).clamp(min=1)
    return response_log_probs / response_lengths

def collate_fn(batch, tokenizer, max_length, device):
    prompt_encodings = tokenizer(
        ['Instruct: ' + item['prompt'] + '\n' for item in batch],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    chosen_encodings = tokenizer(
        ['Output: ' + item['chosen'] for item in batch],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    rejected_encodings = tokenizer(
        ['Output: ' + item['rejected'] for item in batch],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # [B, max_length]
    prompt_preferred_ids = torch.cat([
        prompt_encodings.input_ids,
        chosen_encodings.input_ids
    ], dim=-1).to(device)
    
    # [B, max_length]
    prompt_dispreferred_ids = torch.cat([
        prompt_encodings.input_ids,
        rejected_encodings.input_ids
    ], dim=-1).to(device)

    # [B, max_length]    
    prompt_preferred_mask = torch.cat([
        prompt_encodings.attention_mask,
        chosen_encodings.attention_mask
    ], dim=-1).to(device)

    # [B, max_length]        
    prompt_dispreferred_mask = torch.cat([
        prompt_encodings.attention_mask,
        rejected_encodings.attention_mask
    ], dim=-1).to(device)

    # [B]
    prompt_lengths = prompt_encodings.attention_mask.sum(dim=-1).to(device)

    return {
        'prompt_preferred_ids': prompt_preferred_ids,
        'prompt_dispreferred_ids': prompt_dispreferred_ids,
        'prompt_preferred_mask': prompt_preferred_mask,
        'prompt_dispreferred_mask': prompt_dispreferred_mask,
        'prompt_lengths': prompt_lengths
    }
