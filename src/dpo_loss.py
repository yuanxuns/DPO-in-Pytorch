import argparse
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def seed_everything(seed=2003):
    """
    Seeds all random number generators used in this codebase.

    Args:
        seed: The seed to use for seeding the random number generators.
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_dpo_loss(
    model_prefered_logprob: torch.Tensor,  # [B]
    model_disprefered_logprob: torch.Tensor,  # [B]
    ref_prefered_logprob: torch.Tensor,  # [B]
    ref_disprefered_logprob: torch.Tensor,  # [B]
    beta: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the DPO loss and other metrics.

    Args:
        model_prefered_logprob (tensor): [B] The model's log prob of the preferred response.
        model_disprefered_logprob (tensor): [B] The model's log prob of the dispreferred response.
        ref_prefered_logprob (tensor): [B] The reference model's log prob of the preferred response.
        ref_disprefered_logprob (tensor): [B] The reference model's log prob of the dispreferred response.
        beta (float): The hyperparameter for the reward margin. Defaults to 0.5.

    Returns:
        tuple:
            - loss (pt.float): The total loss of the model over the given batch.
            - prefered_relative_logprob (pt.float): The mean relative log prob of the preferred response.
            - disprefered_relative_logprob (pt.float): The mean relative log prob of the dispreferred response.
            - reward_accuracies (pt.float): The mean accuracy of the reward.
            - reward_margins (pt.float): The mean margin of the reward.
    """

    # [B]
    prefered_relative_logprob = model_prefered_logprob - ref_prefered_logprob
    # [B]
    disprefered_relative_logprob = model_disprefered_logprob - ref_disprefered_logprob

    # If the preferred relative logprob is greater than the dispreferred
    # relative logprob, it is considered a correct prediction. pt.float.
    reward_accuracies = (
        (prefered_relative_logprob > disprefered_relative_logprob).float().mean(dim=-1)
    )

    # The reward margin is defined as the difference between the preferred and
    # dispreferred relative log prob. pt.float
    reward_margins = (prefered_relative_logprob - disprefered_relative_logprob).mean(
        dim=-1
    )

    # The average loss over the batch. pt.float
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


def get_log_prob(
    logits: torch.Tensor,  # [B, 2*max_length, vocab_size]
    labels: torch.Tensor,  # [B, 2*max_length]
    prompt_lengths: torch.Tensor,  # [B]
) -> torch.Tensor:
    """
    Computes the log probability of the responsive tokens for the given batch.

    Args:
        logits (tensor):         [B, 2*max_length, vocab_size]
        labels (tensor):         [B, 2*max_length]
        prompt_lengths (tensor): [B]
            Note: seq_len = 2*max_length due to the concatenation of prompt and response

    Returns:
        tensor: [B] The summation of log probs of all responsive tokens for each batch
    """

    # [B, 2*max_length, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)
    # [B, 2*max_length]
    token_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)

    # Creates a mask for the non-prompt part of the sequence, [B, seq_len]
    seq_len = labels.shape[1]
    response_mask = torch.arange(seq_len, device=labels.device).unsqueeze(
        0
    ) >= prompt_lengths.unsqueeze(1)
    response_mask = response_mask.float()

    # Sums the log probs of the responsive tokens, [B]
    response_log_probs = (token_log_probs * response_mask).sum(dim=-1)
    # Calculates the total number of tokens in the response, [B]
    response_lengths = response_mask.sum(dim=-1).clamp(min=1)
    return response_log_probs / response_lengths


def collate_fn(batch, tokenizer, max_length, device):
    """
    Collate function for the dataset.

    Args:
        batch: A list of examples from the dataset.
        tokenizer: The tokenizer to use for encoding the text.
        max_length: The maximum length of the sequence to return.
        device: The device to use for the returned tensors.

    Returns:
        A dict of the following keys:
            - prompt_preferred_ids: The input ids for the preferred response, [B, max_length]
            - prompt_dispreferred_ids: The input ids for the dispreferred response, [B, max_length]
            - prompt_preferred_mask: The attention mask for the preferred response, [B, max_length]
            - prompt_dispreferred_mask: The attention mask for the dispreferred response, [B, max_length]
            - prompt_lengths: The lengths of the prompts, [B]
    """

    prompt_encodings = tokenizer(
        ["Instruct: " + item["prompt"] + "\n" for item in batch],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    chosen_encodings = tokenizer(
        ["Output: " + item["chosen"] for item in batch],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    rejected_encodings = tokenizer(
        ["Output: " + item["rejected"] for item in batch],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Concatenates the prompt and response, [B, max_length]
    prompt_preferred_ids = torch.cat(
        [prompt_encodings.input_ids, chosen_encodings.input_ids], dim=-1
    ).to(device)

    # Concatenates the prompt and response, [B, max_length]
    prompt_dispreferred_ids = torch.cat(
        [prompt_encodings.input_ids, rejected_encodings.input_ids], dim=-1
    ).to(device)

    # The EOS mask, [B, max_length]
    prompt_preferred_mask = torch.cat(
        [prompt_encodings.attention_mask, chosen_encodings.attention_mask], dim=-1
    ).to(device)

    # The EOS mask, [B, max_length]
    prompt_dispreferred_mask = torch.cat(
        [prompt_encodings.attention_mask, rejected_encodings.attention_mask], dim=-1
    ).to(device)

    # The unmasked prompt length [B]
    prompt_lengths = prompt_encodings.attention_mask.sum(dim=-1).to(device)

    return {
        "prompt_preferred_ids": prompt_preferred_ids,
        "prompt_dispreferred_ids": prompt_dispreferred_ids,
        "prompt_preferred_mask": prompt_preferred_mask,
        "prompt_dispreferred_mask": prompt_dispreferred_mask,
        "prompt_lengths": prompt_lengths,
    }
