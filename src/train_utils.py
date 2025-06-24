import numpy
import pandas
import torch
import torch.functional as F
import torch.nn as nn
import wandb
from tqdm import tqdm

from src.dpo_loss import compute_dpo_loss, get_log_prob


def train(model, ref_model, tokenizer, optimizer, train_dataloader, epochs=1, beta=0.1):
    """
    Trains the model using Direct Preference Optimization (DPO) loss.

    Args:
        model: The model to be trained.
        ref_model: The reference model used for computing relative log probs.
        tokenizer: The tokenizer used for processing input text.
        optimizer: The optimizer used for updating model parameters.
        train_dataloader: DataLoader providing batches of training data.
        epochs: Number of training epochs.
        beta: The hyperparameter for adjusting the reward margin.

    This function trains the model by iterating over the training data, computing
    the DPO loss, and updating the model parameters. It also logs the loss and
    other metrics to Weights & Biases (wandb) for tracking.
    """

    model.train()
    ref_model.eval()

    for epoch in tqdm(range(epochs), desc="Epochs"):
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()

            prompt_prefered_ids = batch["prompt_preferred_ids"]
            prompt_disprefered_ids = batch["prompt_dispreferred_ids"]
            prompt_prefered_mask = batch["prompt_preferred_mask"]
            prompt_disprefered_mask = batch["prompt_dispreferred_mask"]
            prompt_lengths = batch["prompt_lengths"]

            model_prefered_log_prob = get_log_prob(
                model(prompt_prefered_ids, attention_mask=prompt_prefered_mask).logits,
                prompt_prefered_ids,
                prompt_lengths,
            )
            model_disprefered_log_prob = get_log_prob(
                model(
                    prompt_disprefered_ids, attention_mask=prompt_disprefered_mask
                ).logits,
                prompt_disprefered_ids,
                prompt_lengths,
            )

            ref_prefered_log_prob = get_log_prob(
                ref_model(
                    prompt_prefered_ids, attention_mask=prompt_prefered_mask
                ).logits,
                prompt_prefered_ids,
                prompt_lengths,
            )
            ref_disprefered_log_prob = get_log_prob(
                ref_model(
                    prompt_disprefered_ids, attention_mask=prompt_disprefered_mask
                ).logits,
                prompt_disprefered_ids,
                prompt_lengths,
            )

            (
                loss,
                prefered_relative_logprob,
                disprefered_relative_logprob,
                reward_accuracies,
                reward_margins,
            ) = compute_dpo_loss(
                model_prefered_log_prob,
                model_disprefered_log_prob,
                ref_prefered_log_prob,
                ref_disprefered_log_prob,
                beta=beta,
            )

            loss.backward()
            optimizer.step()

            wandb.log(
                data={
                    "loss": loss.item(),
                    "prefered_relative_logprob": prefered_relative_logprob,
                    "disprefered_relative_logprob": disprefered_relative_logprob,
                    "reward_accuracy": reward_accuracies,
                    "reward_margin": reward_margins,
                }
            )
