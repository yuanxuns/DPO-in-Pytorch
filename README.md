# Direct-Preference-Optimization-DPO-in-Pytorch

This repository provides a clean, reproducible implementation of **Direct Preference Optimization (DPO)**—a simple, stable, and effective framework for aligning language models with human preferences, without the complexities of RLHF.

---

## What is DPO?

Direct Preference Optimization is a method introduced by Rafailov *et al.* (NeurIPS 2023) that streamlines preference-based fine-tuning of language models. Unlike traditional RLHF, DPO:
<img width="1027" alt="image" src="https://github.com/user-attachments/assets/b51129d3-f28d-4ca2-bd26-1e8037cf0ca8" />

- Removes the need to train a separate reward model.
- Avoids reinforcement learning loops and hyperparameter tuning.
- Uses a simple binary cross‑entropy loss on *preferred vs. dispreferred* model outputs.
- Matches or outperforms RLHF in tasks like sentiment control, summarization, and dialogue.
<br />

## Why DPO?

- **Simplicity**: Direct classification-based loss; no RL or reward model :contentReference 
- **Performance**: Matches or beats PPO‑based RLHF in key benchmarks 
- **Efficiency**: Lightweight training; fewer computations & no sampling loops
<br />


## DPO Loss Module

In `src/dpo_loss.py`, the heart of DPO is implemented:
<img width="886" alt="image" src="https://github.com/user-attachments/assets/43913747-6a99-43c3-b1de-1b82813f088b" />


```python
    # The average loss over the batch. (pt.float)
    loss = -F.logsigmoid(
        beta * (prefered_relative_logprob - disprefered_relative_logprob)
    ).mean(dim=-1)
```
<br />


## Training Script
```
bash run_training.sh
```
<br />

## Empirical Results
Due to the limitation of a single 3060 GPU, I use `SmolLM-135M-Instruct` with `batch size 2` on the `jondurbin/truthy-dpo-v0.1 dataset`.

```
python train.py \
    --epochs 5 \
    --batch_size 2 \
    --max_length 256 \
    --lr 1e-6 \
    --beta 0.1 \
    --seed 2003 \
    --model_name "HuggingFaceTB/SmolLM-135M-Instruct" \
    --dataset_name "jondurbin/truthy-dpo-v0.1" \
    --wandb_project "dpo"
```

`The training is pretty stable with naive settings above. The loss, reward accuracy and reward margin converged.`
![Screenshot from 2025-06-24 17-26-21](https://github.com/user-attachments/assets/b0853299-db31-40d6-99af-8f75fe55febf)

<br />

## References:
https://arxiv.org/abs/2305.18290

https://github.com/mrunalmania/Direct-Preference-Optimization/tree/main

https://github.com/0xallam/Direct-Preference-Optimization/tree/main
