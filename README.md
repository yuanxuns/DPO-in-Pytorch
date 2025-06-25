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


## Why DPO?

- **Simplicity**: Direct classification-based loss; no RL or reward model :contentReference 
- **Performance**: Matches or beats PPO‑based RLHF in key benchmarks 
- **Efficiency**: Lightweight training; fewer computations & no sampling loops :contentReference  


## DPO Loss Module

In `src/dpo_loss.py`, the heart of DPO is implemented:
<img width="886" alt="image" src="https://github.com/user-attachments/assets/43913747-6a99-43c3-b1de-1b82813f088b" />


```python
    # The average loss over the batch. (pt.float)
    loss = -F.logsigmoid(
        beta * (prefered_relative_logprob - disprefered_relative_logprob)
    ).mean(dim=-1)
```


## Training Script
```
bash run_training.sh
```


## Reference:
https://arxiv.org/abs/2305.18290

https://github.com/mrunalmania/Direct-Preference-Optimization/tree/main

https://github.com/0xallam/Direct-Preference-Optimization/tree/main
