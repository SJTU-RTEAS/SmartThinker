from collections import defaultdict
from functools import partial
from typing import Any, Callable

import numpy as np
import torch

from verl import DataProto
from verl.utils.import_utils import deprecated

def compute_extra_metrics(batch: DataProto) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - extra/acc_score/mean, max, min
            - extra/opt_len/mean, max, min
            - extra/step_sim_mean
            - extra/step_sim_std
    """
    acc_score = batch.non_tensor_batch.get("acc_score", None)
    opt_len = batch.non_tensor_batch.get("opt_len", None)
    step_sim_mean = batch.non_tensor_batch.get("step_sim_mean", None)
    step_sim_std = batch.non_tensor_batch.get("step_sim_std", None)

    if acc_score is not None:
        acc_score_mean = np.mean(acc_score)
        acc_score_max = np.max(acc_score)
        acc_score_min = np.min(acc_score)

    if opt_len is not None:
        opt_len = np.array([l for l in opt_len if l is not None])
        opt_len_mean = np.mean(opt_len)
        opt_len_max = np.max(opt_len)
        opt_len_min = np.min(opt_len)

    if step_sim_mean is not None:
        step_sim_mean = np.mean(step_sim_mean)
    if step_sim_std is not None:
        step_sim_std = np.mean(step_sim_std)

    # Aborted samples and non-aborted response length statistics
    # response_length_non_aborted/*: statistics computed on non-aborted samples only
    metrics = {
        "extra/acc_score/mean": acc_score_mean if acc_score is not None else None,
        "extra/acc_score/max": acc_score_max if acc_score is not None else None,
        "extra/acc_score/min": acc_score_min if acc_score is not None else None,
        "extra/opt_len/mean": opt_len_mean if opt_len is not None else None,
        "extra/opt_len/max": opt_len_max if opt_len is not None else None,
        "extra/opt_len/min": opt_len_min if opt_len is not None else None,
        "extra/step_sim_mean": step_sim_mean if step_sim_mean is not None else None,
        "extra/step_sim_std": step_sim_std if step_sim_std is not None else None,
    }

    return metrics