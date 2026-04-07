import torch  
import numpy as np  
from typing import Optional  
from verl.trainer.ppo.core_algos import register_adv_est  
from verl.trainer.config import AlgoConfig
from collections import defaultdict


@register_adv_est("grpo_smartthinker")  # or simply: @register_adv_est("grpo")
def compute_grpo_stepprune_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    response_length: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    id2len = defaultdict(list)
    id2len_correct = defaultdict(list)
    id2mean_len = {}
    id2mean_len_correct = {}
    id2std_len = {}
    id2std_len_correct = {}
    id2max_len = {}
    id2min_len = {}
    id2opt_len = {}
    id2score_len = defaultdict(list)
    id2score_len_correct = defaultdict(list)
    id2coe = {}
    id2id = defaultdict(list)

    len_scores = torch.zeros_like(scores)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])
            if scores[i] > 0:
                id2len_correct[index[i]].append(response_length[i]) 
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for idx in id2len_correct:
            len_tensor = torch.tensor(id2len_correct[idx], dtype=torch.float32)
            id2mean_len_correct[idx] = torch.mean(len_tensor)
            id2std_len_correct[idx] = torch.std(len_tensor)
        for idx in id2len:
            len_tensor = torch.tensor(id2len[idx], dtype=torch.float32)
            id2mean_len[idx] = torch.mean(len_tensor)
            id2std_len[idx] = torch.std(len_tensor)
            id2max_len[idx] = torch.max(len_tensor)
            id2min_len[idx] = torch.min(len_tensor)

            if idx not in id2len_correct:
                id2opt_len[idx] = id2max_len[idx]
            else:
                if abs(id2std_len_correct[idx] - id2std_len[idx]) < epsilon:
                    if id2mean_len_correct[idx] > id2mean_len[idx]:
                        id2opt_len[idx] = id2max_len[idx]
                    else:
                        id2opt_len[idx] = id2min_len[idx]
                elif id2std_len_correct[idx] > id2std_len[idx]:
                    id2opt_len[idx] = id2min_len[idx]
                else:
                    id2opt_len[idx] = (id2std_len_correct[idx]**2 * id2mean_len[idx] - id2std_len[idx]**2 * id2mean_len_correct[idx]) / (id2std_len_correct[idx]**2 - id2std_len[idx]**2)
                    id2opt_len[idx] = max(id2min_len[idx], min(id2max_len[idx], id2opt_len[idx]))
        
        for i in range(bsz):
            if scores[i] <= 0:
                len_scores[i] = 0
                id2score_len[index[i]].append(0)
            elif response_length[i] < id2opt_len[index[i]]:
                len_scores[i] = 0
                id2score_len[index[i]].append(0)
                id2score_len_correct[index[i]].append(0)
            elif response_length[i] >= id2opt_len[index[i]]:
                len_scores[i] = response_length[i] - id2opt_len[index[i]]
                id2score_len[index[i]].append(response_length[i] - id2opt_len[index[i]])
                id2score_len_correct[index[i]].append(response_length[i] - id2opt_len[index[i]])

        for idx in id2score:
            acc_ratio = sum(id2score[idx]) / len(id2score[idx])
            err_ratio = 1 - acc_ratio
            if acc_ratio == 0 or err_ratio == 0:
                id2coe[idx] = 0.0
            else:
                id2coe[idx] = err_ratio / (max(id2score_len_correct[idx]) - sum(id2score_len[idx]) / len(id2score_len[idx]) + epsilon)
        
        for i in range(bsz):
            scores[i] = scores[i] - id2coe[index[i]] * len_scores[i]
                
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

        for i in range(bsz):
            id2id[index[i]].append(i)

        for idx in id2opt_len:
            print("="*20)
            print(f"Prompt index: {idx}, Optimal length: {id2opt_len[idx]}")
            for i in id2id[idx]:
                print(f"  Sample {i}: length: {response_length[i]}, Score: {token_level_rewards[i].sum().item()}, Adjusted score: {scores[i].item()}")

        scores = scores.unsqueeze(-1)
        scores = scores * response_mask

        # print optimal length, response length, score, and adjusted score for each sample

    return scores, scores
