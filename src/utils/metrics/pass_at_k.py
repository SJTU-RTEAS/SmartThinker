from scipy.special import comb
import numpy as np

def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def calc_mean_pass_at_k(correctness_list, k_list=[1, 10, 100]):
    if not correctness_list:
        return {}
    
    num_samples = len(correctness_list[0]) if correctness_list else 0

    for i, results in enumerate(correctness_list):
        if len(results) != num_samples:
            raise ValueError(f"第{i}个问题的采样数量({len(results)})与其他问题不一致({num_samples})")
    
    results = {}
    
    for k in k_list:
        if k > num_samples:
            print(f"警告: k={k} 大于采样数量{num_samples}，跳过计算")
            continue
            
        pass_rates = []
        
        # 对每个问题计算pass@k
        for problem_results in correctness_list:
            # 统计正确的采样次数
            correct_count = sum(problem_results)
            total_count = len(problem_results)
            
            # 计算该问题的pass@k
            pass_rate = pass_at_k(total_count, correct_count, k)
            pass_rates.append(pass_rate)
        
        # 计算平均值和标准差
        mean_pass_rate = np.mean(pass_rates)
        std_pass_rate = np.std(pass_rates)
        
        results[f'pass@{k}'] = {
            'mean': mean_pass_rate,
            'std': std_pass_rate,
            #'values': pass_rates
        }
        
    return results