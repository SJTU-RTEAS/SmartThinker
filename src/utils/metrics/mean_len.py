import numpy as np

def calc_mean_len(all_generations, tokenizer):
    len_list = []
    for generations in all_generations:
        for gen in generations:
            len_list.append(len(tokenizer.encode(gen, add_special_tokens=False)))
    return np.mean(len_list)