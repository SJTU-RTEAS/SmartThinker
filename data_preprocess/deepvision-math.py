"""
Preprocess the DeepScaleR dataset  to parquet format
"""

import argparse
import os
from pathlib import Path

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default=os.path.join(Path(__file__).parent.parent, "datasets"), help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path
    
    data_source = "skylenage/DeepVision-103K"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset("parquet", data_files=os.path.join(local_dataset_path, "math-77k.parquet"))
    else:
        dataset = datasets.load_dataset(data_source, "math")
    
    train_dataset = dataset["train"]

    instruction_following = '\nPlease reason step by step, and put your final answer within \\boxed{}.'
    #instruction_following = ' \n\nGive the final answer within \\boxed{}.'
    
        # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = "<image>" + question_raw + instruction_following
            images = example.pop("images")
            reward_model = example.pop("reward_model")
            answer = reward_model["ground_truth"]
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "images": images,
                "ability": "math",
                "reward_model": reward_model,
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_save_dir        
    
    train_dataset.to_parquet(os.path.join(local_save_dir, "deepvision-math.parquet"))
    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)