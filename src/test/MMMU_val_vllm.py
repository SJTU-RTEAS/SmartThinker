from datasets import load_dataset, load_from_disk, get_dataset_config_names
from vllm import LLM, SamplingParams
from math_verify import verify, parse, LatexExtractionConfig, StringExtractionConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
import os
import argparse
import json
import re
from PIL import Image
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics.pass_at_k import calc_mean_pass_at_k
from utils.metrics.mean_len import calc_mean_len
from utils.rewards.norm_answer import normalize_latex

def evaluate_subset(llm, tokenizer, sampling_params, dataset, subset_name):
    """Evaluate a single subset."""
    print(f"\n{'='*60}")
    print(f"Evaluating subset: {subset_name}")
    print(f"{'='*60}")
    print(f"Number of examples: {len(dataset)}")
    
    prompts = []
    for idx, example in enumerate(dataset):
        options = example.get("options", None)
        # options are string, convert to list. for example, "options": "['$6', '$7', '$8', '$9']" -> ["$6", "$7", "$8", "$9"]
        if options is not None and isinstance(options, str):
            try:
                options = eval(options)
            except Exception as e:
                print(f"Error parsing options for example {idx}: {e}")
                options = None
        options_text = ""
        if options is not None:
            options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            options_text = f"Options:\n{options_text}\nPlease select the correct answer from the options above. \n"
        question_text = example["question"]
        cot_prompt = "Put your final answer within \\boxed{}. If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely—provide your best guess even if unsure. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering."

        prompt_text = "Question: " + question_text + "\n" + options_text + cot_prompt
        
        # Collect all available images (image_1 to image_7)
        
        #num_placeholders = len(re.findal/l(r'<image\s*\d*>', example["question"]))
        #print(f"Example {idx}: {num_placeholders} placeholders, {len(images)} images")
        
        images = []
        image_placeholders = re.findall(r'<image\s*(\d+)>', prompt_text)
        for img_num in image_placeholders:
            img_key = f"image_{img_num}"
            if img_key in example and example[img_key] is not None:
                img = example[img_key]
                images.append(img)
                prompt_text = prompt_text.replace(f'<image {img_num}>', '<|vision_start|><|image_pad|><|vision_end|>')
            else:
                print(f"Warning: Placeholder <image {img_num}> found but no corresponding image_{img_num} in example {idx}.")
        
        # Build chat prompt - <image n> tags are already in question
        chat_prompt = [
            {
                "role": "user",
                "content":[
                    {"type": "text", "text": prompt_text},
                ]
            }
        ]

        formatted = tokenizer.apply_chat_template(
            chat_prompt, tokenize=False, add_generation_prompt=True
        )
        
        prompt_dict = {
            "prompt": formatted,
            "multi_modal_data": {"image": images}
        }
        prompts.append(prompt_dict)

    print("Generating responses...")
    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=True,
    )

    def parse_generations(request_output):
        generations = []
        for output in request_output.outputs:
            text = output.text
            generations.append(text)
        return generations

    all_generations = [parse_generations(output) for output in outputs]
    target_answers = [example["answer"] for example in dataset]

    assert len(all_generations) == len(dataset), "The number of generations does not match the number of test examples."

    def extract_after_think(response):
        end_tag = '</think>'
        return response.split(end_tag)[-1]

    def verify_solutions(solutions, target_answer):
        correctness = []

        if target_answer is not None:
            target_answer = normalize_latex(target_answer)

        for sol in solutions:
            sol_after_think = normalize_latex(extract_after_think(sol))
            try:
                if target_answer is not None:
                    parsed_sol = parse(sol_after_think)
                    parsed_target = parse(f"${target_answer}$")
                    is_correct = verify(
                        parsed_sol,
                        parsed_target
                    )
                else:
                    is_correct = False
            except Exception as e:
                print(f"Verification error: {e}")
                is_correct = False
            correctness.append(is_correct)
            print(f"Solution: {sol_after_think} | Target: {target_answer} | Correct: {is_correct}")
        return correctness
    
    correctness_list = []
    for i in range(len(all_generations)):
        correctness_list.append(verify_solutions(all_generations[i], target_answers[i]))

    # calculate pass@1, pass@2, pass@4, mean_len
    result = calc_mean_pass_at_k(correctness_list, k_list=[1, 2, 4])
    result["mean_len"] = calc_mean_len(all_generations, tokenizer)
    result["num_examples"] = len(dataset)
    result["subset_name"] = subset_name
    
    print(f"\nResults for {subset_name}:")
    for k, v in result.items():
        if k not in ["subset_name"]:
            print(f"  {k}: {v}")
    
    return result, correctness_list


def main(args):
    model_path = args.model_path
    dataset_path = args.dataset
    
    # Get all available configs for MMMU
    print(f"Loading dataset: {dataset_path}")
    available_configs = get_dataset_config_names(dataset_path)
    print(f"Available configs: {available_configs}")
    
    # Load all subsets with validation split
    subsets_to_eval = []
    for config_name in available_configs:
        try:
            ds = load_dataset(dataset_path, config_name, trust_remote_code=True)
            if "validation" in ds:
                subsets_to_eval.append((config_name, ds["validation"]))
                print(f"Loaded: {config_name}/validation ({len(ds['validation'])} examples)")
        except Exception as e:
            print(f"Failed to load {config_name}: {e}")
    
    if not subsets_to_eval:
        raise ValueError("No validation splits found in dataset!")

    # Setup LLM
    sampling_params = SamplingParams(
        n=args.sample_num,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        presence_penalty=1.5,
        max_tokens=32768,
        stop=[]
    )
    
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Evaluate all subsets
    all_results = []
    all_correctness_lists = []
    total_examples = 0

    for subset_name, subset_dataset in subsets_to_eval:
        result, correctness_list = evaluate_subset(
            llm, tokenizer, sampling_params, subset_dataset, subset_name
        )
        all_results.append(result)
        all_correctness_lists.extend(correctness_list)
        total_examples += len(subset_dataset)

    # Calculate aggregated results across all subsets
    print(f"\n{'='*60}")
    print("FINAL AGGREGATED RESULTS")
    print(f"{'='*60}")
    print(f"Total subsets evaluated: {len(all_results)}")
    print(f"Total examples: {total_examples}")

    # Aggregate pass@k metrics
    aggregated_result = calc_mean_pass_at_k(all_correctness_lists, k_list=[1, 2, 4])
    
    # Calculate weighted mean_len
    total_len = sum(r["mean_len"] * r["num_examples"] for r in all_results)
    aggregated_result["mean_len"] = total_len / total_examples if total_examples > 0 else 0
    
    # Add per-subset breakdown
    aggregated_result["num_subsets"] = len(all_results)
    aggregated_result["total_examples"] = total_examples
    aggregated_result["per_subset"] = {
        r["subset_name"]: {
            "pass@1": r.get("pass@1", r.get("mean_pass@1")),
            "pass@2": r.get("pass@2", r.get("mean_pass@2")),
            "pass@4": r.get("pass@4", r.get("mean_pass@4")),
            "mean_len": r["mean_len"],
            "num_examples": r["num_examples"]
        } for r in all_results
    }

    print("\nAggregated metrics:")
    for k, v in aggregated_result.items():
        if k not in ["per_subset", "num_subsets", "total_examples"]:
            print(f"  {k}: {v}")

    print("\nPer-subset breakdown:")
    for subset_name, metrics in aggregated_result["per_subset"].items():
        print(f"  {subset_name}:")
        for mk, mv in metrics.items():
            print(f"    {mk}: {mv}")

    # Save results to file
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(aggregated_result, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")

    return aggregated_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model for evaluation.")
    parser.add_argument("--dataset", type=str, default="MMMU/MMMU", help="Dataset name or path to the dataset.")
    parser.add_argument("--sample_num", type=int, default=1, help="Number of samples to generate per question.")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Tensor parallel size for VLLM inference.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save results JSON file.")
    args = parser.parse_args()
    main(args)