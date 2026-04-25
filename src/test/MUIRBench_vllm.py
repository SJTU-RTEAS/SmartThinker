from datasets import load_dataset, load_from_disk
from vllm import LLM, SamplingParams
from math_verify import verify, parse, LatexExtractionConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
import os
import argparse
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics.pass_at_k import calc_mean_pass_at_k
from utils.metrics.mean_len import calc_mean_len
from utils.rewards.norm_answer import normalize_latex

def main(args):
    model_path = args.model_path
    dataset_path = args.dataset
    dataset = load_dataset(dataset_path)
    test_dataset = dataset["test"]

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

    prompts = []
    for example in test_dataset:
        question = example["question"]
        images = example["image_list"]
        options = example["options"]
        
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        options_text = f"Options:\n{options_text}\nPlease select the correct answer from the options above. \n"
        question_text = question.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        options_text = options_text.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        
        prompt_text = f'Question: {question_text}\n'
        prompt_text += options_text

        cot_prompt = "Put your final answer within \\boxed{}. If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely—provide your best guess even if unsure. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering."
        prompt_text += cot_prompt
        chat_prompt = [
            {
                "role": "user", "content":[
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        formatted = tokenizer.apply_chat_template(
            chat_prompt, tokenize=False, add_generation_prompt=True
        )
        prompts.append(
            {
                "prompt": formatted,
                "multi_modal_data": {"image": images}
            }
        )

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
    target_answers = [example["answer"] for example in test_dataset]

    assert len(all_generations) == len(test_dataset), "The number of generations does not match the number of test examples."

    def extract_after_think(response):
        end_tag = '</think>'
        return response.split(end_tag)[-1]

    def verify_solutions(solutions, target_answer):
        correctness = []

        if target_answer is not None:
            target_answer = normalize_latex(target_answer)

        for sol in solutions:
            sol_after_think = normalize_latex(extract_after_think(sol))
            #print(f"Solution after think: {sol_after_think[-100:]}")
            try:
                #parsed_sol = parse(sol_after_think)
                if target_answer is not None:
                    parsed_sol = parse(sol_after_think)
                    parsed_target = parse(f"${target_answer}$")
                    is_correct = verify(
                        parsed_sol,
                        parsed_target
                    )
                    #print(f"Parsed solution: {parsed_sol}, Parsed target: {parsed_target}, Correct: {is_correct}")
                else:
                    is_correct = False
            except Exception as e:
                print(f"Verification error: {e}")
                is_correct = False
            correctness.append(is_correct)
        return correctness
    
    correctness_list = []
    for i in range(len(all_generations)):
        correctness_list.append(verify_solutions(all_generations[i], target_answers[i]))

    # calculate pass@1, pass@10, mean_len
    result = calc_mean_pass_at_k(correctness_list, k_list=[1, 2, 4])
    result["mean_len"] = calc_mean_len(all_generations, tokenizer)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model for evaluation.")
    parser.add_argument("--dataset", type=str, default="yentinglin/aime_2025", help="Dataset name or path to the dataset.")
    parser.add_argument("--sample_num", type=int, default=1, help="Number of samples to generate per question.")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Tensor parallel size for VLLM inference.")
    args = parser.parse_args()
    main(args)