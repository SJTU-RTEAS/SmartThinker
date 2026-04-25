from datasets import load_dataset
from vllm import LLM, SamplingParams
from openai import OpenAI
from PIL import Image
import io
import base64
import sys
import os
import argparse
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics.pass_at_k import calc_mean_pass_at_k
from utils.metrics.mean_len import calc_mean_len

def main(args):
    model_path = args.model_path
    dataset_path = args.dataset

    dataset = load_dataset(dataset_path, trust_remote_code=True)
    if "test" in dataset:
        test_dataset = dataset["test"]
    else:
        test_dataset = dataset[list(dataset.keys())[0]]

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

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompts = []
    for example in test_dataset:
        prompt = example["question"]
        image = example["image"]

        chat_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        formatted = tokenizer.apply_chat_template(
            chat_prompt, tokenize=False, add_generation_prompt=True
        )
        prompts.append({
            "prompt": formatted,
            "multi_modal_data": {"image": image}
        })

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

    assert len(all_generations) == len(test_dataset)

    def extract_after_think(response):
        end_tag = '</think>'
        return response.split(end_tag)[-1]

    def verify_solutions(solutions, target_answer):
        correctness = []
        for sol in solutions:
            sol_after_think = extract_after_think(sol)
            #print(f"Solution after think: {sol_after_think[-100:]}")
            try:
                #parsed_sol = parse(sol_after_think)
                if target_answer is not None:
                    assert type(target_answer) == list
                    for ans in target_answer:
                        parsed_sol = sol_after_think.lower().strip().replace("\n", " ")
                        parsed_ans = ans.lower().strip().replace("\n", " ")
                        if parsed_ans in parsed_sol:
                            is_correct = True
                            break
                        else:
                            is_correct = False
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

    result = calc_mean_pass_at_k(correctness_list, k_list=[1, 2, 4])
    result["mean_len"] = calc_mean_len(all_generations, tokenizer)
    print(result)

    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model for evaluation.")
    parser.add_argument("--dataset", type=str, default="echo840/OCRBench", help="Dataset name or path to the dataset.")
    parser.add_argument("--sample_num", type=int, default=1, help="Number of samples to generate per question.")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Tensor parallel size for VLLM inference.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save results JSON file.")
    args = parser.parse_args()
    main(args)