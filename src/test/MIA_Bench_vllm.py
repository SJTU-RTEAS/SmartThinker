from datasets import load_dataset
from vllm import LLM, SamplingParams
from openai import OpenAI
from PIL import Image, ImageOps
import io
import base64
import sys
import os
import argparse
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

# load .env
from dotenv import load_dotenv
load_dotenv()

print("OpenAI API Key:", os.environ.get("OPENAI_API_KEY"))
print("OpenAI Base URL:", os.environ.get("OPENAI_BASE_URL"))

MAX_JUDGE_IMAGE_BYTES = 4 * 1024 * 1024
JUDGE_IMAGE_MAX_SIDES = (2048, 1600, 1280, 1024, 768)
JUDGE_IMAGE_JPEG_QUALITIES = (90, 80, 70, 60, 50, 40)

def generate_prompt(d, response):
    instruction = d['instruction']
    weight = d['component_weight'] * 1
    d['num_of_component'] = len(d['components'])
    for i in range(len(weight)):
        weight[i] = str(weight[i])
    
    if d['num_of_component'] == 1:
        components = '''The first component is:' ''' + d['components'][0] + "'"  
        score = '''The first component is worth ''' + weight[0] + ' scores.'
    elif d['num_of_component'] == 2:
        components = '''The first component is:' ''' + d['components'][0] + '''', and the second component is:' ''' + d['components'][1] + "'" 
        score = '''The first and second component is each worth ''' + weight[0] + ' and ' + weight[1]+ ' scores.'
    elif d['num_of_component'] == 3:
        components = '''The first component is:' ''' + d['components'][0] + '''', and the second component is:' ''' + d['components'][1] + '''', and the third component is:' ''' + d['components'][2] + "'" 
        score = '''The first second, and third component is each worth ''' + weight[0] + ', ' + weight[1]+ ' and ' + weight[2] + ' scores.'
    elif d['num_of_component'] == 4:
        components = '''The first component is:' ''' + d['components'][0] + '''', and the second component is:' ''' + d['components'][1] + '''', and the third component is:' ''' + d['components'][2] +  '''', and the fourth component is:' ''' + d['components'][3] + "'" 
        score = '''The first second, third, and fourth component is each worth ''' + weight[0] + ', ' + weight[1]+ ', ' + weight[2] + ' and ' + weight[3] + ' scores.'
    elif d['num_of_component'] == 5:
        components = '''The first component is:' ''' + d['components'][0] + '''', and the second component is:' ''' + d['components'][1] + '''', and the third component is:' ''' + d['components'][2] +  '''', and the fourth component is:' ''' + d['components'][3] +  '''', and the fifth component is:' ''' + d['components'][4] + "'" 
        score = '''The first second, third, fourth and fifth component is each worth ''' + weight[0] + ', ' + weight[1]+ ', ' + weight[2] + ', ' + weight[3] + ' and ' + weight[4] + ' scores.'      
    
    return '''Here is an instruction for a multimodal LLM: ' ''' + instruction + ''' You need to grade if the response from the model follows each component of the instruction. ''' + components + ''' The response is:' '''  + response +  ''' You need to score the response and be strict. The total score ranges from 0 to 10, depending on if the response follows the instruction. ''' + score + ' List scores of each component, and the total score in one sentence in this format: score of component 1: x/2, score of component 2: y/8, total score: z/10. Then explain your reasons.'


def process_rawscore(component_type, raw_score):
    first_sentence = raw_score.split('.')[0].split(',')
    score_dict = {}
    for i in range(len(first_sentence) - 1):
        score_ = first_sentence[i].split(':')[1][1:].split('/')
        score = int(score_[0])/int(score_[1])
        score_dict[component_type[i]] = score
    total_score_ = first_sentence[i+1].split(':')[1][1:].split('/')
    total_score = int(total_score_[0])/int(total_score_[1])
    score_dict['total_score'] = total_score
    return score_dict  


def get_score_dict(df, column_name):
    cat_score_dict = {}
    for i in range(len(df)):
        try:
            score_dict = process_rawscore(df['component_type'][i], df[column_name][i])
            for key, val in score_dict.items():
                if key not in cat_score_dict.keys():
                    cat_score_dict[key] = [val]
                else:
                    cat_score_dict[key].append(val)
        except:
            pass
    cat_score_dict_average = {}
    for key, val in cat_score_dict.items():
        cat_score_dict_average[key] = sum(val)/len(val)
    return cat_score_dict_average


def aggregate_judge_completion_tokens(df, column_name="judge_completion_tokens"):
    token_list = []
    for i in range(len(df)):
        token_count = df[column_name][i]
        if token_count is None:
            continue
        try:
            token_list.append(int(token_count))
        except (TypeError, ValueError):
            continue

    if not token_list:
        return {
            "judge_completion_tokens_total": 0,
            "judge_completion_tokens_mean": 0,
            "judge_completion_tokens_count": 0,
        }

    return {
        "judge_completion_tokens_total": sum(token_list),
        "judge_completion_tokens_mean": sum(token_list) / len(token_list),
        "judge_completion_tokens_count": len(token_list),
    }

def _decode_data_url_image(image_url):
    if not isinstance(image_url, str) or not image_url.startswith("data:image/"):
        return None

    _, encoded = image_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    with Image.open(io.BytesIO(image_bytes)) as image:
        return image.copy()


def _normalize_image_for_judge(image):
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba_image = image.convert("RGBA")
        background = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba_image).convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _image_bytes_to_data_url(image_bytes, mime_type):
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def prepare_judge_image_url(image, max_bytes=MAX_JUDGE_IMAGE_BYTES):
    if isinstance(image, Image.Image):
        pass
    elif isinstance(image, str) and image.startswith("data:image/"):
        image = _decode_data_url_image(image)
    else:
        return image

    image = _normalize_image_for_judge(image)
    resampling = getattr(Image, "Resampling", Image).LANCZOS

    for max_side in JUDGE_IMAGE_MAX_SIDES:
        resized = image.copy()
        if max(resized.size) > max_side:
            resized.thumbnail((max_side, max_side), resampling)

        for quality in JUDGE_IMAGE_JPEG_QUALITIES:
            buffered = io.BytesIO()
            resized.save(
                buffered,
                format="JPEG",
                quality=quality,
                optimize=True,
            )
            image_bytes = buffered.getvalue()
            if len(image_bytes) <= max_bytes:
                return _image_bytes_to_data_url(image_bytes, "image/jpeg")

    raise ValueError(
        f"Image could not be compressed below {max_bytes} bytes for judge evaluation."
    )


def main(args):
    model_path = args.model_path
    dataset_path = args.dataset

    dataset = load_dataset(dataset_path)
    test_dataset = dataset["test"]

    sampling_params = SamplingParams(
        n=1,
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
        instruction = example["instruction"]
        image = example["image"]

        chat_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction}
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

    all_generations = [parse_generations(output)[0] for output in outputs]

    assert len(all_generations) == len(test_dataset)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    df_data = []
    for i in tqdm(range(len(test_dataset))):
        example = test_dataset[i]
        d = {
            'instruction': example['instruction'],
            'components': example['components'],
            'component_weight': example['component_weight'],
            'component_type': example['component_type'],
        }
        response = all_generations[i]
        
        question = generate_prompt(d, response)
        
        image = example["image"]
        image_url = prepare_judge_image_url(image)
        
        generated = False
        attempt = 5
        judge_completion_tokens = None
        while attempt > 0 and generated == False:
            try:
                response_eval = client.chat.completions.create(
                    model="openai/gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question},
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        }
                    ],
                    max_tokens=2000
                )
                print(response_eval.choices[0].message.content.strip())
                score_raw = response_eval.choices[0].message.content.strip()
                if getattr(response_eval, "usage", None) is not None:
                    judge_completion_tokens = response_eval.usage.completion_tokens
                generated = True
            except Exception as e:
                print(f"Error: {e}")
                attempt -= 1
                score_raw = "error"
        
        df_data.append({
            'instruction': example['instruction'],
            'components': example['components'],
            'component_weight': example['component_weight'],
            'component_type': example['component_type'],
            'text': all_generations[i],
            'url': image_url,
            'score_raw': score_raw,
            'judge_completion_tokens': judge_completion_tokens,
        })

    import pandas as pd
    df = pd.DataFrame(df_data)
    
    result = get_score_dict(df, 'score_raw')
    result["benchmark_score"] = result.get("total_score", 0)
    result.update(aggregate_judge_completion_tokens(df))
    print("Result:", result)

    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model for evaluation.")
    parser.add_argument("--dataset", type=str, default="lmms-lab/MIA-Bench", help="Dataset name or path to the dataset.")
    parser.add_argument("--sample_num", type=int, default=1, help="Number of samples to generate per question.")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Tensor parallel size for VLLM inference.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save results JSON file.")
    args = parser.parse_args()
    main(args)
