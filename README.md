<center>

# SmartThinker: Progressive Chain-of-Thought Length Calibration for Efficient Large Language Model Reasoning

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-green)](https://huggingface.co/collections/etherwindy/smartthinker)

</center>

## Abastrct

Large reasoning models (LRMs) like OpenAI o1 and DeepSeek-R1 achieve high accuracy on complex tasks by adopting long chain-of-thought (CoT) reasoning paths.  However, the inherent verbosity of these processes frequently results in redundancy and overthinking. To address this issue, existing works leverage Group Relative Policy Optimization (GRPO) to reduce LRM output length, but their static length reward design cannot dynamically adapt according to the relative problem difficulty and response length distribution, causing over-compression and compromised accuracy. Therefore, we propose *SmartThinker*, a novel GRPO-based efficient reasoning method with progressive CoT length calibration. *SmartThinker* makes a two-fold contribution: First, it dynamically estimates the optimal length with peak accuracy during training and guides overlong responses toward it to reduce response length while sustaining accuracy. Second, it dynamically modulates the length reward coefficient to avoid the unwarranted penalization of correct reasoning paths. Extensive experiment results show that *SmartThinker* achieves up to 52.5\% average length compression with improved accuracy, and achieves up to 16.6\% accuracy improvement on challenging benchmarks like AIME25.

---

## Dependencies

- python: 3.12
- verl: 0.7.0.dev0
- CUDA: 12.8
- pytorch: 2.8.0
- vllm: 0.11.0

## Setup

Clone the repository:

``` bash
git clone https://github.com/SJTU-RTEAS/SmartThinker.git
cd SmartThinker
```

Install dependencies:

``` bash
conda create -n SmartThinker python==3.12
conda activate SmartThinker
pip install -r requirement.txt
```

## Training

First you need to configure your wandb api key:

``` bash
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
```

You can also configure it in the `.env` file at the root directory of the repository:

``` bash
WANDB_API_KEY="YOUR_WANDB_API_KEY"
```

The training scripts are located in the `script` folder. Take 1.5B model as an example, run the following command to start training:

``` bash
bash scripts/SmartThinker_Distill_1.5B.sh --model "YOUR_MODEL_PATH"
```

After training is complete, run the following command to convert the specified checkpoint to the Hugging Face model format：

``` bash
bash scripts/fsdp_merge_Distill_1.5B.sh
```

## Test

The test scripts are located in the `src/test` folder. Take AIME25 as an example, run the following command to test the finetuned model:

``` bash
python src/test/aime25_vllm.py --model_path "YOUR_MODEL_PATH"
```