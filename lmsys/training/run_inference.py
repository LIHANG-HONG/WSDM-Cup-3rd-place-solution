import argparse
import gc
import os
import importlib
import numpy as np
import pandas as pd
import random
import torch
import yaml
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, LogitsProcessor
import torch.nn.functional as F
from vllm import LLM, SamplingParams
import json

from lmsys.training.config import ExperimentConfig
from lmsys.training.dataset import DatasetManager
from lmsys.training.model import ModelManager

class DigitLogitsProcessor(LogitsProcessor):
    def __init__(self, keep):
        self.allowed_ids = keep
        
    def __call__(self, input_ids, scores):
        scores[self.allowed_ids] += 100
        return scores

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    set_seed(seed)
    print('> SEEDING DONE')

def main(exp_config, filepath, model_name_or_path, output, chunk_size, cache_file):
    seed_everything(exp_config.seed)
    print('overriding model_name_or_path with', model_name_or_path)
    exp_config.llm_config.backbone = model_name_or_path

    df = pd.read_parquet(filepath)
    df_valid = df
    spec = importlib.util.spec_from_file_location("PromptManager","prompt/{prompt_name}/preference_prompt.py".format(prompt_name=exp_config.dataset_config.prompt_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    PromptManager = module.PromptManager

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if cache_file is None:
        valid_prompts, valid_answers = DatasetManager.prepare_test_prompt(df_valid, exp_config, PromptManager, model_name_or_path)
    else:
        print('using cache file: ', cache_file)
        df_cache = pd.read_parquet(cache_file)
        valid_prompts = df_cache['whole_prompt'].values.tolist()
    print(valid_prompts[0])
    df_valid['whole_prompt'] = valid_prompts
    
    keep = []
    for x in ['A','B']:
        c = tokenizer.encode(x,add_special_tokens=False)[0]
        keep.append(c)
    
    outputs_a = []
    outputs_b = []
    num_gpu = torch.cuda.device_count()
    llm = LLM(
        model_name_or_path,
        tensor_parallel_size=num_gpu,
        gpu_memory_utilization=0.98,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=6656,
        disable_log_stats=True,
        max_num_seqs=256,
        enable_prefix_caching= True,
        swap_space=5,
    )
    tokenizer = llm.get_tokenizer()
    keep = []
    for x in ['A','B']:
        c = tokenizer.encode(x,add_special_tokens=False)[0]
        keep.append(c)
    logits_processors = [DigitLogitsProcessor(keep)]
    sampling_params = SamplingParams(
        n=1,  # Number of output sequences to return for each prompt.
        top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
        temperature=0,  # randomness of the sampling
        seed=exp_config.seed, # Seed for reprodicibility
        skip_special_tokens=True,  # Whether to skip special tokens in the output.
        max_tokens=1,  # Maximum number of tokens to generate per output sequence.
        logits_processors=logits_processors,
        logprobs = 2,
    )
    all_texts = df_valid['whole_prompt'].values
    chunks = len(all_texts) // chunk_size if len(all_texts) % chunk_size == 0 else len(all_texts) // chunk_size + 1
    for chunk in range(chunks):
        print(f'Chunk {chunk+1}/{chunks}')
        responses = llm.generate(all_texts[chunk*chunk_size:(chunk+1)*chunk_size], sampling_params=sampling_params)
        for response in responses:
            try:
                response_logprobs = response.outputs[0].logprobs[0]
                logprobs = []
                for k in keep:
                    if k in response_logprobs:
                        logprobs.append(response_logprobs[k].logprob)
                    else:
                        logprobs.append(-100)
            except:
                logprobs = [-100, -100]
            outputs_a.append(logprobs[0])
            outputs_b.append(logprobs[1])

    df_valid['model_a_logit'] = outputs_a
    df_valid['model_b_logit'] = outputs_b
    df_valid.to_parquet(output)

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--filepath', type=str, required=True)
    parser.add_argument('--model-name-or-path', type=str, required=False, default=None)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--chunk', type=int, required=False, default=10000)
    parser.add_argument('--cache-file', type=str, required=False, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    exp_config = ExperimentConfig.parse_obj(config_dict)
    print(exp_config)
    main(exp_config, args.filepath, args.model_name_or_path, args.output, args.chunk, args.cache_file)
