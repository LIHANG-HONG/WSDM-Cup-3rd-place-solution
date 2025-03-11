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
from peft import LoraConfig, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed, TrainingArguments
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import torch.nn.functional as F
 
from lmsys.training.config import ExperimentConfig
from lmsys.training.trainer import SFTChoiceTrainer
from lmsys.training.dataset import DatasetManager
from lmsys.training.model import ModelManager

os.environ["WANDB_DISABLED"] = "true"

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

def main(exp_config, output_dir, debug):
    seed_everything(exp_config.seed)
    df1 = pd.read_parquet('../../data/2nd_place_external.parquet')
    df2 = pd.read_parquet('../../data/train_ultrachat_all.parquet')
    df = pd.concat([df1, df2], axis=0).reset_index(drop=True)
    df = df[df['winner_tie'] == 0].reset_index(drop=True)
    df_train = df

    if debug:
        df_train = df_train.iloc[:100].reset_index(drop=True)

    spec = importlib.util.spec_from_file_location("PromptManager","prompt/{prompt_name}/preference_prompt.py".format(prompt_name=exp_config.dataset_config.prompt_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    PromptManager = module.PromptManager

    model, tokenizer, peft_config = ModelManager.load_train_model(exp_config)

    train_prompts, train_answers, train_shuffles = DatasetManager.prepare_prompt(df_train, exp_config, PromptManager, tokenizer, is_train=True)
    print(train_prompts[0])
    df_train['whole_prompt'] = train_prompts
    df_train['answer'] = train_answers
    df_train['shuffle'] = train_shuffles

    train_dataset = Dataset.from_pandas(df_train)
    # LLMの応答部分のスコアのみ計算
    collator = DataCollatorForCompletionOnlyLM(response_template=PromptManager.sep, tokenizer=tokenizer, mlm=False)

    max_seq_length = 4096 * 2
    trainer = SFTChoiceTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        #eval_dataset=valid_dataset,
        data_collator=collator, 
        peft_config=peft_config, 
        args = SFTConfig(
            per_device_train_batch_size = exp_config.train_batch_size,
            gradient_accumulation_steps = exp_config.gradient_accumulation_steps,
            warmup_steps = int(exp_config.lr_scheduler_config.params['warmup_steps']),
            num_train_epochs = exp_config.max_epochs, # Set this for 1 full training run.
            learning_rate = exp_config.optimizer_config.params['lr'],
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 500,
            save_steps=500,
            optim = exp_config.optimizer_config.name,
            weight_decay = exp_config.optimizer_config.params['weight_decay'],
            lr_scheduler_type = exp_config.lr_scheduler_config.name,
            seed = exp_config.seed,
            output_dir = os.path.join(output_dir, f'{exp_config.exp_name}'),
            save_total_limit=1,
            remove_unused_columns=True,
            dataset_text_field = 'whole_prompt',
            max_seq_length = max_seq_length,
            packing = False, # Can make training 5x faster for short sequences.
            dataset_num_proc = exp_config.num_workers,
        ),
    )
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    if exp_config.llm_config.resume_from_checkpoint is not None:
        trainer_stats = trainer.train(resume_from_checkpoint=exp_config.llm_config.resume_from_checkpoint)
    else:
        trainer_stats = trainer.train()
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    exp_config = ExperimentConfig.parse_obj(config_dict)
    print(exp_config)
    debug = args.debug
    main(exp_config, args.output_dir, debug)
