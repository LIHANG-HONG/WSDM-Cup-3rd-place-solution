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
from lmsys.training.trainer import SFTDistillTrainer
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

def main(exp_config, filepath, fold, output_dir, debug):
    seed_everything(exp_config.seed)
    df = pd.read_parquet(filepath)
    df = df[df['winner_tie'] == 0].reset_index(drop=True)
    df['unique_id'] = list(range(len(df)))
    idx_to_id = {unique_id: id for unique_id, id in zip(df['unique_id'].values, df['id'].values)}
    #df = df[df['author']=='lmsys2'].reset_index(drop=True)
    df_train = df[df['fold'] != fold].reset_index(drop=True)
    df_valid = df[df['fold'] == fold].reset_index(drop=True)

    if debug:
        df_train = df_train.iloc[:100].reset_index(drop=True)
        df_valid = df_valid.iloc[:100].reset_index(drop=True)

    spec = importlib.util.spec_from_file_location("PromptManager","prompt/{prompt_name}/preference_prompt.py".format(prompt_name=exp_config.dataset_config.prompt_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    PromptManager = module.PromptManager

    model, tokenizer, peft_config = ModelManager.load_train_model(exp_config)

    train_prompts, train_answers, train_shuffles = DatasetManager.prepare_prompt(df_train, exp_config, PromptManager, tokenizer, is_train=True)
    valid_prompts, valid_answers, valid_shuffles = DatasetManager.prepare_prompt(df_valid, exp_config, PromptManager, tokenizer, is_train=False)
    print(train_prompts[0])
    df_train['whole_prompt'] = train_prompts
    df_valid['whole_prompt'] = valid_prompts
    df_train['answer'] = train_answers
    df_valid['answer'] = valid_answers
    df_train['shuffle'] = train_shuffles
    df_valid['shuffle'] = valid_shuffles

    train_dataset = Dataset.from_pandas(df_train)
    valid_dataset = Dataset.from_pandas(df_valid)
    # LLMの応答部分のスコアのみ計算
    collator = DataCollatorForCompletionOnlyLM(response_template=PromptManager.sep, tokenizer=tokenizer, mlm=False)

    max_seq_length = 4096 * 2
    trainer = SFTDistillTrainer(
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
            output_dir = os.path.join(output_dir, f'{exp_config.exp_name}_fold_{fold}'),
            save_total_limit=1,
            remove_unused_columns=False,
            dataset_text_field = 'whole_prompt',
            max_seq_length = max_seq_length,
            packing = False, # Can make training 5x faster for short sequences.
            dataset_num_proc = exp_config.num_workers,
        ),
    )
    trainer.post_init(exp_config, idx_to_id)
    remove_cols = [col for col in df_train.columns if col!='unique_id' and col!='shuffle']
    trainer.train_dataset = trainer.train_dataset.remove_columns(remove_cols)

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
    parser.add_argument('--filepath', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    exp_config = ExperimentConfig.parse_obj(config_dict)
    print(exp_config)
    debug = args.debug
    main(exp_config, args.filepath, args.fold, args.output_dir, debug)
