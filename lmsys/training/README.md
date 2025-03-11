# prepare data(download fasttext model from [huggingface](https://huggingface.co/facebook/fasttext-language-identification))
```
python3 run_natural_language_infer.py --filepath ./train.parquet --exp-name language_fasttext --model-path ./model.bin --output-dir .
```


# download the model
```
python3 download_model.py --model-path Qwen/Qwen2.5-72B-Instruct
```

# run pretrain
```
export DS_ACCELERATOR=cuda

nohup accelerate launch --config_file=deepspeed_zero1.yaml run_pretrain.py --config ./config/p001-athene-listwise-choice-loss.yaml --output-dir ../../artifacts/pretrain_ckpt > train_log.txt 2>&1
python3 merge_and_unload.py --config ./config/p001-athene-listwise-choice-loss.yaml ../../artifacts/pretrain_ckpt/p001-athene-listwise-choice-loss/checkpoint-17915 --output-dir ../../artifacts/pretrain_ckpt/qwen_14b_pretrained_model

nohup accelerate launch --config_file=deepspeed_zero1.yaml run_pretrain.py --config ./config/p004-qwen-14b-listwise-choice-loss.yaml --output-dir ../../artifacts/pretrain_ckpt > train_log.txt 2>&1
python3 merge_and_unload.py --config ./config/p004-qwen-14b-listwise-choice-loss.yaml ../../artifacts/pretrain_ckpt/p004-qwen-14b-listwise-choice-loss/checkpoint-8957 --output-dir ../../artifacts/pretrain_ckpt/athene_pretrained_model
```

# run train
```
python3 run_train.py --config ./config/m001-qwen25-3b-listwise.yaml --filepath ../../data/train_lmsys_all.parquet --fold 0 --output-dir ../../artifacts/train_ckpt

python3 run_train_distil.py --config ./config/d002-qwen25-14b-listwise-choice-loss.yaml --filepath ../../data/train_lmsys_all.parquet --fold 0 --output-dir ../../artifacts/train_ckpt
python3 merge_and_unload.py --config ./config/d002-qwen25-14b-listwise-choice-loss.yaml ../../artifacts/train_ckpt/d002-qwen25-14b-listwise-choice-loss-binary-distil-t5-lr10-pretrain-model_fold_0/checkpoint-13925 --output-dir ../../artifacts/train_ckpt/qwen_14b_distil_t5_lr10_pretrain_model
```

# run evaluation
```
python3 run_oof.py --config ./config/d002-qwen25-14b-listwise-choice-loss.yaml --filepath ../../data/train_lmsys_all.parquet --fold 0 --adapter ../../artifacts/train_ckpt/d002-qwen25-14b-listwise-choice-loss-binary-distil-t5-lr10-pretrain-model_fold_0/checkpoint-13925 --output-dir .

python3 run_oof.py --config ./config/d002-qwen25-14b-listwise-choice-loss.yaml --filepath ../../data/train_lmsys_all.parquet --fold 0 --model-name-or-path ../../artifacts/train_ckpt/qwen_14b_distil_t5_lr10_pretrain_model --output-dir . --vllm
```

# run quantize
```
python3 quantize.py --config ./config/d002-qwen25-14b-listwise-choice-loss.yaml --filepath ../../data/train_lmsys_all.parquet --fold 0 --adapter ../../artifacts/train_ckpt/d002-qwen25-14b-listwise-choice-loss-binary-distil-t5-lr10-pretrain-model_fold_0/checkpoint-13925 --output-dir ../../artifacts/train_ckpt/qwen_14b_distil_t5_lr10_pretrain_model_quantize
```