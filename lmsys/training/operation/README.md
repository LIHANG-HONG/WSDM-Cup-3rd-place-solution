```
python3 run_extract_logit.py --config ./config/m005-qwen25-72b-listwise-choice-loss.yaml --filepath ../../data/train_lmsys_all.parquet --adapter train_ckpt/m005-qwen25-72b-listwise-choice-loss-no-pretrain_fold_100/checkpoint-13925 --output-dir ../../distil_logits/qwen_72b_no_pretrain

python3 run_extract_logit_tta.py --config ./config/m005-qwen25-72b-listwise-choice-loss.yaml --filepath ../../data/train_lmsys_all.parquet --adapter train_ckpt/m005-qwen25-72b-listwise-choice-loss-no-pretrain_fold_100/checkpoint-13925 --output-dir ../../distil_logits/qwen_72b_no_pretrain
```