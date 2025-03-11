from tqdm.auto import tqdm
import numpy as np 
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer

class DatasetManager(object):
    @staticmethod
    def prepare_prompt(df, exp_config, prompt_cls, tokenizer, is_train=False, prompt_max_seq_len=2048, response_max_seq_len=2048, synthetic_max_seq_len=512):
        if 'winner_model_a' not in df.columns:
            df = df.copy()
            df['winner_model_a'] = 1
            df['winner_model_b'] = 0
            print('No winner_model_a column found, setting all to model_a. Be sure it is in test phase!')
        
        dataset_config = exp_config.dataset_config
        must_cols = dataset_config.must_cols
        synthetic_cols = dataset_config.synthetic_cols
        preference_prompt_template = prompt_cls.preference_prompt_train_template if is_train else prompt_cls.preference_prompt_predict_template
        response_template = prompt_cls.response_template

        prompts = []
        answers = []
        shuffles = []
        for i in tqdm(range(len(df))):
            row = df.iloc[i]
            prompt = row['prompt']
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(tokens)
            if prompt_len > prompt_max_seq_len:
                prompt = tokenizer.decode(tokens[:prompt_max_seq_len])
            language = row['prompt_language_fasttext']

            responses = [row[col] for col in must_cols]
            responses = [tokenizer.decode(tokenizer.encode(response, add_special_tokens=False, max_length=response_max_seq_len, truncation=True)) for response in responses]
            response_langs = [row['response_a_language_fasttext'], row['response_b_language_fasttext']]
            labels = [0 for _ in range(len(responses))]
            if row['winner_model_a'] == 1:
                labels[0] = 1
            elif row['winner_model_b'] == 1:
                labels[1] = 1
            else:
                raise ValueError('No winner found')

            shuffle = False
            if is_train:
                synthetic_responses = [row[col] for col in synthetic_cols if (isinstance(row[col], str)) and (row[col]!='')]
                k = min([np.random.randint(0, len(synthetic_responses)+1),3])
                k = 0
                if k > 0:
                    synthetic_responses_choice = np.random.choice(synthetic_responses, k, replace=False).tolist()
                    synthetic_responses_choice = [tokenizer.decode(tokenizer.encode(response, add_special_tokens=True, max_length=synthetic_max_seq_len)) for response in synthetic_responses_choice]
                    responses += synthetic_responses_choice
                    labels += [0 for _ in range(k)]
                rand_perm = np.random.permutation(len(responses))
                responses = [responses[i] for i in rand_perm]
                labels = [labels[i] for i in rand_perm]
                response_langs = [response_langs[i] for i in rand_perm]
                if rand_perm[0] != 0:
                    shuffle = True
            
            letters = [chr(i+65) for i in range(len(responses))]
            idx = None
            for k, label in enumerate(labels):
                if label == 1:
                    idx = k
                    break
            answer = letters[idx]
            response_str = ''
            for j in range(len(responses)):
                response = response_template.format(choice=letters[j], response=responses[j], response_language=response_langs[j])
                response_str += response
            whole_prompt = preference_prompt_template.format(prompt=prompt, language=language, responses=response_str, answer=answer)
            prompts.append(whole_prompt)
            answers.append(answer)
            shuffles.append(shuffle)
        return prompts, answers, shuffles

    @staticmethod
    def _prepare_test_prompt(df, exp_config, prompt_cls, model_name_or_path, prompt_max_seq_len=2048, response_max_seq_len=2048):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if 'winner_model_a' not in df.columns:
            df = df.copy()
            df['winner_model_a'] = 1
            df['winner_model_b'] = 0
            print('No winner_model_a column found, setting all to model_a. Be sure it is in test phase!')
        
        dataset_config = exp_config.dataset_config
        must_cols = dataset_config.must_cols
        synthetic_cols = dataset_config.synthetic_cols
        preference_prompt_template = prompt_cls.preference_prompt_predict_template
        response_template = prompt_cls.response_template

        prompts = []
        answers = []
        for i in range(len(df)):
            row = df.iloc[i]
            prompt = row['prompt']
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(tokens)
            if prompt_len > prompt_max_seq_len:
                prompt = tokenizer.decode(tokens[:prompt_max_seq_len])
            language = row['prompt_language_fasttext']

            responses = [row[col] for col in must_cols]
            responses = [tokenizer.decode(tokenizer.encode(response, add_special_tokens=False, max_length=response_max_seq_len, truncation=True)) for response in responses]

            labels = [0 for _ in range(len(responses))]
            if row['winner_model_a'] == 1:
                labels[0] = 1
            elif row['winner_model_b'] == 1:
                labels[1] = 1
            else:
                raise ValueError('No winner found')

            letters = [chr(i+65) for i in range(len(responses))]
            idx = None
            for k, label in enumerate(labels):
                if label == 1:
                    idx = k
                    break
            answer = letters[idx]
            response_str = ''
            for j in range(len(responses)):
                response = response_template.format(choice=letters[j], response=responses[j])
                response_str += response
            whole_prompt = preference_prompt_template.format(prompt=prompt, language=language, responses=response_str, answer=answer)
            prompts.append(whole_prompt)
            answers.append(answer)
        return prompts, answers

    @staticmethod
    def prepare_test_prompt(df, exp_config, prompt_cls, model_name_or_path, prompt_max_seq_len=2048, response_max_seq_len=2048):
        max_num_workers = 4
        if len(df) < max_num_workers:
            num_workers = len(df)
        else:
            num_workers = max_num_workers
        chunk_size = len(df) // num_workers
        dfs_chunk = []
        for i in range(num_workers):
            if i == num_workers - 1:
                chunk_df = df.iloc[i*chunk_size:].reset_index(drop=True)
            else:
                chunk_df = df.iloc[i*chunk_size:(i+1)*chunk_size].reset_index(drop=True)
            dfs_chunk.append(chunk_df)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(DatasetManager._prepare_test_prompt, dfs_chunk, [exp_config]*num_workers, [prompt_cls]*num_workers, [model_name_or_path]*num_workers, [prompt_max_seq_len]*num_workers, [response_max_seq_len]*num_workers)
            results_list = list(results)
            prompts = []
            answers = []
            for chunk_prompts, chunk_answers in results_list:
                prompts.extend(chunk_prompts)
                answers.extend(chunk_answers)
        return prompts, answers

    @staticmethod
    def prepare_quant_prompt(df, exp_config, prompt_cls, tokenizer):
        if 'winner_model_a' not in df.columns:
            df = df.copy()
            df['winner_model_a'] = 1
            df['winner_model_b'] = 0
            print('No winner_model_a column found, setting all to model_a. Be sure it is in test phase!')
        
        dataset_config = exp_config.dataset_config
        must_cols = dataset_config.must_cols
        synthetic_cols = dataset_config.synthetic_cols
        preference_prompt_template = prompt_cls.preference_prompt_train_template
        response_template = prompt_cls.response_template

        prompts = []
        answers = []
        for i in tqdm(range(len(df))):
            row = df.iloc[i]
            prompt = row['prompt']
            language = row['prompt_language_fasttext']

            responses = [row[col] for col in must_cols]
            response_langs = [row['response_a_language_fasttext'], row['response_b_language_fasttext']]
            labels = [0 for _ in range(len(responses))]
            if row['winner_model_a'] == 1:
                labels[0] = 1
            elif row['winner_model_b'] == 1:
                labels[1] = 1
            else:
                raise ValueError('No winner found')
            letters = [chr(i+65) for i in range(len(responses))]
            idx = None
            for k, label in enumerate(labels):
                if label == 1:
                    idx = k
                    break
            answer = letters[idx]
            response_str = ''
            for j in range(len(responses)):
                response = response_template.format(choice=letters[j], response=responses[j], response_language=response_langs[j])
                response_str += response
            whole_prompt = preference_prompt_template.format(prompt=prompt, language=language, responses=response_str, answer=answer)
            prompts.append(whole_prompt)
            answers.append(answer)
        return prompts, answers