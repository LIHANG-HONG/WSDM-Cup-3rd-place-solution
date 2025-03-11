import datasets
import pandas as pd
import json
from datasets import load_dataset, concatenate_datasets
import random
from tqdm import tqdm
random.seed(42)
dataset1 = load_dataset("argilla/Capybara-Preferences", split="train")
df_lmsys21k = pd.read_csv('lmsys-33k-deduplicated.csv')
columns = df_lmsys21k.columns
new_df = []
ct = 0
for dd in tqdm(dataset1):
    response_chosen = dd['chosen']
    response_rejected = dd['rejected']
    prompt = []
    r_chosen = []
    r_rejected = []
    for i in range(len(response_chosen)):
        if response_chosen[i]['role'] == 'user':
            prompt.append(response_chosen[i]['content'])
        if response_chosen[i]['role'] == 'assistant':
            r_chosen.append(response_chosen[i]['content'])
        if response_rejected[i]['role'] == 'assistant':
            r_rejected.append(response_rejected[i]['content'])
    if random.random() > 0.5:
        winner_model_a = 1
        winner_model_b = 0
        response_a = r_chosen
        response_b = r_rejected
    else:
        winner_model_a = 0
        winner_model_b = 1
        response_a = r_rejected
        response_b = r_chosen
    new_id = "argilla/Capybara-Preferences" + '-' + str(ct)
    new_df.append([new_id, 'unknown', 'unknown', json.dumps(prompt), json.dumps(response_a), json.dumps(response_b), winner_model_a, winner_model_b, 0])
    ct += 1
    # print('a')
new_df = pd.DataFrame(new_df, columns=columns)
new_df.to_csv('other-external/Capybara.csv', index=False)
print('a')