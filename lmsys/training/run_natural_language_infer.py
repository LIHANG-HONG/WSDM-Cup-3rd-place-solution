import argparse
import fasttext
import pandas as pd
import os
from langcodes import Language
from tqdm.auto import tqdm

def main(filepath, exp_name, model_path, output_dir):
    df = pd.read_parquet(filepath)
    model = fasttext.load_model(model_path)
    for col in ['prompt','response_a','response_b']:
        texts = df[col].tolist()
        langs = []
        for text in tqdm(texts):
            lang = model.predict(text.replace('\n',''))[0][0].replace('__label__','').split('_')[0]
            lang_name = Language.get(lang).display_name("en")
            langs.append(lang_name)
        df[col + f'_{exp_name}'] = langs
    print('saving file to', os.path.join(output_dir,os.path.basename(filepath)))
    df.to_parquet(os.path.join(output_dir,os.path.basename(filepath)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, required=True)
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()
    print(args)
    main(args.filepath, args.exp_name, args.model_path, args.output_dir)