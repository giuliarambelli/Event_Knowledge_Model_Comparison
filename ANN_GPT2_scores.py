from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM, AutoModelForCausalLM, AutoTokenizer

import torch
import pandas as pd

import argparse
import logging
import tqdm

import os

from ANN_MLM_scores import prepare_data

# -*- coding: utf-8 -*-

_logger = logging.getLogger(__name__)


def score(model, tokenizer, sentence, average=True):
    tokenize_input = tokenizer.tokenize(tokenizer.eos_token + sentence + tokenizer.eos_token)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss = model(tensor_input, labels=tensor_input)[0]
    if not average:
        sent_score = -loss.item() * len(tokenize_input) #sentence log likelihood
    else:
        sent_score = -loss.item() #average log likelihood per token
    return sent_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--dataset_names', nargs='+', default=['ev1', 'dtfit', 'new-EventsAdapt'])
    parser.add_argument('--average', action='store_true')
    parser.add_argument('--versions', nargs='+', default=['gpt2-medium', 'gpt2-xl', 'gpt-neo', 'gpt-j'])
    args = parser.parse_args()

    dict_tokenizers = {"gpt2-medium": GPT2Tokenizer.from_pretrained('gpt2-medium'),
                       "gpt2-xl": GPT2Tokenizer.from_pretrained('gpt2-xl'),
                       "gpt-neo": GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B'),
                       "gpt-j": AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")}

    dict_models = {"gpt2-medium": GPT2LMHeadModel.from_pretrained('gpt2-medium'),
                   "gpt2-xl": GPT2LMHeadModel.from_pretrained('gpt2-xl'),
                   "gpt-neo": GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B'),
                   "gpt-j": AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")}

    out_dir = f'results/ANNs/'
    os.makedirs(out_dir, exist_ok=True)

    # path to files in dataset/id_verbs subdirectory (position of the verb has to be given)
    dtfit = pd.read_csv('datasets/id_verbs/DTFit_vassallo_deps.verbs.txt', sep='\t', header=None)
    ev1 = pd.read_csv('datasets/id_verbs/ev1_deps.verbs.txt', sep='\t', header=None)
    events_adapt = pd.read_csv('datasets/id_verbs/newsentences_EventsAdapt.verbs.txt', sep='\t', header=None)

    datasets = {'ev1': prepare_data(ev1),
                'dtfit': prepare_data(dtfit),
                'new-EventsAdapt': prepare_data(events_adapt)}

    # Load pre-trained model (weights)
    for version in args.versions:
        model = dict_models[version]
        model.eval()
        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = dict_tokenizers[version]
        _logger.info(f"*********** Getting scores for model {version} ***********")
        for dataset_name in args.dataset_names:
            _logger.info(f"*********** Processing: {dataset_name} ***********")
            _, sentences = datasets[dataset_name]
            scores = []
            for sent in tqdm.tqdm(sentences):
                sent_score = score(model, tokenizer, sent, average=args.average)
                scores.append(sent_score)

            print(scores)
            out_name = os.path.join(out_dir, f'{dataset_name}.{version}.sentence-prob.txt')
            print(out_name)
            if args.average:
                out_name = out_name.rstrip(".txt") + ".average.txt"
            with open(out_name, "w") as fout:
                for i, sent, sent_score in zip(range(len(sentences)), sentences, scores):
                    fout.write(f'{i}\t{sent}\t{sent_score}\n')

if __name__ == "__main__":
    main()
