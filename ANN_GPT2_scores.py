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


def score(model, tokenizer, sentence):
    tokenize_input = tokenizer.tokenize(tokenizer.eos_token + sentence + tokenizer.eos_token)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss = model(tensor_input, labels=tensor_input)[0]
    nr_tok = len(tokenize_input)
    nr_word = len(sentence.split())
    sent_score = -loss.item() * nr_tok #sentence log-likelihood
    sent_score_avg_by_nrtoken = -loss.item() #average per-token LL
    sent_score_avg_by_nrwords = -loss.item() * nr_tok / nr_word
    return sent_score, sent_score_avg_by_nrtoken, sent_score_avg_by_nrwords, nr_tok, nr_word


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--dataset_names', nargs='+', default=['ev1', 'dtfit', 'new-EventsAdapt'])
    parser.add_argument('--versions', nargs='+', default=['gpt2-xl', 'gpt-j', 'gpt2-medium'])
    args = parser.parse_args()

    dict_tokenizers = {"gpt2-medium": GPT2Tokenizer.from_pretrained('gpt2-medium'),
                       "gpt2-xl": GPT2Tokenizer.from_pretrained('gpt2-xl'),
                       "gpt-neo": GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B'),
                       "gpt-j": AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")}

    dict_models = {"gpt2-medium": GPT2LMHeadModel.from_pretrained('gpt2-medium'),
                   "gpt2-xl": GPT2LMHeadModel.from_pretrained('gpt2-xl'),
                   "gpt-neo": GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B'),
                   "gpt-j": AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")}

    out_dir = 'results/ANNs/'
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
            
            sent_scores = []
            sent_scores_avg_by_nrtokens = []
            sent_scores_avg_by_nrwords = []
            nr_tokens = []
            nr_words = []
            
            for sent in tqdm.tqdm(sentences):
                sent_score, sent_score_avg_by_nrtoken, sent_score_avg_by_nrwords, nr_tok, nr_word = score(model, tokenizer, sent)
                sent_scores.append(sent_score)
                sent_scores_avg_by_nrtokens.append(sent_score_avg_by_nrtoken)
                sent_scores_avg_by_nrwords.append(sent_score_avg_by_nrwords)
                nr_tokens.append(nr_tok)
                nr_words.append(nr_word)

            out_name = os.path.join(out_dir, f'{dataset_name}.{version}.sentence-LL.txt')
            print(out_name)
            with open(out_name, "w") as fout:
                for i, sent, sent_score, nr_tok, nr_word in zip(range(len(sentences)), sentences, sent_scores, nr_tokens, nr_words):
                    fout.write(f'{i}\t{sent}\t{sent_score}\t{nr_tok}\t{nr_word}\n')
                    
            out_name = os.path.join(out_dir, f'{dataset_name}.{version}.sentence-LL.average_byNrTokens.txt')
            print(out_name)
            with open(out_name, "w") as fout:
                for i, sent, sent_score, nr_tok, nr_word in zip(range(len(sentences)), sentences, sent_scores_avg_by_nrtokens, nr_tokens, nr_words):
                    fout.write(f'{i}\t{sent}\t{sent_score}\t{nr_tok}\t{nr_word}\n')
                    
            out_name = os.path.join(out_dir, f'{dataset_name}.{version}.sentence-LL.average_byNrWords.txt')
            print(out_name)
            with open(out_name, "w") as fout:
                for i, sent, sent_score, nr_tok, nr_word in zip(range(len(sentences)), sentences, sent_scores_avg_by_nrwords, nr_tokens, nr_words):
                    fout.write(f'{i}\t{sent}\t{sent_score}\t{nr_tok}\t{nr_word}\n')

if __name__ == "__main__":
    main()