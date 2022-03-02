from transformers import BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM
import torch
import re
import os
import sys
import pandas as pd
import argparse
import logging
import tqdm

# -*- coding: utf-8 -*-

_logger = logging.getLogger(__name__)
debug = False

class TransformerModel:

    def __init__(self, transf_model, dict_tokenizers, dict_mlm_models):
        self.model_name = transf_model
        self.tokenizer = dict_tokenizers[transf_model]
        self.mlm_model = dict_mlm_models[transf_model].eval()

    def prepare_input(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        print(tokens)
        nr_tokens = len(tokens)
        list_of_sents = [sentence] * nr_tokens
        return tokens, nr_tokens, list_of_sents

    def compute_filler_probabilities(self, tokens, nr_tokens, list_of_sents, l2r=False):
        if debug: print(list_of_sents)
        inputs = self.tokenizer(list_of_sents, padding=True, return_tensors="pt")
        for i in range(nr_tokens):
            inputs['input_ids'][i][i + 1] = self.tokenizer.mask_token_id  # i+1 due to ['CLS'] token, whose index is 101
            if debug: print(inputs['input_ids'][i])
        if debug: print(inputs)

        log_probs_fillers = []

        if not l2r:
            outputs = self.mlm_model(**inputs)[
                0]  # get loss, with **inputs pass entire input, with attention mask to model
            if debug: print(outputs.shape)

            for batch_elem, token, index in zip(range(outputs.shape[0]), tokens, range(1, nr_tokens + 1)):
                if debug: print(index, token)
                all_log_probs = torch.nn.functional.log_softmax(outputs[batch_elem, index])
                log_probs_fillers.append(all_log_probs[self.tokenizer.convert_tokens_to_ids(token)].item())
                _logger.debug(
                    f"{self.tokenizer.convert_tokens_to_ids(token)} | {token} | {all_log_probs[self.tokenizer.convert_tokens_to_ids(token)].item()}")

        else:  # if l2r
            for i in range(nr_tokens):
                if debug: print(inputs['attention_mask'][i])
                inputs['attention_mask'][i][i + 1:] = 0  # i+1 due to ['CLS'] token, whose index is 101
                if debug: print(inputs['attention_mask'][i])

            outputs = self.mlm_model(**inputs)[0]
            if debug: print(outputs.shape)

            for batch_elem, token, index in zip(range(outputs.shape[0]), tokens, range(1, nr_tokens + 1)):
                if debug: print(index, token)
                all_log_probs = torch.nn.functional.log_softmax(outputs[batch_elem, index])  # log_softmax
                log_probs_fillers.append(all_log_probs[self.tokenizer.convert_tokens_to_ids(token)].item())
                _logger.debug(
                    f"{self.tokenizer.convert_tokens_to_ids(token)} | {token} | {all_log_probs[self.tokenizer.convert_tokens_to_ids(token)].item()}")

        return log_probs_fillers


def get_sentence_score(model, sentences, average=False, l2r=False):
    sentence_scores = []
    for sent in tqdm.tqdm(sentences):
        tokens, nr_tokens, list_of_sents = model.prepare_input(sent)
        probabilities_fillers = model.compute_filler_probabilities(tokens, nr_tokens, list_of_sents, l2r=l2r)
        sentence_score = sum(probabilities_fillers)  # returns log likelihood score
        if average:
            nr_words = len(sent.split())
            sentence_score = sentence_score / nr_words
        sentence_scores.append(sentence_score)
        print("*****")

    return sentence_scores


def get_word_score(model, sentences, verb_indices=None, average=False, l2r=False):
    word_scores = []

    if verb_indices is None:  # if last word probability
        word_indices = [-1] * len(sentences)
    else:
        word_indices = verb_indices

    for ind, sent in zip(word_indices, sentences):
        if debug: print(sent)

        word = sent.split()[ind]

        if word.endswith("."):  # exclude final period
            word = word.rstrip(".")
        print(f"Getting probability for word: {word}")

        word_tokens = model.tokenizer.tokenize(word)
        sent_tokens = model.tokenizer.tokenize(sent)
        if debug: print(word_tokens, sent_tokens)
        word_start = sent_tokens.index(word_tokens[0])
        word_span = len(word_tokens)
        if debug: print(word_start, word_span)
        for i in range(word_span):
            assert sent_tokens[word_start + i] == word_tokens[i]

        tokens, nr_tokens, list_of_sents = model.prepare_input(sent)
        probabilities_fillers = model.compute_filler_probabilities(tokens, nr_tokens, list_of_sents, l2r=l2r)
        print(probabilities_fillers[word_start:word_start + word_span])
        word_score = sum(probabilities_fillers[word_start:word_start + word_span])  # returns log likelihood score
        if average:
            word_score = word_score / word_span
        word_scores.append(word_score)
        print("*****")

    return word_scores


def prepare_data(df):
    verb_ids = []
    sents = []
    for index, row in df.iterrows():
        row[1] = re.sub("\s*\.$", "", row[1]) #strip any number of whitespace plus final period
        row[1] += "."
        assert row[1].endswith(".") and not row[1].endswith(" .")
        sents.append(row[1])
        verb_ids.append(row[2])
    return verb_ids, sents

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--which_score', nargs='+', default=['w2w', 'l2r', 'verb', 'last_word'])
    parser.add_argument('--dataset_names', nargs='+', default=['ev1', 'dtfit', 'new-EventsAdapt'])
    parser.add_argument('--average', action='store_true')
    parser.add_argument('--models', nargs='+', default=['bert-large-cased', 'roberta-large'])
    args = parser.parse_args()

    log_level = logging.getLevelName(args.log_level)
    logging.basicConfig(stream=sys.stdout, level=log_level)
    _logger.info("Running with args %s", vars(args))

    out_dir = f'results/ANNs/'
    os.makedirs(out_dir, exist_ok=True)

    dict_tokenizers = {"bert-large-cased": BertTokenizer.from_pretrained('bert-large-cased'),
                        "roberta-large": RobertaTokenizer.from_pretrained('roberta-large')}

    dict_mlm_models = {"bert-large-cased": BertForMaskedLM.from_pretrained('bert-large-cased'),
                        "roberta-large": RobertaForMaskedLM.from_pretrained('roberta-large')}

    # path to files in dataset/id_verbs subdirectory (position of the verb has to be given)
    dtfit = pd.read_csv('datasets/id_verbs/DTFit_vassallo_deps.verbs.txt', sep='\t', header=None)
    ev1 = pd.read_csv('datasets/id_verbs/ev1_deps.verbs.txt', sep='\t', header=None)
    events_adapt = pd.read_csv('datasets/id_verbs/newsentences_EventsAdapt.verbs.txt', sep='\t', header=None)

    datasets = {'ev1': prepare_data(ev1),
                'dtfit': prepare_data(dtfit),
                'new-EventsAdapt': prepare_data(events_adapt)}

    for model_name in args.models:
        model = TransformerModel(model_name, dict_tokenizers, dict_mlm_models)
        model.eval()
        _logger.info(f"*********** Getting scores for model {model_name} ***********")
        for dataset_name in args.dataset_names:
            _logger.info(f"*********** Processing: {dataset_name} ***********")
            for task in args.which_score:
                verb_ids, sentences = datasets[dataset_name]

                if task == 'w2w':
                    _logger.info(f">> Getting w2w sentence scores")
                    scores = get_sentence_score(model, sentences, average=args.average, l2r=False)
                    savename = 'sentence-PLL'
                elif task == 'l2r':
                    _logger.info(f">> Getting l2r sentence scores")
                    scores = get_sentence_score(model, sentences, average=args.average, l2r=True)
                    savename = 'sentence-l2r-PLL'
                elif task == 'verb':
                    _logger.info(f">> Getting verb scores")
                    scores = get_word_score(model, sentences, verb_indices=verb_ids, average=args.average, l2r=False)
                    savename = 'verb-PLL'
                elif task == 'last_word':
                    _logger.info(f">> Getting last word scores")
                    scores = get_word_score(model, sentences, verb_indices=None, average=args.average, l2r=False)
                    savename = 'last-word-PLL'
                else:
                    raise NotImplementedError(f"Task {task} not defined!")

                print(scores)
                out_name = os.path.join(out_dir, f'{dataset_name}.{model_name}.{savename}.txt')
                if args.average:
                    out_name = out_name.rstrip(".txt") + ".average.txt"
                with open(out_name, "w") as fout:
                    for i, sent, score in zip(range(len(sentences)), sentences, scores):
                        fout.write(f'{i}\t{sent}\t{score}\n')

if __name__ == "__main__":
    main()


