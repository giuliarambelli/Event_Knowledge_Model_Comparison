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

class TransformerModel:

    def __init__(self, transf_model, dict_tokenizers, dict_mlm_models):
        self.model_name = transf_model
        self.tokenizer = dict_tokenizers[transf_model]
        self.mlm_model = dict_mlm_models[transf_model].eval()

    def prepare_input(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        _logger.debug(f"{tokens}")
        nr_tokens = len(tokens)
        list_of_sents = [sentence] * nr_tokens
        return tokens, nr_tokens, list_of_sents

    def compute_filler_probabilities(self, tokens, nr_tokens, list_of_sents, l2r=False):
        """
        :param tokens: sentence tokens
        :param nr_tokens: nr. of sentence tokens
        :param list_of_sents: list containing nr_tokens-times input sentence > used for masking word-by-word
        :param l2r: whether to only use leftward context or not
        :return: list containing pseudo-log-likelihoods per token
        """
        inputs = self.tokenizer(list_of_sents, padding=True, return_tensors="pt")
        for i in range(nr_tokens):
            inputs['input_ids'][i][i + 1] = self.tokenizer.mask_token_id  # i+1 due to ['CLS'] token, whose index is 101
            _logger.debug(f"{inputs['input_ids'][i]}")
        _logger.debug(f"{inputs}")

        log_probs_fillers = []

        if not l2r:
            outputs = self.mlm_model(**inputs)[0]  # get loss | **inputs passes entire input dict to model

            for batch_elem, token, index in zip(range(outputs.shape[0]), tokens, range(1, nr_tokens + 1)):
                all_log_probs = torch.nn.functional.log_softmax(outputs[batch_elem, index])
                log_probs_fillers.append(all_log_probs[self.tokenizer.convert_tokens_to_ids(token)].item())
                _logger.debug(
                    f"{self.tokenizer.convert_tokens_to_ids(token)} | {token} | {all_log_probs[self.tokenizer.convert_tokens_to_ids(token)].item()}")

        else:  # if l2r
            for i in range(nr_tokens):
                inputs['attention_mask'][i][i + 1:] = 0  # i+1 due to ['CLS'] token, whose index is 101
                _logger.debug(f"{inputs['attention_mask'][i]}")

            outputs = self.mlm_model(**inputs)[0]

            for batch_elem, token, index in zip(range(outputs.shape[0]), tokens, range(1, nr_tokens + 1)):
                all_log_probs = torch.nn.functional.log_softmax(outputs[batch_elem, index])  # log_softmax
                log_probs_fillers.append(all_log_probs[self.tokenizer.convert_tokens_to_ids(token)].item())
                _logger.debug(
                    f"{self.tokenizer.convert_tokens_to_ids(token)} | {token} | {all_log_probs[self.tokenizer.convert_tokens_to_ids(token)].item()}")

        return log_probs_fillers


def get_sentence_score(model, sentences, l2r=False):
    """
    Run function to get sequence scores
    :param model: Transformer model
    :param sentences: list of sentences for which scores are to be computed
    :param average: Whether to length-normalize scores or not
    :param l2r: whether to only use leftward context or not
    :return: list of PLL/Unidirectional-PLL scores for sequences
    """
    sent_scores = []
    sent_scores_avg_by_nrtoken = []
    sent_scores_avg_by_nrwords = []
    
    for sent in tqdm.tqdm(sentences):
        tokens, nr_tokens, list_of_sents = model.prepare_input(sent)
        probabilities_fillers = model.compute_filler_probabilities(tokens, nr_tokens, list_of_sents, l2r=l2r)
        sentence_score = sum(probabilities_fillers)  # returns pseudo log likelihood score
        #
        sentence_score_avg_by_nrtoken = sentence_score / nr_tokens
        #
        nr_words = len(sent.split())
        sentence_score_avg_by_nrwords = sentence_score / nr_words
        #
        sent_scores.append(sentence_score)
        sent_scores_avg_by_nrtoken.append(sentence_score_avg_by_nrtoken)
        sent_scores_avg_by_nrwords.append(sentence_score_avg_by_nrwords)
        #
        _logger.info(f" {tokens} | {sentence_score}")

    return sent_scores, sent_scores_avg_by_nrtoken, sent_scores_avg_by_nrwords


def get_word_score(model, model_name, sentences, verb_indices=None, average=False, l2r=False):
    """
    Run function to get word scores
    :param model: Transformer model
    :param model_name: model version
    :param sentences: list of sentences for which scores are to be computed
    :param verb_indices: list of verb indices
    :param average: Whether to length-normalize scores or not
    :param l2r: whether to only use leftward context or not
    :return: list of log probability scores for target word per sequence (either verb or last word)
    """
    word_scores = []

    if verb_indices is None:  # if last word probability
        word_indices = [-1] * len(sentences)
    else:
        word_indices = verb_indices

    for ind, sent in zip(word_indices, sentences):

        word = sent.split()[ind]

        if word.endswith("."):  # exclude final period
            word = word.rstrip(".")

        if model_name.startswith("roberta"):
            word_tokens = model.tokenizer.tokenize(f" {word}")  # account for Byte-level BPE tokenizer special behavior
        else:
            word_tokens = model.tokenizer.tokenize(word)

        sent_tokens = model.tokenizer.tokenize(sent)
        _logger.debug(f"{word_tokens}, {sent_tokens}")

        word_starts = [i for i, x in enumerate(sent_tokens) if x == word_tokens[0]]
        if len(word_starts) == 1:
            word_start = word_starts[0]
            word_span = len(word_tokens)
            _logger.debug(f"{word_start}, {word_span}")
            assert all(sent_tokens[word_start + i] == word_tokens[i] for i in range(word_span))
        else:
            for ind, word_start in enumerate(word_starts):
                word_start = word_starts[ind]
                word_span = len(word_tokens)
                _logger.debug(f"{word_start}, {word_span}")
                if all(sent_tokens[word_start + i] == word_tokens[i] for i in range(word_span)):
                    break

        tokens, nr_tokens, list_of_sents = model.prepare_input(sent)
        probabilities_fillers = model.compute_filler_probabilities(tokens, nr_tokens, list_of_sents, l2r=l2r)
        _logger.info(f" {word} | {probabilities_fillers[word_start:word_start + word_span]}")
        word_score = sum(probabilities_fillers[word_start:word_start + word_span])  # returns log likelihood score
        if average:
            word_score = word_score / word_span
        word_scores.append(word_score)

    return word_scores


def prepare_data(df):
    """
    :param df: dataset containing sentences and the word position id of the word per sentence
    :return: list of verb position ids & list of preprocessed sentences
    """
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
    """
    Main run function for one of 4 input score types:
    * w2w: This implements pseudo-log-likelihood scores of a sequence as proposed by Salazar et al. (2019)
            Results obtained are equivalent as from 
    * l2r: This implements pseudo-log-likelihood scores with access only to the unidirectional context
            (Implemented via attention masks, "most like" GPT2 scoring, but unlike MLM's training objective)
    * verb: Getting the log probability of a verb, relies on verb_ids from df
    * last_word: Getting the log probability of the last word in the sequence

    * average flag can be set to get
        - word-length normalized PLL scores for sequences (see Salazar et al. (2019), Appendix C)
        - word-scores normalized by subtoken number
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--which_score', nargs='+', default=['w2w', 'l2r', 'verb', 'last_word'])
    parser.add_argument('--dataset_names', nargs='+', default=['ev1', 'dtfit', 'new-EventsAdapt'])
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
        _logger.info(f"*********** Getting scores for model {model_name} ***********")
        for dataset_name in args.dataset_names:
            _logger.info(f"*********** Processing: {dataset_name} ***********")
            for task in args.which_score:
                print(f"{model_name} | {dataset_name} | {task}")

                verb_ids, sentences = datasets[dataset_name]

                if task == 'w2w':
                    _logger.info(f">> Getting w2w sentence scores")
                    sent_scores, sent_scores_avg_by_nrtoken, sent_scores_avg_by_nrwords = get_sentence_score(model, sentences, l2r=False)
                    savename = 'sentence-PLL'
                elif task == 'l2r':
                    _logger.info(f">> Getting l2r sentence scores")
                    sent_scores, sent_scores_avg_by_nrtoken, sent_scores_avg_by_nrwords = get_sentence_score(model, sentences, l2r=True)
                    savename = 'sentence-l2r-PLL'
                elif task == 'verb':
                    _logger.info(f">> Getting verb scores")
                    scores = get_word_score(model, model_name, sentences, verb_indices=verb_ids, average=True, l2r=False)
                    savename = 'verb-PLL'
                elif task == 'last_word':
                    _logger.info(f">> Getting last word scores")
                    scores = get_word_score(model, model_name, sentences, verb_indices=None, average=True, l2r=False)
                    savename = 'last-word-PLL'
                else:
                    raise NotImplementedError(f"Task {task} not defined!")

                out_name = os.path.join(out_dir, f'{dataset_name}.{model_name}.{savename}.txt')

                if task in ["w2w","l2r"]:
                    print(out_name)

                    with open(out_name, "w") as fout:
                        for i, sent, sent_score in zip(range(len(sentences)), sentences, sent_scores):
                            fout.write(f'{i}\t{sent}\t{sent_score}\n')

                    out_name = os.path.join(out_dir, f'{dataset_name}.{model_name}.{savename}.sentence_surp.average_byNrTokens.txt')
                    print(out_name)
                    with open(out_name, "w") as fout:
                        for i, sent, sent_score in zip(range(len(sentences)), sentences, sent_scores_avg_by_nrtoken):
                            fout.write(f'{i}\t{sent}\t{sent_score}\n')

                    out_name = os.path.join(out_dir, f'{dataset_name}.{model_name}.{savename}.sentence_surp.average_byNrWords.txt')
                    print(out_name)
                    with open(out_name, "w") as fout:
                        for i, sent, sent_score in zip(range(len(sentences)), sentences, sent_scores_avg_by_nrwords):
                            fout.write(f'{i}\t{sent}\t{sent_score}\n')

                else:
                    with open(out_name, "w") as fout:
                        for i, sent, score in zip(range(len(sentences)), sentences, scores):
                            fout.write(f'{i}\t{sent}\t{score}\n')


if __name__ == "__main__":
    main()


