import re
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import tqdm
import pandas as pd
import numpy as np
import logging
import argparse
import os
import sys

logger = logging.getLogger(__name__)


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


def get_tokenized_words(tokens):
    """
    list of lists of tokens of length words of the sentence
    each sublist carries the subtokens belonging to the given word at the same index
    """
    tokenized_words = []
    for ind, tok in enumerate(tokens):
        if not re.match("#", tok):
            curr_word = [tok]
        else:
            curr_word.append(tok)

        #end a word
        if ind == len(tokens) - 1:
            tokenized_words.append(curr_word)
        else:
            if not re.match("#", tokens[ind+1]):
                tokenized_words.append(curr_word)
    return tokenized_words


def get_masked_seq(tokens, tokenized_words, tokenizer):
    """
    Masking out sentences word by word. If a word is tokenized into multiple subtokens, mask out subtokens linearly.
    i.e. for "hooligan":
    * we predict "ho"  knowing that 2 masks are still to come
    * we predict "##oli" with "ho" in context & knowing a mask is still to come
    * we predict "##gan" with "ho" and "##oli" in context
    """
    nr_masks = [len(elm) for elm in tokenized_words]
    
    masked_seq = []
    i = 1 #start with 1 because we don't want to predict anything for CLS
    j = 1
    while i < len(tokens) - 1: #-1 since we don't want a mask for SEP
        curr_masked_seq = []
            
        if nr_masks[j] == 1:
            curr_masked_seq = [tokens[ind] if ind != i else tokenizer.mask_token for ind in range(len(tokens))]
            masked_seq.append(curr_masked_seq)
            i += 1
            j += 1

        else:
            for k in range(nr_masks[j]):
                curr_nr_masks = nr_masks[j] - k
                curr_masked_seq = [tokens[ind] if (ind < i+k or ind >= i+k+curr_nr_masks) else tokenizer.mask_token for ind in range(len(tokens))]
                masked_seq.append(curr_masked_seq)
            i += nr_masks[j]
            j += 1
            
    return masked_seq

def get_masks_for_sentence(sentence, tokenizer):
    """
    tokenize sentence and getting masking templates
    """
    logger.debug(sentence)
    words = sentence.split()
    tokens = tokenizer.tokenize(sentence, add_special_tokens=True)
    tokenized_words = get_tokenized_words(tokens)
    logger.debug(tokenized_words)
    logger.debug("\n")
#     assert len(tokenized_words) == len(words) + 3 #+1 because of final period, which is split off in tokenization + CLS + SEP
#     took out because sometimes apostrophes are split off
    masked_seq = get_masked_seq(tokens, tokenized_words, tokenizer)
#     for mask in masked_seq:
#         logger.debug(mask)
    return tokens, masked_seq


def prep_input(tokens, list_of_sents, masked_seq, tokenizer):
    """
    add mask tokens to input map (based on masking templates)
    """
    inputs = tokenizer(list_of_sents, padding=True, return_tensors="pt")
    for i in range(len(masked_seq)):
        for j in range(len(masked_seq[i])):
            if masked_seq[i][j] == tokenizer.mask_token:
                inputs['input_ids'][i][j] = tokenizer.mask_token_id  # i+1 due to ['CLS'] token, whose index is 101
        logger.debug(inputs['input_ids'][i])
        #inputs['attention_mask'][i][i+1] = 0
        logger.debug(f"PREDICTING TOKEN: {tokens[i+1]}")
        logger.debug([tokenizer.convert_ids_to_tokens(elm.item()) for elm in inputs['input_ids'][i]])
        logger.debug(f"{[elm.item() for elm in inputs['attention_mask'][i]]}")
        logger.debug("\n")
    return inputs


def get_probabilities(inputs, tokens, model, tokenizer):
    """
    calculate log probabilities for masked tokens
    """
    outputs = model(**inputs)[0]
    log_probs_fillers = []
    predict_tokens = tokens[1:-1]
    
    for batch_elem, token, index in zip(range(outputs.shape[0]), predict_tokens, range(1, len(predict_tokens) + 1)): #no need for CLS & SEP
        all_log_probs = torch.nn.functional.log_softmax(outputs[batch_elem, index])
        log_probs_fillers.append(all_log_probs[tokenizer.convert_tokens_to_ids(token)].item())
        logger.info(f"{tokenizer.convert_tokens_to_ids(token)} | {token} | {all_log_probs[tokenizer.convert_tokens_to_ids(token)].item()}\n")

    return log_probs_fillers


def get_sentence_score(sentence, model, tokenizer):
    """
    full pipeline for one sentence from sentence to sentence score
    """
    tokens, masked_seq = get_masks_for_sentence(sentence, tokenizer)
    
    nr_tokens = len(tokens)
    list_of_sents = [sentence] * nr_tokens
    
    inputs = prep_input(tokens, list_of_sents, masked_seq, tokenizer)
    log_probs_fillers = get_probabilities(inputs, tokens, model, tokenizer)
    
    sentence_score = sum(log_probs_fillers)
    
    logger.info(f" {sentence} | {sentence_score}\n")

    return sentence_score, tokens



def main():
    """
    This implements pseudo-log-likelihood scores of a sequence as proposed by Salazar et al. (2019), adjusted for better
    within-word tokenization
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--dataset_names', nargs='+', default=['ev1', 'dtfit', 'new-EventsAdapt'])
    parser.add_argument('--models', nargs='+', default=['bert-large-cased', 'roberta-large'])
    args = parser.parse_args()

    log_level = logging.getLevelName(args.log_level)
    logging.basicConfig(stream=sys.stdout, level=log_level)
    logger.info("Running with args %s", vars(args))

    out_dir = f'results/ANNs/'
    os.makedirs(out_dir, exist_ok=True)


    # path to files in dataset/id_verbs subdirectory (position of the verb has to be given)
    dtfit = pd.read_csv('datasets/id_verbs/DTFit_vassallo_deps.verbs.txt', sep='\t', header=None)
    ev1 = pd.read_csv('datasets/id_verbs/ev1_deps.verbs.txt', sep='\t', header=None)
    events_adapt = pd.read_csv('datasets/id_verbs/newsentences_EventsAdapt.verbs.txt', sep='\t', header=None)

    datasets = {'ev1': prepare_data(ev1),
                'dtfit': prepare_data(dtfit),
                'new-EventsAdapt': prepare_data(events_adapt)}
    
    for model_name in args.models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name).eval()
        logger.info(f"\n***********\nGETTING SCORES FOR {model_name}\n***********\n")
        
        for dataset_name in args.dataset_names:
            logger.info(f"\n***********\nGETTING SCORES FOR {model_name} | {dataset_name}\n***********\n")
            _, sentences = datasets[dataset_name]
            
            out_name = os.path.join(out_dir, f'{dataset_name}.{model_name}.new-sentence-PLL.txt')
            logger.info(f"\n*************\nSTARTING writing to file: {out_name}\n*************\n")
            fout = open(out_name, 'w')
            
            for ind, sent in tqdm.tqdm(enumerate(sentences)):
                sent_score, tokens = get_sentence_score(sent, model, tokenizer)
                nr_tokens = len([elm for elm in tokens if elm not in [tokenizer.cls_token , tokenizer.sep_token]])
                nr_words = len(sent.split())
                fout.write(f'{ind}\t{sent}\t{sent_score}\t{nr_tokens}\t{nr_words}\n')
            fout.close()
            logger.info(f"\n*************\FINISHED writing to file: {out_name}\n*************\n")
    
if __name__ == "__main__":
    main()