import pandas as pd
import argparse
import logging
import subprocess
import os

from ANN_MLM_scores import prepare_data

# -*- coding: utf-8 -*-

_logger = logging.getLogger(__name__)


def run_lmzoo(dataset_name, sentences, filedir, model):
    infile = os.path.join(filedir, f"into_lmzoo_{dataset_name}.txt")
    outfile = os.path.join(filedir, f"lmzoo_{dataset_name}_surprisal_{model}.txt")

    # write sentences to file (this is the input file to lmzoo)
    with open(infile, "w") as f:
        for item in sentences:
            f.write("%s\n" % item)

    with open(outfile, "w") as f:
        subprocess.run(["lm-zoo", "get-surprisals", f"{model}", f"{infile}"], stdout=f)

    return outfile


def get_surprisal_df(lmzoo_file, sentences, dataset_name, model, out_dir):
    data_df = pd.read_csv(lmzoo_file, sep='\t')

    sentence_ids = [i + 1 for i in range(len(sentences))]

    unk = []
    sentence_surprisals = []
    lmzoo_sentences = []
    for elm in sentence_ids:
        curr_df = data_df.loc[data_df['sentence_id'] == str(elm)]
        sent = ' '.join(list(curr_df['token']))
        # print(curr_df)
        sent_surp = curr_df['surprisal'].sum()
        # print(sentence, sent_surp)
        sentence_surprisals.append(sent_surp)
        lmzoo_sentences.append(sent)
        # print('\n')

        if "UNK" in sent:
            unk.append("unk")
        else:
            unk.append("nounk")

    out_df = pd.DataFrame({
        "orig_sentences": sentences,
        "surprisal": sentence_surprisals,
        "sentence": lmzoo_sentences,
        "unk": unk
    })
    savename = os.path.join(out_dir, f'{dataset_name}.{model}.surprisal_scores.txt')

    out_df.to_csv(savename, header=False, sep='\t')
    return out_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--model', type=str, default='tinylstm')
    parser.add_argument('--dataset_names', nargs='+', default=['ev1', 'dtfit', 'new-EventsAdapt'])
    args = parser.parse_args()

    out_dir = f'results/ANNs/'
    os.makedirs(out_dir, exist_ok=True)
    filedir = f'{args.model}_files/'
    os.makedirs(filedir, exist_ok=True)

    NOPUNCT = False #If set to True, we replicate original tinyLSTM results.

    # path to files in dataset/id_verbs subdirectory (position of the verb has to be given)
    dtfit = pd.read_csv('datasets/id_verbs/DTFit_vassallo_deps.verbs.txt', sep='\t', header=None)
    ev1 = pd.read_csv('datasets/id_verbs/ev1_deps.verbs.txt', sep='\t', header=None)
    events_adapt = pd.read_csv('datasets/id_verbs/newsentences_EventsAdapt.verbs.txt', sep='\t', header=None)

    datasets = {'ev1': prepare_data(ev1),
                'dtfit': prepare_data(dtfit),
                'new-EventsAdapt': prepare_data(events_adapt)}

    for dataset_name in args.dataset_names:
        _logger.info(f"*********** Processing: {dataset_name} ***********")
        _, sentences = datasets[dataset_name]
        if NOPUNCT:
            sentences = [sent.rstrip(".") for sent in sentences]
        _logger.info(f">> Running lmzoo")
        lmzoo_file = run_lmzoo(dataset_name, sentences, filedir, args.model)
        _logger.info(f">> Formatting and saving results")
        get_surprisal_df(lmzoo_file, sentences, dataset_name, args.model, out_dir)

    print("Done")

if __name__ == "__main__":
    main()






