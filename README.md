# Event_Knowledge_Model_Comparison

A series of experiments with the aim of study which computational model performs better among others.

## Requirements
- numpy
- pandas
- math
- tqdm
- tensorflow (version >= 2.0
- pytokenizations
- sentencepiece
- transformers

## Datasets
- DTFit
- ev1
- newsentences_EventsAdapt

## Models

- 2 baselines: 
   + **PPMI (structured input, input annotated with grammatical roles)**
   + **ngram sentence surprisal**



- **ANNs**
   + [ANNs_predict-token-masked.ipynb](https://github.com/giuliarambelli/Event_Knowledge_Model_Comparison/blob/master/ANNs_predict-token-masked.ipynb)
   
     This notebook contains the code to compute the probability of a masked word in a sentence, using a bidirectional ANNs (BERT, RoBERTa, XLNET).
   It performs 2 tasks: 1) Verb prediction  and 2) Last word prediction.
   + [ANNs_sequential-word-prediction.ipynb](https://github.com/giuliarambelli/Event_Knowledge_Model_Comparison/blob/master/ANNs_sequential-word-prediction.ipynb)
   This notebook contains the code to compute the probability of a sentence from the probability of single words using a bidirectional ANNs.
   It performs 2 tasks: 1) Sequential word prediction, or Pseudo-log likelihood (start from unmasked sentence, mask each word iteratively), and 2) Left-to-right generation, or Sequential Sampling (start from completely masked sentence and unmask words left-to-right)

Sequential word prediction (unidirectional ANNs)


