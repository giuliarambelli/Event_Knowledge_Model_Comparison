# Computational Models Comparison on Event Knowlege tasks

A series of experiments with the aim of study which computational model performs better among others.

## Requirements
- numpy
- pandas
- math
- tqdm
- tensorflow (version >= 2.0)
- pytokenizations
- sentencepiece
- transformers
- lm-scorer

## Datasets
- DTFit
- ev1
- newsentences_EventsAdapt

## Models

- **Baselines**
   + [baselines_PPMI_structured_and_unstructured.ipynb](https://github.com/giuliarambelli/Event_Knowledge_Model_Comparison/blob/master/baselines_PPMI_structured_and_unstructured.ipynb) 

   It consist of 2 baselines:
   
   1. **PPMI** (structured input, input annotated with grammatical roles)    
   
      After extracting triples < verbal head, nominal dependent, relation > from the corpora (with a frequency >= 2), we compute the PPMI as follows 
      (N= total frequency of all triples).
      ![ppmi baseline 1](https://github.com/giuliarambelli/Event_Knowledge_Model_Comparison/blob/master/img/baseline1.gif)

   2. **ngram sentence surprisal**
   
      We select lemmas of minimum frequency 50 and extract bigrams of words (mechanism: for each word in the sentence, we take the 10 words to its right and then advance one position (minimum bigram frequency is 5). We then compute the Pointwise PMI of a given bigram (no syntactic information).
      ![ppmi baseline 2](https://github.com/giuliarambelli/Event_Knowledge_Model_Comparison/blob/master/img/baseline2.gif)

NOTE: frequency files can be found here: [drive_folder](https://drive.google.com/drive/folders/1MK2Ff3LqXuTwIQe9ukXmhIQDWUcFIoO_?usp=sharing) 

- **TF- update**

   Create the prototype of the object considering the most associated fillers of the subject AND the verb.
     
   Procedure:
   1. we retrieve the *N* most strongly associated objects for the subject and the verb respectively, and we take the intersection of the two lists;
   2. we update their association scores using either the sum (*add*) or the product (*prod*) function;
   3. we select the embeddings corresponding to the first *M* objects in this list and we average them together (centroid) to create the prototype vector of the object given the subject and the verb;
   4. the thematic fit of the object x with respect to the other items in the sentence is computed as the similarity score of its corresponding lexical vector v(x) with the prototype vector. 

   To avoid zero scores, we apply the following methodology in case the intersection of fillers is empty:
   + in the two lists are not empty, we use verb's fillers to create the prototype;
   + if one list is empty, we take the other one.

- **ANNs**
   + [ANNs_predict-token-masked.ipynb](https://github.com/giuliarambelli/Event_Knowledge_Model_Comparison/blob/master/ANNs_predict-token-masked.ipynb)
   
     This notebook contains the code to compute the probability of a masked word in a sentence, using a bidirectional ANNs (BERT, RoBERTa, XLNET).
     
     It performs 2 tasks: 1) Verb prediction  and 2) Last word prediction.
   + [ANNs_sequential-word-prediction.ipynb](https://github.com/giuliarambelli/Event_Knowledge_Model_Comparison/blob/master/ANNs_sequential-word-prediction.ipynb)
   
      This notebook contains the code to compute the probability of a sentence from the probability of single words using a bidirectional ANNs.
      
      It performs 2 tasks: 1) **Sequential word prediction**, or Pseudo-log likelihood (start from unmasked sentence, mask each word iteratively), and 2) **Left-to-right generation**, or Sequential Sampling (start from completely masked sentence and unmask words left-to-right)
    
   + [ANNs-unidirectional-predict-sentence.ipynb](https://github.com/giuliarambelli/Event_Knowledge_Model_Comparison/blob/master/ANNs-unidirectional-predict-sentence.ipynb)
   
      Use GPT-2 to compute the probability of a given sentence. The score of a sentence is the product of each word probability.


