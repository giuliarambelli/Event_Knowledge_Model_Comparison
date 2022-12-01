# Code to compute model scores for "Event knowledge in large language models: the gap between the impossible and the unlikely"

By Carina Kauf*, Anna A. Ivanova*, Giulia Rambelli, Emmanuele Chersoni, Jingyuan S. She, Zawad Chowdhury, Evelina Fedorenko, Alessandro Lenci

(the two lead authors contributed equally to this work)

Paper repository at: [https://github.com/carina-kauf/lm-event-knowledge](https://github.com/carina-kauf/lm-event-knowledge)

## MODELS

## I. Large language models (LLMs)
We tested four attention-based Transformer (Vaswani et al., 2017) language models:
1. RoBERTa (Liu et al., 2019)
2. BERT (Devlin et al., 2018)
3. GPT-J (B. Wang & Komatsuzaki, 2021)
4. GPT-2 (Radford et al., 2019)

### I.I Score calculation for RoBERTa & BERT

**Main metric: Adapted Pseudo-log-likelihood (PLL)**<p>
We use a modified version of the sentence’s pseudo-log-likelihood under the model (PLL; Salazar et al., 2020; A. Wang & Cho, 2019), which defines the sentence score as the sum of the log-probabilities of each token given all other tokens. To avoid biasing the scores in favor of multi-token lexical items, we modify the original procedure to additionally mask tokens within multi-token words if they are located to the right of the target.
* associated script at: [ANN_MLM_adapted.py](https://github.com/giuliarambelli/Event_Knowledge_Model_Comparison/blob/master/ANN_MLM_adapted.py)
   
**Secondary metrics**<p>
   1. **PLL** (Salazar et al., 2020)
   2. **Verb probability**, i.e., the average log-likelihood of the verb’s tokens v = w<sub>t</sub> ...w<sub>t'</sub>  conditioned on their bidirectional sentence context
   3. **Last-word probability**, i.e., the average log-likelihood of the subtokens that compose the last word in the sequence according to the model’s tokenizer
   4. **Left-to-right (l2r)**, causal sentence-generation probability,  i.e., average log-likelihood for each token w<sub>i</sub> in the sequence, conditioned on only the preceding tokens w<sub><i</sub> according to the model.
* associated script at: [ANN_MLM_scores.py](https://github.com/giuliarambelli/Event_Knowledge_Model_Comparison/blob/master/ANN_MLM_scores.py)

### I.II Score calculation for GPT-J & GPT-2

We define the sentence score as the sum of the log-probabilities of each token w<sub>i</sub> in the sequence, conditioned on the preceding sentence tokens w<sub><i</sub>}.
* associated script at: [ANN_GPT2_scores.py](https://github.com/giuliarambelli/Event_Knowledge_Model_Comparison/blob/master/ANN_GPT2_scores.py)

## II. Baseline models
1. **tinyLSTM** (Gauthier et al., 2020): computes the surprisal of a sentence as the sum of the surprisals of each token in the sentence
*  associated script at: [lmzoo_tinylstm.py](https://github.com/giuliarambelli/Event_Knowledge_Model_Comparison/blob/master/lmzoo_tinylstm.py)

2. **thematic Fit**
   Create the prototype of the object considering the most associated fillers of the subject AND the verb.
     
   Procedure:
   1. we retrieve the *N* most strongly associated objects for the subject and the verb respectively, and we take the intersection of the two lists;
   2. we update their association scores using either the product (*prod*) function;
   3. we select the embeddings corresponding to the first *M* objects in this list and we average them together (centroid) to create the prototype vector of the object given the subject and the verb;
   4. the thematic fit of the object x with respect to the other items in the sentence is computed as the similarity score of its corresponding lexical vector v(x) with the prototype vector. 

   To avoid zero scores, we apply the following methodology in case the intersection of fillers is empty:
   + in the two lists are not empty, we use verb's fillers to create the prototype;
   + if one list is empty, we take the other one.
   
3. **Structured Distributional Model (SDM; Chersoni et al., 2019)**
   

4. **PPMI-syntax** (structured input, input annotated with grammatical roles)    
   
   After extracting triples < verbal head, nominal dependent, relation > from the corpora (with a frequency >= 2), we compute the PPMI as follows 
   (N= total frequency of all triples).
   ![ppmi baseline 1](https://github.com/giuliarambelli/Event_Knowledge_Model_Comparison/blob/master/img/baseline1.gif)
      
* associated script at: [baselines_PPMI_structured_and_unstructured.ipynb](https://github.com/giuliarambelli/Event_Knowledge_Model_Comparison/blob/master/baselines_PPMI_structured_and_unstructured.ipynb) 
* NOTE: frequency files can be found here: [drive_folder](https://drive.google.com/drive/folders/1MK2Ff3LqXuTwIQe9ukXmhIQDWUcFIoO_?usp=sharing)

## DATASETS (name aliases)

**Dataset 1 - EventsAdapt (based on Fedorenko et al, 2020)** : newsentences_EventsAdapt

**Dataset 2 - DTFit (based on Vassallo et al, 2018)** : DTFit

**Dataset 3 - EventsRev (based on Ivanova et al, 2021)** : ev1
