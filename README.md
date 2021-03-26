# Aspect Based Sentiment Analysis

## Team

- German Kolosov (german.kolosov@student.ecp.fr)
- Mounia El Baz (mounia.el-baz@student.ecp.fr)


## Approach

### Feature Engineering
In order to tackle the ASBA task, we used different methods to engineer features before classifying our data : 
- One hot encoding of aspects : each aspect was assigned a one hot tensor of size 12, indicating which aspect it was out of the 12 given in our dataset. 
The considered aspects are the following : ['AMBIENCE#GENERAL', 'DRINKS#PRICES', 'DRINKS#QUALITY',
           'DRINKS#STYLE_OPTIONS', 'FOOD#PRICES', 'FOOD#QUALITY',
           'FOOD#STYLE_OPTIONS', 'LOCATION#GENERAL', 'RESTAURANT#GENERAL',
           'RESTAURANT#MISCELLANEOUS', 'RESTAURANT#PRICES', 'SERVICE#GENERAL']
           
- Bert embeddings for the sentence and the target word : 
    - To incorporate the target word, we truncated the sentence around it, leaving only the closest words to the target word, which we felt almost always have a lot of meaning concerning the polarity. This defines a hyperparameter n which represents the number of words we consider at each side of the target. For example, if our sentence is : _Other guests enjoyed pizza, santa fe chopped salad and fish and chips._ and the target word is _pizza_ and n=2, we kept the following phrase : _guests enjoyed pizza, santa fe_ 
    - The mini sentence was then concatenated to the whole sentence (through a ‘.’ Separator)
    - After pre-processing the sentence for Bert, we have the following : x = ([CLS] target_neighbourhood [SEP] sentence [SEP])
    - Since Bert supports pairs of sentences as input, it is fed the pre-processed pair of sentences. h=BERT(x)
    
### Architecture
Once we have our features which are : 
- Bert embeddings h[CLS] which is a target aware representation of the sentence, of size 768 : this representation improves as we finetune BERT
- One hot vector encoding the aspect, of size 12 : this is a dummy representation which is hard-coded.

These embeddings are then concatenated to produce a vector of size 780 and fed to a two-layer classifier, with a sigmoid at the end. 

### Training 
There are two steps which need to be performed during the training of our network : 
- Finetuning Bert : this step is essential in order to adapt BERT to the task and have quality embeddings. It is done by unfreezing the weights of BERT (transformers library). We use a very small learning rate for this step.
- Training the classifier : the learning rate is bigger for this step. We use a weighted Cross Entropy Loss in order to circumvent the unbalanced classes. This is done to penalize the most frequent classes and give some importance to less frequent ones. 

### Hyperparameter Tuning
- Parameters relative to features:
    - number of words around the target word : 3
    - maximum length for Bert pre-processing : 150

- Parameters relative to the training :
    - The weights used in the Cross Entropy Loss are retrieved using the sklearn.utils.class_weight module. They were then adjusted since we felt that they were too penalizing. 
    - Learning Rate to finetune BERT : 1e-5
    - Learning Rate for the classifier : 6e-4
    - Batch Size : 16 (this was the maximum batch size we could use without having a CUDA out of memory error)
    - Number of epochs : 3 (with early stopping)

### Results
This approach gave an average accuracy of 85.69 (0.64) on the devdata.csv dataset (with 5 runs). 

### Sources 
Spring 2021 CentraleSupélec NLP Course

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Jacob Devlin Ming-Wei Chang Kenton Lee Kristina Toutanova, 2019

BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis. Hu Xu, Bing Liu, Lei Shu and Philip S. Yu, 2019

Aspect Sentiment Classification with both Word-level and Clause-level Attention Networks. Jingjing Wang, Jie Li, Shoushan Li, Yangyang Kang, Min Zhang, Luo Si, Guodong Zhou, 2018

Fine Tuning Bert for sentiment Analysis : https://skimai.com/fine-tuning-bert-for-sentiment-analysis/

### To run code : 
First : `cd src` 

Then : `python tester.py`