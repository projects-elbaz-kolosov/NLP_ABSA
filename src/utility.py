from transformers import BertTokenizer
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import pandas as pd

MAX_LEN = 150

ASPECTS = ['AMBIENCE#GENERAL', 'DRINKS#PRICES', 'DRINKS#QUALITY',
           'DRINKS#STYLE_OPTIONS', 'FOOD#PRICES', 'FOOD#QUALITY',
           'FOOD#STYLE_OPTIONS', 'LOCATION#GENERAL', 'RESTAURANT#GENERAL',
           'RESTAURANT#MISCELLANEOUS', 'RESTAURANT#PRICES', 'SERVICE#GENERAL']

LABELS = {0: 'negative', 1: 'positive', 2: 'neutral'}


# This function truncates the sentence, leaving only n words around the target word
def neighbors(x, n):
    sentence, start, end = x['sentence'], x['start'], x['end']
    count = 0
    left_i = start
    right_i = len(sentence) - end

    for i in range(len(sentence) - end):
        if sentence[end + i] == ' ':
            count += 1
        if count == n:
            right_i = i
            break

    count = 0
    for i in range(start):
        if sentence[start - i] == ' ':
            count += 1
        if count == n:
            left_i = i
            break

    return sentence[start - left_i:end + right_i]


# This function assigns labels
def assign_label(x):
    for num, lab in LABELS.items():
        if lab == x:
            return num


def assign_reverse_label(x):
    return LABELS.get(x)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Create a function to tokenize a set of texts
def preprocessing_for_bert(data, max_len):

    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


# This is sued to estimate the max_len of the training+val test, the max len is then hardcoded to take into account possible variations in the test set
def compute_max_len(X_train, X_val):
    all_sentence = np.concatenate([X_train, X_val])
    encoded_sentence = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_sentence]
    max_len = max([len(sent) for sent in encoded_sentence])

    return max_len


def preprocess_data(data_path):
    data = pd.read_csv(data_path, sep='\t', header=None).rename(
        columns={0: 'label', 1: 'aspect', 2: 'target', 3: 'location', 4: 'sentence'})
    data[['start', 'end']] = data['location'].str.split(':', expand=True).astype(int)
    data['trn'] = data.apply(lambda x: neighbors(x, 3), axis=1)
    data['class_label'] = data['label'].apply(assign_label)
    #    data['new_aspect'] = data.aspect.str.lower().str.replace("#", ", ").str.replace("_", " ")
    data['train'] = data['trn'] + '. ' + data['sentence']
    aspect_feat = data.pivot(columns='aspect', values='label').fillna(0)
    aspect_feat[aspect_feat != 0] = 1
    aspect_feat = aspect_feat.T.reindex(ASPECTS).T.fillna(0)
    return data.train.values, aspect_feat, data.class_label.values


def construct_dataloaders(inputs, masks, aspect_feat, y, batch_size=16, test=False):
    labels = torch.tensor(y)
    data = TensorDataset(torch.tensor(aspect_feat.values.astype(int)), inputs, masks, labels)

    if test:
        sampler = SequentialSampler(data)
    else:
        sampler = RandomSampler(data)

    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader


def final_data_preprocessing(path, batch_size=16, test=False):
    X_train, aspect_feat_train, y_train = preprocess_data(path)
    max_len = MAX_LEN
    train_inputs, train_masks = preprocessing_for_bert(X_train, max_len)
    train_dataloader = construct_dataloaders(train_inputs, train_masks, aspect_feat_train, y_train,
                                             batch_size=batch_size, test=test)
    return train_dataloader
