#data

import json
from transformers import AutoTokenizer




def load_train_data(concise_labels=False):
   
    with open("../input/train.json") as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small", use_fast=False)
    num_of_docs = len(data)
    
    list_of_docs_tokenized = []
    list_of_token_labels = []

    for index in range(num_of_docs):
        tokenized = tokenizer(data[index]["tokens"], add_special_tokens=False)
        tokens_list = tokenized["input_ids"]
        tokens_of_a_doc = []
        token_labels = []
        for idx, tokens in enumerate(tokens_list):
            length = len(tokens)
            if length > 0:
                tokens_of_a_doc += tokens
                token_labels += [data[index]["labels"][idx]] * length
        list_of_docs_tokenized.append(tokens_of_a_doc)
        list_of_token_labels.append(token_labels)



    max_length = 510
    docs = []
    labels = []
    for idx, doc in enumerate(list_of_docs_tokenized):
        length = len(doc)
        partition_num = length // max_length
        partitions = []
        label_partitions = []
        for i in range(partition_num):
            partitions.append(doc[i*510:(i+1)*510])
            label_partitions.append(list_of_token_labels[idx][i*510:(i+1)*510])
        if partition_num*510 < length:
            partitions.append(doc[partition_num*510:])
            label_partitions.append(list_of_token_labels[idx][partition_num*510:])
        docs += partitions
        labels += label_partitions

    for idx, doc in enumerate(docs):
        docs[idx] = [1] + docs[idx] + [2]

    label_dict = {"O": 0, "B-NAME_STUDENT": 1, "I-NAME_STUDENT": 2, "B-ID_NUM": 3, "I-ID_NUM": 4,
             "B-PHONE_NUM": 5, "I-PHONE_NUM": 6, "B-STREET_ADDRESS": 7, "I-STREET_ADDRESS": 8, "B-EMAIL": 9,
             "I-EMAIL": 10, "B-USERNAME": 11, "I-USERNAME": 12, "B-URL_PERSONAL": 13, "I-URL_PERSONAL": 14}

    if concise_labels:
        label_dict = {"O": 0, "B-NAME_STUDENT": 1, "I-NAME_STUDENT": 1, "B-ID_NUM": 2, "I-ID_NUM": 2,
             "B-PHONE_NUM": 3, "I-PHONE_NUM": 3, "B-STREET_ADDRESS": 4, "I-STREET_ADDRESS": 4, "B-EMAIL": 5,
             "I-EMAIL": 5, "B-USERNAME": 6, "I-USERNAME": 6, "B-URL_PERSONAL": 7, "I-URL_PERSONAL": 7}
        

    for idx, label in enumerate(labels):
        labels[idx] = list(map(lambda x: label_dict[x], label))
    
    return docs, labels

    

def load_test_data(concise_labels=False):

    with open("../input/test.json") as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small", use_fast=False)
    num_of_docs = len(data)
    
    list_of_docs_tokenized = []
    list_of_token_labels = []
    list_of_token_ids = []
    list_of_attention_masks = []

    for index in range(num_of_docs):
        tokenized = tokenizer(data[index]["tokens"], add_special_tokens=False)
        tokens_list = tokenized["input_ids"]
        tokens_of_a_doc = []
        ids_of_tokens = []
        token_labels = []
        attention_mask = []
        for idx, tokens in enumerate(tokens_list):
            length = len(tokens)
            if length > 0:
                tokens_of_a_doc += tokens
                ids_of_tokens += [idx] * length
                token_labels += [data[index]["labels"][idx]] * length
                attention_mask += tokenized["attention_mask"][idx]
        list_of_docs_tokenized.append(tokens_of_a_doc)
        list_of_token_ids.append(ids_of_tokens)
        list_of_token_labels.append(token_labels)
        list_of_attention_masks.append(attention_mask)


    list_of_doc_ids = []
    for doc in data:
        list_of_doc_ids.append(doc["document"])



    max_length = 510
    docs = []
    labels = []
    token_ids = []
    for idx, doc in enumerate(list_of_docs_tokenized):
        length = len(doc)
        partition_num = length // max_length
        partitions = []
        label_partitions = []
        token_id_partitions = []
        for i in range(partition_num):
            partitions.append(doc[i*510:(i+1)*510])
            label_partitions.append(list_of_token_labels[idx][i*510:(i+1)*510])
            token_id_partitions.append(list_of_token_ids[idx][i*510:(i+1)*510])
        if partition_num*510 < length:
            partitions.append(doc[partition_num*510:])
            label_partitions.append(list_of_token_labels[idx][partition_num*510:])
            token_id_partitions.append(list_of_token_ids[idx][partition_num*510:])
        docs += partitions
        labels += label_partitions
        token_ids += token_id_partitions

    for idx, doc in enumerate(docs):
        docs[idx] = [1] + docs[idx] + [2]

    label_dict = {"O": 0, "B-NAME_STUDENT": 1, "I-NAME_STUDENT": 2, "B-ID_NUM": 3, "I-ID_NUM": 4,
             "B-PHONE_NUM": 5, "I-PHONE_NUM": 6, "B-STREET_ADDRESS": 7, "I-STREET_ADDRESS": 8, "B-EMAIL": 9,
             "I-EMAIL": 10, "B-USERNAME": 11, "I-USERNAME": 12, "B-URL_PERSONAL": 13, "I-URL_PERSONAL": 14}

    if concise_labels:
        label_dict = {"O": 0, "B-NAME_STUDENT": 1, "I-NAME_STUDENT": 1, "B-ID_NUM": 2, "I-ID_NUM": 2,
             "B-PHONE_NUM": 3, "I-PHONE_NUM": 3, "B-STREET_ADDRESS": 4, "I-STREET_ADDRESS": 4, "B-EMAIL": 5,
             "I-EMAIL": 5, "B-USERNAME": 6, "I-USERNAME": 6, "B-URL_PERSONAL": 7, "I-URL_PERSONAL": 7}
        

    for idx, label in enumerate(labels):
        labels[idx] = list(map(lambda x: label_dict[x], label))
    
    
    
    return docs, labels
    
