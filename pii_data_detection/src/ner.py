#ner

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

from transformers import AutoModel

from tqdm import tqdm

import argparse

class DataSet(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __getitem__(self, index):
        token = torch.tensor(self.tokens[index], dtype=torch.long)
        label = torch.tensor(self.labels[index], dtype=torch.long)

        return token, label

    def __len__(self):
        return len(self.labels)


class NER(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = AutoModel.from_pretrained("microsoft/deberta-v3-small")
        self.linear_1 = nn.Linear(in_features=768, out_features=768)
        self.activation = nn.ReLU()
        self.linear_2 = nn.Linear(in_features=768, out_features=7)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, attention_mask):
        x = self.transformer(x, return_dict=False, attention_mask=attention_mask)
        x = self.linear_1(x[0])
        x = self.activation(x)
        x = self.linear_2(x)
        x = self.softmax(x)

        return x  


def train(model, optimizer, data_loader, device):
    model.train()

    for data in tqdm(data_loader):
        docs, labels, attention_masks, lengths = data
        docs, labels, attention_masks, lengths = docs.to(device), labels.to(device), attention_masks.to(device), lengths.to(device)
        
        optimizer.zero_grad()
        
        probs = model(docs, attention_mask=attention_masks)
        probs = unpad_sequence(probs, lengths=lengths, batch_first=True)
        probs = [prob[1:prob.shape[0]-1,:] for prob in probs]
        probs = torch.cat(probs, axis=0)
        
        loss = nn.CrossEntropyLoss(reduction="mean")
        l = loss(probs,labels)
        l.backward()
        optimizer.step()
        


def evaluate(model, data_loader,device):
    model.eval()
    
    predictions = []
    correct_classes = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            docs, labels, attention_masks, lengths = data
            docs, labels, attention_masks, lengths = docs.to(device), labels.to(device), attention_masks.to(device), lengths.to(device)

            probs = model(docs, attention_mask=attention_masks)
            probs = unpad_sequence(probs, lengths=lengths, batch_first=True)
            probs = [prob[1:prob.shape[0]-1,:] for prob in probs]
            preds = [torch.argmax(prob, dim=1) for prob in probs]
            preds = torch.cat(preds, axis=0)
            predictions.extend(preds.tolist())
            correct_classes.extend(labels.tolist())

    return predictions, correct_classes


def collator(data):
    docs, labels = zip(*data)
    attention_masks = [torch.ones_like(doc) for doc in docs]
    lengths = torch.tensor([len(doc) for doc in docs])
    docs = pad_sequence(docs, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = torch.cat(labels, axis=0)

    return docs, labels, attention_masks, lengths



def f5_micro(predictions, correct_classes):
    
    predictions = torch.tensor(predictions)
    correct_classes = torch.tensor(correct_classes)

    false_positives = torch.sum(predictions != correct_classes) - torch.sum((predictions == 0) * (correct_classes != 0))
    false_negatives = torch.sum(predictions != correct_classes) - torch.sum((predictions != 0) * (correct_classes == 0))
    true_positives = torch.sum(predictions == correct_classes) - torch.sum((predictions == 0) * (correct_classes == 0))
    
    f_score = (26*true_positives) / (26*true_positives + false_positives + 25*false_negatives)

    print(false_positives, false_negatives, true_positives)
    return f_score.item()
    
    



    
        
        
        

    
        
                                                                        
                                                                        
    
    

    
    







