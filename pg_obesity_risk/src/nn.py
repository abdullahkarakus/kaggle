#NeuralNetwork

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import argparse

import data


class DataSet(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __getitem__(self, index):
        x = torch.tensor(self.data[index], dtype=torch.float)
        y = torch.tensor(self.target[index], dtype=torch.long)

        return x,y

    def __len__(self):
        return len(self.target)

class NNetwork(nn.Module):
    def __init__(self, input_dim,hidden_dim):
        super().__init__()

        self.layers = nn.Sequential(
                                nn.Linear(input_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim,hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim,7),
                                nn.Softmax(dim=1)
        )

    def forward(self,x):
        x = self.layers(x)

        return x

def train(model, optimizer, data_loader):
    model.train()

    for data_point in data_loader:
        data, target = data_point
        optimizer.zero_grad()
        
        probs = model(data)
        loss = nn.CrossEntropyLoss()
        l = loss(probs, target)
        l.backward()
        optimizer.step()

def evaluate(model, data_loader):
    model.eval()

    predictions = []
    correct_classes = []
    with torch.no_grad():
        for data_point in data_loader:
            data, target = data_point
    
            probs = model(data)
            preds = torch.argmax(probs, dim=1)
            predictions.extend(preds.tolist())
            correct_classes.extend(target.tolist())

    return predictions, correct_classes



def compute_loss(model, data_loader, data_set):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data_point in data_loader:
            data, target = data_point
        
            probs = model(data)
            loss = nn.CrossEntropyLoss()
            l = loss(probs, target)
            total_loss += l

    return total_loss / len(data_set)
            
        

def run(num_epochs, hidden_dim):
    df_1 = data.load_train_data()
    df_2 = data.load_extra_data()

    df = pd.concat((df_1,df_2), axis=0)

    target = df.NObeyesdad.values
    df = df.drop(columns = "NObeyesdad")

    enc_lbl = LabelEncoder()
    target = enc_lbl.fit_transform(target)

    input_dim = df.shape[1]

    train_set, test_set, target_train, target_test = train_test_split(df, target, test_size=0.2, random_state=39, stratify=target)

    scaler_training = StandardScaler()
    scaler_test = StandardScaler()
    train_set = scaler_training.fit_transform(train_set)
    test_set = scaler_test.fit_transform(test_set)
    

    train_set = DataSet(train_set, target_train)
    test_set = DataSet(test_set, target_test)

    train_data_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    test_data_loader = DataLoader(test_set, batch_size=1)

    model = NNetwork(input_dim, hidden_dim)
    optimizer = torch.optim.NAdam(model.parameters(), lr=1e-4)

    train_accuracy = []
    test_accuracy = []
    train_loss = []
    test_loss = []
    
    for epoch in range(num_epochs):
        print(f"Training epoch {epoch}...")
        train(model, optimizer, train_data_loader)
        
        train_preds, train_correct_class = evaluate(model, train_data_loader)
        test_preds, test_correct_class = evaluate(model, test_data_loader)

        accuracy_train = accuracy_score(train_correct_class, train_preds)
        accuracy_test = accuracy_score(test_correct_class, test_preds)
        train_accuracy.append(accuracy_train)
        test_accuracy.append(accuracy_test)

        loss_train = compute_loss(model, train_data_loader, train_set)
        loss_test = compute_loss(model, test_data_loader, test_set)
        train_loss.append(loss_train)
        test_loss.append(loss_test)
        

    
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    ax1.plot(list(range(1,num_epochs+1)), train_accuracy, label="train accuracy")
    ax1.plot(list(range(1,num_epochs+1)), test_accuracy, label="test accuracy")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("accuracy")
    ax1.legend()

    ax2.plot(list(range(1,num_epochs+1)), train_loss, label="train loss")
    ax2.plot(list(range(1,num_epochs+1)), test_loss, label="test loss")
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("loss")
    ax2.legend()
    
    plt.show()
    

def get_predictions(train_data, train_target, num_epochs, hidden_dim):
    
    enc_lbl = LabelEncoder()
    target = enc_lbl.fit_transform(train_target)

    input_dim = train_data.shape[1]
    
    df_test = data.load_test_data()
    id = df_test.id
    df_test.drop(columns="id", inplace=True)

    scaler_training = StandardScaler()
    scaler_test = StandardScaler()
    train_set = scaler_training.fit_transform(train_data)
    test_set = scaler_test.fit_transform(df_test)
    
    train_set = DataSet(train_set, target)
    test_set = DataSet(test_set, id)

    train_data_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    test_data_loader = DataLoader(test_set, batch_size=1)

    model = NNetwork(input_dim, hidden_dim)
    optimizer = torch.optim.NAdam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        print(f"Training epoch {epoch}...")
        train(model, optimizer, train_data_loader)

    preds, ids = evaluate(model, test_data_loader)
        
    index = ids
    preds = enc_lbl.inverse_transform(preds) 
    
    return preds
        
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--hidden_dim", type=int)

    args = parser.parse_args()

    torch.manual_seed(37)
    run(args.num_epochs, args.hidden_dim)
    
            
           
            

    
    

    

    



















