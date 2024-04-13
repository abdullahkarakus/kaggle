#run.py

import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import argparse

import data
import ner

def run(num_epochs):

    print("Loading Data...")
    docs, labels = data.load_train_data(concise_labels=True)

    docs_train, docs_test, labels_train, labels_test = train_test_split(
                                                            docs, labels, test_size=0.2, random_state=43, shuffle=True) 

    training_data = ner.DataSet(docs_train, labels_train)
    validation_data = ner.DataSet(docs_test, labels_test)

    data_loader_train = DataLoader(training_data, batch_size=32, collate_fn=ner.collator)
    data_loader_valid = DataLoader(validation_data, batch_size=32, collate_fn=ner.collator)

    mps_device = torch.device("mps")

    model = ner.NER()
    model = model.to(mps_device)

    model.transformer.requires_grad_(False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    
    for epoch in range(num_epochs):
        print(f"Training for epoch {epoch}")
        ner.train(model, optimizer, data_loader_train, mps_device)
        print("evaluating")
        predictions, correct_classes = ner.evaluate(model, data_loader_valid, mps_device)
        f_score = ner.f5_micro(correct_classes, predictions)

        print(f"f_score {f_score}")

def submit():
    pass



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_epochs", type=int)

    args = parser.parse_args()

    run(args.num_epochs)