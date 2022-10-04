import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import csv
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
from language_model import PhonemeLangModel
from phoneme_dataset import PhonemeDataset, collate_fn, dataset_spliter

from torchvision import transforms
from trainer_phoneme import Trainer
import json

if __name__ == '__main__':
    BATCH_SIZE = 64
    EPOCHS = 50
    dict_path = 'data/label.pkl'
    labels_path = 'data/phoneme.csv'
    with open(dict_path, 'rb') as f:
        phone_dict = pickle.load(f)
    with open(labels_path, 'r') as f:
        csv_reader = csv.reader(f)
        datas = list(csv_reader)
    
    """    
    with open('build/debug/token.json') as f:
        phone_dict = json.load(f)
    """

    datas_train, datas_valid, datas_test = dataset_spliter(datas)
    dataset_train = PhonemeDataset(phone_dict, datas_train)
    dataset_valid = PhonemeDataset(phone_dict, datas_valid)
    dataset_test = PhonemeDataset(phone_dict, datas_test)

    dataloader_train = DataLoader(dataset_train,
                                  shuffle=True,
                                  batch_size=BATCH_SIZE,
                                  collate_fn=collate_fn,
                                  drop_last=True)
    dataloader_valid = DataLoader(dataset_valid,
                                  batch_size=BATCH_SIZE,
                                  collate_fn=collate_fn,
                                  drop_last=True)
    dataloader_test  = DataLoader(dataset_test,
                                  batch_size=BATCH_SIZE,
                                  collate_fn=collate_fn,
                                  drop_last=True)
    model = PhonemeLangModel().to('cuda:0')

    criterion = nn.CTCLoss(blank=phone_dict['sil'], zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, dataloader_train, dataloader_valid, criterion, optimizer)
    trainer.train(EPOCHS, BATCH_SIZE)