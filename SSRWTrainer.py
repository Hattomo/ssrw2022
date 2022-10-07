import torch
import torch.nn as nn
import numpy as np
# from tqdm.notebook import tqdm
from tqdm import tqdm_notebook as tqdm
from torch.utils.data import Dataset, DataLoader
import pickle
import csv
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
from language_model import PhonemeLangModel
from phoneme_dataset import PhonemeDataset, collate_fn, dataset_spliter

from my_utils.my_util import calculate_error
from my_utils import my_util
from phoneme_dataset import PAD_NUM, MASK_NUM

class SSRWTrainer:
    """
    This class train the model with train().
    """
    def __init__(self,
                 model: nn.Module,
                 dataloader_train: DataLoader,
                 dataloader_valid: DataLoader,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device=torch.device('cuda:0'),):
        """Constractor

        Args:
            model (nn.Module): model for train.
            dataloader_train (DataLoader): train dataloader
            dataloader_valid (DataLoader): validation dataloader
            criterion (nn.Module): criterion (ex. CrossEntropyLoss)
            optimizer (torch.optim.Optimizer): Optimizer for optimize model (ex. Adam) 
            device (torch.device, optional): Device for calculation. Defaults to torch.device('cuda:0').
        """
        self.device = device
        self.model = model.to(device=device)
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.criterion = criterion
        self.optimizer = optimizer
    
    def detach(self, states):
        return [state.detach() for state in states]
    
    def train(self,
              epochs: int=20,
              batch_size: int=4,
              hidden_dim: int=128):
        self.train_loss_list =[]
        self.train_acc_list = []
        self.valid_loss_list = []
        self.valid_acc_list = []
        for epoch in tqdm(range(epochs), total=epochs):
            train_loss = 0.0
            train_acc = 0.0
            valid_loss = 0.0
            valid_acc = 0.0

            # 初期隠れ状態とせる状態を設定する
            states = (torch.zeros(1, batch_size, hidden_dim).to(self.device),
                      torch.zeros(1, batch_size, hidden_dim).to(self.device))
             # 学習
            self.model.train()
            print(f'train: {epoch}')
            for i, batch in tqdm(enumerate(self.dataloader_train), total=len(self.dataloader_train)):
                # print(f'train: {i}/{len(self.dataloader_train)}')
                data, label = batch
                data = data.to(self.device)
                label = label.to(self.device)
                self.optimizer.zero_grad()
                states = self.detach(states)
                outputs, states = self.model(data, states)
                label = label.to(torch.long)
                outputs = outputs.reshape(outputs.shape[0]*outputs.shape[1], outputs.shape[2])
                label = label.reshape(label.shape[0]*label.shape[1])
                loss = self.criterion(outputs, label)
                train_loss += loss.item()
                acc = accuracy_score(label.tolist(), outputs.argmax(dim=1).tolist())
                train_acc += acc
                # print(f'loss: {loss}    acc: {acc}')
                loss.backward()
                self.optimizer.step()
