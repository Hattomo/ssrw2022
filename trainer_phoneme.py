import torch
import torch.nn as nn
import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
import glob
import pickle
import csv
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
from language_model import PhonemeLangModel
from phoneme_dataset import PhonemeDataset, collate_fn, dataset_spliter

from torchvision import transforms

class Trainer:
    """
    This class train the model with train().s
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
    
    def train(self, epochs: int=20, batch_size: int=16, hidden_dim:int=128):
        """
        Train method.

        Args:
            epochs (int, optional): _description_. Defaults to 20.
            batch_size (int, optional): _description_. Defaults to 16.
            hidden_dim (int, optional): _description_. Defaults to 128.
        """
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
            print('train')
            for i, batch in enumerate(self.dataloader_train):
                print(f'train: {i}/{len(self.dataloader_train)}')
                data = batch
                # ここから先書き換える
                images = images.to(device=self.device)
                label = label.to(device=self.device)

                self.optimizer.zero_grad()
                states = self.detach(states)
                outputs, states = self.model(images, states)
                label_max_len = label.shape[1]
                outputs = outputs[:,:label_max_len,:]
                outputs = outputs.reshape(outputs.shape[0]*outputs.shape[1], outputs.shape[2])
                label = label.reshape(label.size(0)*label.size(1))
                label = label.to(torch.long)
                loss = self.criterion(outputs, label)
                train_loss += loss.item()
                acc = accuracy_score(label.tolist(), outputs.argmax(dim=1).tolist())
                train_acc += acc
                print(f'loss: {loss}    acc: {acc}')
                loss.backward()
                self.optimizer.step()
            
            # 検証
            states = (torch.zeros(1, batch_size, hidden_dim).to(self.device),
                    torch.zeros(1, batch_size, hidden_dim).to(self.device))
            self.model.eval()
            print('validation')
            for i, batch in enumerate(self.dataloader_valid):
                print(f'valid: {i}/{len(self.dataloader_valid)}')
                with torch.no_grad():
                    images, label = batch
                    images = images.to(self.device)
                    label = label.to(self.device)
                    self.optimizer.zero_grad()
                    states = self.detach(states)
                    outputs, states = self.model(images, states)
                    label = label.reshape(label.size(0)*label.size(1))
                    loss = self.criterion(outputs, label)
                    valid_loss += loss.item()
                    acc = accuracy_score(label.tolist(), outputs.argmax(dim=1).tolist())
                    valid_acc += acc
            
            epoch_loss_train = train_loss / len(self.dataloader_train)
            epoch_acc_train = train_acc / len(self.dataloader_train)
            epoch_loss_valid = valid_loss / len(self.dataloader_valid)
            epoch_acc_valid = valid_acc / len(self.dataloader_valid)
            self.train_loss_list.append(epoch_loss_train)
            self.train_acc_list.append(epoch_acc_train)
            self.valid_loss_list.append(epoch_loss_valid)
            self.valid_acc_list.append(epoch_acc_valid)
            print(f'train: Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss_train:.4f} Acc: {epoch_acc_train:.4f}')
            print(f'valid: Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss_valid:.4f} Acc: {epoch_acc_valid:.4f}')



    
if __name__ == '__main__':
    dataset = PhonemeDataset()
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True,
        collate_fn=collate_fn
    )
    for batch in dataloader:
        print(batch[0].shape)
