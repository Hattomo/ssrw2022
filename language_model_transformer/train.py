import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import glob
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from sklearn.metrics import accuracy_score
from my_dataset import PhonemeDataset, collate_fn, dataset_spliter
from my_model import TransformerModel
import json

device = "cuda" if torch.cuda.is_available() else "cpu"


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

    def train(self, epochs: int=20, batch_size: int=16, hidden_dim:int=128):
        """
        Train method.

        Args:
            epochs (int, optional): _description_. Defaults to 20.
            batch_size (int, optional): _description_. Defaults to 16.
            hidden_dim (int, optional): _description_. Defaults to 128.
        """
        data_num  = len(dataloader_train.dataset)   # テストデータの総数
        pbar = tqdm(total=int(data_num/batch_size)) # プログレスバー設定
        self.train_loss_list = []
        self.train_acc_list = []
        self.valid_loss_list = []
        self.valid_acc_list = []
        for epoch in range(epochs):
            train_loss = 0.0
            train_acc = 0.0
            valid_loss = 0.0
            valid_acc = 0.0

            # 学習
            self.model.train()
            print("==== 学習フェーズ ====")
            for i, batch in tqdm(enumerate(self.dataloader_train)):
                # print(f'train: {i}/{len(self.dataloader_train)}')
                data, label = batch
                # ここから先書き換える
                data = data.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()
                src_mask = model.generate_square_subsequent_mask(data.size(1)).to(device)
                outputs = self.model(data, src_mask)
                label_max_len = label.shape[1]
                outputs = outputs[:,:label_max_len,:]
                outputs = outputs.reshape(outputs.shape[0]*outputs.shape[1], outputs.shape[2])
                label = label.reshape(label.size(0)*label.size(1))
                label = label.to(torch.long)
                loss = self.criterion(outputs, label)
                train_loss += loss.item()
                acc = accuracy_score(label.tolist(), outputs.argmax(dim=1).tolist())
                train_acc += acc
                # print(f'train loss: {loss}    acc: {acc}')
                loss.backward()
                self.optimizer.step()
                pbar.update(1)
            pbar.close()

            # 検証
            self.model.eval()
            print("==== 検証フェーズ ====")
            data_num  = len(dataloader_valid.dataset)   # テストデータの総数
            pbar = tqdm(total=int(data_num/batch_size)) # プログレスバー設定
            for i, batch in tqdm(enumerate(self.dataloader_valid)):
                # print(f'valid: {i}/{len(self.dataloader_valid)}')
                with torch.no_grad():
                    data, label = batch
                    data = data.to(self.device)
                    label = label.to(self.device)
                    self.optimizer.zero_grad()
                    src_mask = model.generate_square_subsequent_mask(data.size(1)).to(device)
                    outputs = self.model(data, src_mask)
                    label = label.reshape(label.size(0)*label.size(1))
                    label = label.to(torch.long)
                    loss = self.criterion(outputs, label)
                    valid_loss += loss.item()
                    acc = accuracy_score(label.tolist(), outputs.argmax(dim=1).tolist())
                    valid_acc += acc
                    pbar.update(1)
            pbar.close()

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


if __name__ == "__main__":
    batch_size = 100
    epoch = 20

    dict_path = "../data/phoneme_dict.json"
    with open(dict_path, 'r') as f:
        phone_dict = json.load(f)

    additional_file = '../data/phoneme_data.txt'
    with open(additional_file, 'r') as f:
        sentenses = f.readlines()
    datas = [s.replace('\n', '').split() for s in sentenses]

    # dataset
    datas_train, datas_valid, datas_test = dataset_spliter(datas)
    dataset_train = PhonemeDataset(phone_dict, datas_train)
    dataset_valid = PhonemeDataset(phone_dict, datas_valid)
    dataset_test  = PhonemeDataset(phone_dict, datas_test)

    # dataloader
    dataloader_train = DataLoader(dataset_train,
                                shuffle=True,
                                batch_size=batch_size,
                                collate_fn=collate_fn,
                                drop_last=True)
    dataloader_valid = DataLoader(dataset_valid,
                                batch_size=batch_size,
                                collate_fn=collate_fn,
                                drop_last=True)
    dataloader_test  = DataLoader(dataset_test,
                                batch_size=batch_size,
                                collate_fn=collate_fn,
                                drop_last=True)

    ntokens = 46 # the size of vocabulary
    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train
    trainer = Trainer(model, dataloader_train, dataloader_valid, criterion, optimizer)
    trainer.train(epoch, batch_size)

    # save
    save_path = './transformer_lang_model.pth'
    torch.save(model.to('cpu').state_dict(), save_path)
