import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import csv
from torch.nn.utils.rnn import pad_sequence
import random
import pickle

class PhonemeDataset(Dataset):
    def __init__(self, 
                 phone_dict: dict,
                 datas: list):
        super(PhonemeDataset, self).__init__()
        """
        phone_dict には音素をキーとして一意のIDをバリューとする辞書を渡してください。
        datas には音素データセットをlistで渡してください。形式は以下の通りです。
        sil,n,a,g,a,sh,i,g,i,r,i,g,a,k,a,N,z,e,N,n,i,h,a,i,r,e,b,a,pau,d,e,b,a,f,u,n,o,k,o,o,k,a,g,a,f,u,y,o,s,a,r,e,r,u,sil
        sil,g,e,gw,a,N,w,a,k,o,n,o,t,o,k,o,r,o,t,a,sh,a,o,m,i,k,u,d,a,s,U,sh,i,pau,ch,o,cl,t,o,o,d,o,k,a,s,U,k,a,sil
        sil,gw,e,r,u,ts,o,o,n,i,w,a,pau,s,a,k,e,n,a,r,a,w,o,cl,k,a,t,o,s,U,p,u,r,i,cl,ts,a,a,o,k,o,n,o,m,i,m,a,s,U,n,a,sil
        ...
        データセットトークンID列の本体はself.tokens_listで、torch.Tensor型です。
        """
        self.dict = phone_dict
        # 全発話ディレクトリ、音素アノテーションをロード
        self.labels = datas
        self.tokens_list = []
        for label in self.labels:
            tokens = []
            for e in label:
                tokens.append(self.dict[e])
            self.tokens_list.append(tokens)
        self.tokens_list = torch.Tensor(self.tokens_list)
    
    def __len__(self):
        return len(self.tokens_list)
    
    def __getitem__(self, index):
        return self.tokens_list[index]

def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    """
    バッチ内のテンソルのサイズを合わせるためにパディングを行います。
    DataLoaderをインスタンス化するときにcollate_fn=で渡しください。
    Args:
        batch (torch.Tensor): バッチです。

    Returns:
        torch.Tensor: シーケンス長が最大のものに合わせた形のバッチテンソルを返します。
    """
    label = pad_sequence(batch, batch_first=True)
    return label

def dataset_spliter(dataset: list, train: int=5, valid: int=3, test: int=2) -> list:
    """
    データセットをtrain, valid, testの3つに分けます。

    Args:
        dataset (list): [[音素列], [音素列],...,] のような音素列のリストを渡しください。
        train (int)   : trainデータの比率を整数で指定してください。
        valid (int)   : validデータの比率を整数で指定しください。
        test  (int)   : testデータの比率を整数で指定しください。

    Returns:
        list: [[train音素列のリスト], [valid音素列のリスト], [test音素列のリスト]]のような3要素のリストを返します。
    """
    all_ratio = train+valid+test
    dataset_size = len(dataset)
    train_ratio = float(train)/float(all_ratio)
    valid_ratio = float(valid)/float(all_ratio)
    test_ratio = float(test)/float(all_ratio)
    all_indexes = list(range(0, dataset_size))
    train_sample = random.sample(all_indexes,
                                     k=int(dataset_size*(train_ratio/all_ratio)))
    remain_indexes = [e for e in all_indexes if not (e in train_sample)]
    valid_sample = random.sample(remain_indexes,
                                     k=int(dataset_size*(valid_ratio/all_ratio)))
    test_sample = [e for e in remain_indexes if not (e in valid_sample)]

    train_list = [tokens for i, tokens in enumerate(dataset) if i in train_sample]
    valid_list = [tokens for i, tokens in enumerate(dataset) if i in valid_sample]
    test_list = [tokens for i, tokens in enumerate(dataset) if i in test_sample]

    return [train_list, valid_list, test_list]


if __name__ == "__main__":
    """
    How to use classes
    """

    dict_path = 'data/label.pkl'
    labels_path = 'data/phoneme.csv'
    with open(dict_path, 'r') as f:
        phone_dict = pickle.load(f)
    with open(labels_path, 'r') as f:
        csv_reader = csv.reader(f)
        datas = list(csv_reader)
    dataset = PhonemeDataset(phone_dict, datas)
    dataset_train, _, _ = dataset_spliter(dataset)
    dataloader_train = DataLoader()