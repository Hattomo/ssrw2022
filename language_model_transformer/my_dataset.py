import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import csv
from torch.nn.utils.rnn import pad_sequence
import random
import pickle
from sklearn.model_selection import train_test_split

PAD_NUM = 3
MASK_NUM = 25

class PhonemeDataset(Dataset):
    def __init__(self,
                 phone_dict: dict,
                 datas: list,
                 mask_ratio: float=0.3):
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
        self.mask_ratio = mask_ratio
        self.tokens_list = []
        for label in self.labels:
            tokens = []
            for e in label:
                tokens.append(self.dict[e])
            self.tokens_list.append(tokens)

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, index):
        label = torch.Tensor(self.tokens_list[index]).to(dtype=torch.int)
        all_indexes = list(range(0, len(self.tokens_list[index])))
        mask_indexes = random.sample(all_indexes, k=int(len(self.tokens_list[index])*(self.mask_ratio)))
        data = [token if i in mask_indexes else MASK_NUM for i, token in enumerate(self.tokens_list[index])]

        return torch.Tensor(data).to(dtype=torch.int), label

def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    """
    バッチ内のテンソルのサイズを合わせるためにパディングを行います。
    DataLoaderをインスタンス化するときにcollate_fn=で渡しください。
    Args:
        batch (torch.Tensor): バッチです。
    Returns:
        torch.Tensor: シーケンス長が最大のものに合わせた形のバッチテンソルを返します。
    """
    datas, label = list(zip(*batch))
    datas = pad_sequence(datas, batch_first=True, padding_value=PAD_NUM)
    label = pad_sequence(label, batch_first=True, padding_value=PAD_NUM)
    return datas, label

def dataset_spliter(dataset: list, train: int=6, valid: int=2, test: int=2) -> list:
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
    test_ratio  = float(valid+test)/float(all_ratio)
    train_list, test_list = train_test_split(dataset, test_size=test_ratio)
    test_ratio = float(test)/float(valid+test)
    valid_list, test_list = train_test_split(test_list, test_size=test_ratio)
    print(f'train: {len(train_list)} valid: {len(valid_list)} test: {len(test_list)}')
    return [train_list, valid_list, test_list]
