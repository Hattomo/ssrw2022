import torch
import torch.nn as nn

from phoneme_dataset import PAD_NUM

class PhonemeLangModel(nn.Module):
    def __init__(self,
                 embed_size: int=32,
                 hidden_dim_encoder: int=128,
                 hidden_dim_decoder: int=128,
                 vocab_size: int=46,
                 max_len: int=500,
                 img_size: int=64,
                 device=torch.device('cuda:0')):
        super(PhonemeLangModel, self).__init__()
        self.embed_size = embed_size
        self.hidden_dim_encoder = hidden_dim_encoder
        self.hidden_dim_decoder = hidden_dim_decoder
        self.max_len = max_len
        self.img_size = img_size 
        self.vocab_size = vocab_size
        self.device = device
        self.lstm_encoder = Encoder(self.embed_size, self.hidden_dim_encoder)
        self.lstm_decoder = Decoder(self.vocab_size, self.embed_size, self.hidden_dim_decoder)

    def forward(self, x, h):
        h = self.lstm_encoder(x, h)
        # x = x[:, :-1]
        x, h = self.lstm_decoder(x, h)
        return x, h


class Encoder(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int=54+4):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=PAD_NUM)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    
    def forward(self, x, state):
        x = self.embedding(x)
        _, state = self.lstm(x, state)
        return state

class Decoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=PAD_NUM)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, state):
        x = self.embedding(x)
        x, state = self.lstm(x, state)
        x = self.linear(x)
        return x, state

class PhonemeLangModelv2(nn.Module):
    def __init__(self,
                 embed_size: int=32,
                 hidden_dim_encoder: int=128,
                 hidden_dim_decoder: int=128,
                 vocab_size: int=46,
                 max_len: int=500,
                 img_size: int=64,
                 device=torch.device('cuda:0')):
        super(PhonemeLangModelv2, self).__init__()
        self.embed_size = embed_size
        self.hidden_dim_encoder = hidden_dim_encoder
        self.hidden_dim_decoder = hidden_dim_decoder
        self.max_len = max_len
        self.img_size = img_size 
        self.vocab_size = vocab_size
        self.device = device
        self.lstm_encoder = Encoder(self.embed_size, self.hidden_dim_encoder)
        self.lstm_decoder = Decoder(self.vocab_size, self.embed_size, self.hidden_dim_decoder)

    def forward(self, x, h):
        h = self.lstm_encoder(x, h)
        x = x[:, :-1]
        x, h = self.lstm_decoder(x, h)
        return x, h
