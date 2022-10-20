# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import argparse
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self,
                 phones: list[str],
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int = 46,
                 opts: argparse.Namespace = None):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=phones.index("_"))
        self.lstm = nn.LSTM(1, hidden_dim, num_layers=opts.lstm_layer, batch_first=True, bidirectional=True)
        self.opts = opts

    def forward(self, x):
        x = x.unsqueeze(2).float()
        h0 = torch.randn(self.opts.lstm_layer * 2, x.shape[0], self.hidden_dim).to(self.opts.device)
        c0 = torch.randn(self.opts.lstm_layer * 2, x.shape[0], self.hidden_dim).to(self.opts.device)
        _, state = self.lstm(x, (h0, c0))
        return state

class Decoder(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 phones: list[str],
                 opts: argparse.Namespace = None):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(1, hidden_dim, num_layers=opts.lstm_layer, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, vocab_size)
        self.phones = phones
        self.opts = opts

    def forward(self, state, decoder_output_length, batch_size):
        input_phone = torch.full((batch_size, 1, 1), self.phones.index("mask")).float().to(self.opts.device)
        input_phone = rnn.pad_sequence(input_phone, batch_first=True)
        x = []
        flag = [False] * batch_size
        while len(x) < decoder_output_length:
            out_vec, state = self.lstm(input_phone, state)
            out_vec = self.linear(out_vec)
            input_phone = torch.argmax(out_vec, dim=2)
            for i in range(batch_size):
                if input_phone[i] == self.phones.index("_"):
                    flag[i] = True
                if flag[i]:
                    out_vec[i] *= F.one_hot(torch.tensor([3]), num_classes=len(self.phones)).to(self.opts.device)
            x.append(out_vec)
            input_phone = torch.unsqueeze(input_phone, 2).float()
            input_phone = rnn.pad_sequence(input_phone, batch_first=True)

        x = torch.cat(x, dim=1)
        return x

class PhonemeLangModelv2(nn.Module):

    def __init__(self,
                 phones: list[str],
                 embed_size: int = 32,
                 hidden_dim_encoder: int = 128,
                 hidden_dim_decoder: int = 128,
                 vocab_size: int = 46,
                 max_len: int = 500,
                 img_size: int = 64,
                 opts: argparse.Namespace = None):
        super(PhonemeLangModelv2, self).__init__()
        self.embed_size = embed_size
        self.hidden_dim_encoder = hidden_dim_encoder
        self.hidden_dim_decoder = hidden_dim_decoder
        self.max_len = max_len
        self.img_size = img_size
        self.vocab_size = vocab_size
        self.lstm_encoder = Encoder(phones, self.embed_size, self.hidden_dim_encoder, opts=opts)
        self.lstm_decoder = Decoder(self.vocab_size,
                                    self.embed_size,
                                    self.hidden_dim_decoder,
                                    phones,
                                    opts=opts)

    def forward(self, x, decoder_output_length):
        h = self.lstm_encoder(x)
        x = self.lstm_decoder(h, decoder_output_length, x.size(0))
        return x
