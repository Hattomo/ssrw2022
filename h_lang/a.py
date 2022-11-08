# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

# base = torch.arange(0, 4)
# a = base.float()
# loss = nn.CrossEntropyLoss()
# b = torch.tensor([[1, 0, 0],[1, 0, 0],[1, 0, 0]]).float()
# print(loss(b, torch.tensor([0, 0, 0])))

loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# print(input,target)
# output = loss(input, target)
# print(output)

# b c d
out = torch.tensor([[[1, 2, 3, 4], [4, 5, 6, 7]], [[1, 2, 3, 4], [4, 5, 6, 7]], [[1, 2, 3, 4], [4, 5, 6, 7]]]).float()
print(out.size())

out = out.softmax(1)

tgt = torch.tensor([[0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1]])

print(out.size(),tgt.size())

print(loss(out, tgt))
