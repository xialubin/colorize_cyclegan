from data import data_loader
from option import option
import torch
opt = option()

data = data_loader(opt.dataset, 'resize', batchsize=8)
for i, batch in enumerate(data):
    print(batch)