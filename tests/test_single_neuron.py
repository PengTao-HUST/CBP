import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import cbpy


def test_xor_cbp_example():
    inp = torch.FloatTensor([[1]]) # input sample
    tgt = torch.FloatTensor([[0]]) # target of input
    trainloader = DataLoader(TensorDataset(inp, tgt), 1)

    net = cbpy.MLPS([1, 1], active_last=True, act_layer=torch.nn.Sigmoid(), init_mode=0)
    loss_func = torch.nn.MSELoss() # loss function

    # training api in cbpy
    loss_list, acc_list, weight_list, out_list = cbpy.train_with_chaos(
        model=net, trainloader=trainloader, testloader=trainloader, loss_func=loss_func,
        zs=9, cbp_epoch=1000, cbp_lr=1, beta=0.999)

    assert np.std(loss_list[100:200]) > 0.1