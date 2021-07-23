import pytest
import torch

import cbpy


@pytest.fixture()
def default_params():
    params = {
        'lr': 0.5,
        'max_epoch': 3000,
        'seed': 0,
        'layer_list': [4, 9, 3]
    }
    return params


@pytest.mark.parametrize(('gpu'), [False, True])
def test_xor_cbp_example(default_params, gpu):
    loss_func = torch.nn.CrossEntropyLoss()
    lr = default_params['lr']
    max_epoch = default_params['max_epoch']
    seed = default_params['seed']
    layer_list = default_params['layer_list']

    cbpy.set_random_seed(seed)
    trainloader = cbpy.create_toy_dataloader('iris')
    model = cbpy.MLPS(layer_list)

    zs = [9, 3]
    beta = 0.995
    cbp_epoch = 200
    l_list, a_list, w_list, o_list = cbpy.train_with_chaos(
        model=model,
        trainloader=trainloader,
        testloader=trainloader,
        loss_func=loss_func,
        zs=zs,
        beta=beta,
        cbp_epoch=cbp_epoch,
        max_epoch=max_epoch,
        cbp_lr=lr,
        gpu=gpu
    )

    assert a_list[-1] == 1
    assert l_list[-1] < 0.03


@pytest.mark.parametrize(('gpu'), [False, True])
def test_xor_bp_example(default_params, gpu):
    loss_func = torch.nn.CrossEntropyLoss()
    lr = default_params['lr']
    max_epoch = default_params['max_epoch']
    seed = default_params['seed']
    layer_list = default_params['layer_list']

    cbpy.set_random_seed(seed)
    trainloader = cbpy.create_toy_dataloader('iris')
    model = cbpy.MLPS(layer_list)

    zs = None
    cbp_epoch = 0
    l_list, a_list, w_list, o_list = cbpy.train_with_chaos(
        model=model,
        trainloader=trainloader,
        testloader=trainloader,
        loss_func=loss_func,
        zs=zs,
        cbp_epoch=cbp_epoch,
        max_epoch=max_epoch,
        cbp_lr=lr,
        gpu=gpu
    )

    assert a_list[-1] < 1
    assert l_list[-1] > 0.03
