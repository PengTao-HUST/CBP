import pytest
import torch

import cbpy


@pytest.fixture()
def default_params():
    params = {
        'lr': 0.01,
        'max_epoch': 5,
        'seed': 0,
        'layer_list': [3072, 1024, 256, 64, 16, 10]
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
    trainloader, testloader = cbpy.create_cifar10_dataloader(100, 100)
    model = cbpy.MLPS(layer_list, active_last=True)

    zs = 0.012
    beta = 0.998
    cbp_epoch = 5
    l_list, a_list, w_list, o_list = cbpy.train_with_chaos(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        loss_func=loss_func,
        zs=zs,
        beta=beta,
        flatten_input=True,
        cbp_epoch=cbp_epoch,
        max_epoch=max_epoch,
        cbp_lr=lr,
        gpu=gpu
    )

    assert a_list[-1] > 0.3
    assert l_list[-1] < 2


@pytest.mark.parametrize(('gpu'), [False, True])
def test_xor_bp_example(default_params, gpu):
    loss_func = torch.nn.CrossEntropyLoss()
    lr = default_params['lr']
    max_epoch = default_params['max_epoch']
    seed = default_params['seed']
    layer_list = default_params['layer_list']

    cbpy.set_random_seed(seed)
    trainloader, testloader = cbpy.create_cifar10_dataloader(100, 100)
    model = cbpy.MLPS(layer_list, active_last=True)

    zs = None
    cbp_epoch = 0
    l_list, a_list, w_list, o_list = cbpy.train_with_chaos(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        loss_func=loss_func,
        zs=zs,
        flatten_input=True,
        cbp_epoch=cbp_epoch,
        max_epoch=max_epoch,
        cbp_lr=lr,
        gpu=gpu
    )

    assert a_list[-1] < 0.3
    assert l_list[-1] > 2
