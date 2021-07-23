import pytest
import torch

import cbpy


@pytest.fixture()
def default_params():
    params = {
        'lr': 0.2,
        'max_epoch': 10000,
        'seed': 32,
        'init_mode': 1,
        'layer_list': [2, 2, 1]
    }
    return params


@pytest.mark.parametrize(('gpu'), [False, True])
def test_xor_cbp_example(default_params, gpu):
    trainloader = cbpy.create_xor_dataloader()
    loss_func = torch.nn.BCELoss()

    lr = default_params['lr']
    max_epoch = default_params['max_epoch']
    seed = default_params['seed']
    init_mode = default_params['init_mode']
    layer_list = default_params['layer_list']

    cbpy.set_random_seed(seed)
    model = cbpy.MLPS(layer_list, init_mode=init_mode, act_layer=torch.nn.Sigmoid(), active_last=True)

    zs = 12
    cbp_epoch = max_epoch
    l_list, a_list, w_list, o_list = cbpy.train_with_chaos(
        model=model,
        trainloader=trainloader,
        testloader=trainloader,
        loss_func=loss_func,
        zs=zs,
        record_weight=True,
        whole_weight=True,
        cbp_epoch=cbp_epoch,
        cbp_lr=lr,
        gpu=gpu
    )

    assert a_list[-1] == 1
    assert l_list[-1] < 0.35


@pytest.mark.parametrize(('gpu'), [False, True])
def test_xor_bp_example(default_params, gpu):
    trainloader = cbpy.create_xor_dataloader()
    loss_func = torch.nn.BCELoss()

    lr = default_params['lr']
    max_epoch = default_params['max_epoch']
    seed = default_params['seed']
    init_mode = default_params['init_mode']
    layer_list = default_params['layer_list']

    cbpy.set_random_seed(seed)
    model = cbpy.MLPS(layer_list, init_mode=init_mode, act_layer=torch.nn.Sigmoid(), active_last=True)

    zs = None
    cbp_epoch = 0
    l_list, a_list, w_list, o_list = cbpy.train_with_chaos(
        model=model,
        trainloader=trainloader,
        testloader=trainloader,
        loss_func=loss_func,
        zs=zs,
        record_weight=True,
        whole_weight=True,
        cbp_epoch=cbp_epoch,
        max_epoch=max_epoch,
        bp_lr=lr,
        gpu=gpu
    )

    assert a_list[-1] == 0.5
    assert l_list[-1] > 0.35
