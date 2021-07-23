import torch.nn as nn


__all__ = ['MLPS']


def _my_net_init(layers, init_mode=None, zero_bias=False):
    """ initial the weight in the network. """
    for m in layers:
        if init_mode is not None:
            try:
                init_mode = float(init_mode)
                n = abs(init_mode)
                nn.init.uniform_(m.weight, -n, n)
                nn.init.uniform_(m.bias, -n, n)
            except (ValueError, TypeError):
                if isinstance(init_mode, str):
                    assert init_mode in ['xavier', 'kaiming']
                    if init_mode == 'xavier':
                        nn.init.xavier_uniform_(m.weight)
                    else:
                        nn.init.kaiming_uniform_(m.weight)
                else:
                    raise ValueError(f'Unknown init mode: {init_mode}.')
        if zero_bias:
            nn.init.zeros_(m.bias)


class MLPS(nn.Module):
    """ create the MLP model from a list of layer. """
    def __init__(self, layer_list, act_layer=nn.ReLU(),
                 init_mode=None, zero_bias=False, active_last=False):
        super(MLPS, self).__init__()
        self.layers = []
        for in_d, out_d in zip(layer_list[:-1], layer_list[1:]):
            self.layers.append(nn.Linear(in_d, out_d))
        self.n_layer = len(self.layers)
        self.layer_list = layer_list
        self.model = nn.Sequential(*self.layers)
        self.hid_act = nn.Sigmoid()
        self.fwd_act = act_layer
        self.active_last = active_last

        _my_net_init(self.layers, init_mode=init_mode, zero_bias=zero_bias)

    def forward(self, x, return_hid=False):
        inp_x = x
        hids = []
        for i, layer in enumerate(self.layers):
            out_x_ = layer(inp_x)
            if i != (self.n_layer - 1) or self.active_last:
                out_x = self.fwd_act(out_x_)
            else:
                out_x = out_x_
            inp_x = out_x

            if return_hid:
                hid = self.hid_act(out_x_).detach().mean(0)
                hids.append(hid.view(-1, 1))
                hids.append(hid)

        return hids, out_x

